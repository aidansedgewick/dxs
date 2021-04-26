import json
import logging
import os
import warnings
import yaml
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS, FITSFixedWarning

import reproject as rpj
from astromatic_wrapper.api import Astromatic
from photutils.background import SExtractorBackground, Background2D

from dxs.utils.image import build_mosaic_header
from dxs.utils.misc import (
    check_modules, format_flags, get_git_info, AstropyFilter
)

from dxs import paths

logger = logging.getLogger("mosaic_builder")
astropy_logger = logging.getLogger("astropy")
astropy_logger.addFilter(AstropyFilter())
#warnings.simplefilter(action="once", category=FITSFixedWarning)

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

neighbors_path = paths.config_path / "dxs_arrangement.yaml"
with open(neighbors_path) as f:
    neighbors_config = yaml.load(f, Loader=yaml.FullLoader)
    neighbors_config["arrangement_to_tile"] = {}
    for field, arrangement in neighbors_config["arrangement"].items():
        neighbors_config["arrangement_to_tile"][field] = {
            tuple(v): k for k, v in arrangement.items()
        }

def read_header_data(header_data_path=paths.header_data_path):
    if not header_data_path.exists():
        return None
    return pd.read_csv(header_data_path)
header_data = read_header_data()

class MosaicBuilderError(Exception):
    pass

class MosaicBuilder:
    """
    Class for building mosaics. Calls SWarp.
    """

    def __init__(
        self, 
        mosaic_stacks, 
        mosaic_path,
        neighbor_stacks=None,
        swarp_config=None, 
        swarp_config_file=None,
        hdu_prep_kwargs=None
    ):
        if isinstance(mosaic_stacks, pd.DataFrame):
            self.mosaic_stacks = mosaic_stacks
            if neighbor_stacks is not None:
                assert isinstance(neighbor_stacks, pd.DataFrame)
                relevant_stacks = pd.concat([mosaic_stacks, neighbor_stacks])
            else:
                relevant_stacks = mosaic_stacks
            logger.info("stacks to use: \n")
            print(relevant_stacks.groupby(["field", "tile", "pointing"]).size())          
            self.stack_list = [
                paths.stack_data_path / f"{x}.fit" for x in relevant_stacks["filename"]
            ]
            if "ccds" in relevant_stacks.columns:
                self.ccds_list = [x for x in relevant_stacks["ccds"]] # Unnecessary?
            else:
                self.ccds_list = [None for _ in range(len(relevant_stacks))]
            
        elif isinstance(mosaic_stacks, list):
            self.mosaic_stacks = None # Assume that the input list is NOT with seeing, etc.
            self.stack_list = mosaic_stacks # 
            if neighbor_stacks is not None:
                raise MosaicBuilderError(
                    "to use BOTH, mosaic_stacks and neighbor_stacks must be pd.DataFrame (with \"filename\" column"
                )
            self.ccds_list = [None for _ in range(len(mosaic_stacks))]                

        self.mosaic_path = Path(mosaic_path)
        logger.info(f"output at {self.mosaic_path.name}")
        self.swarp_config = swarp_config or {}
        self.swarp_config_file = swarp_config_file or paths.config_path / "swarp/mosaic.swarp"
        self.hdu_prep_kwargs = hdu_prep_kwargs or {}    

        self.mosaic_dir = self.mosaic_path.parent
        self.mosaic_dir.mkdir(exist_ok=True, parents=True)

        self.aux_dir = self.mosaic_dir / "aux"
        self.aux_dir.mkdir(exist_ok=True, parents=True)

        string_name = str(self.mosaic_path.stem).replace(".", "_")
        self.swarp_list_path = self.aux_dir / f"{string_name}_swarp_list.txt"
        self.swarp_xml_path = self.aux_dir / f"{string_name}.xml"
        self.swarp_run_parameters_path = (
            self.aux_dir / f"{string_name}_swarp_run_parameters.json"
        )

    @classmethod
    def from_dxs_spec(
        cls, field, tile, band, 
        include_neighbors=True,
        prefix=None,
        suffix=None,
        extension=None,
        add_flux_scale=True,
        swarp_config=None, 
        swarp_config_file=None, 
        hdu_prep_kwargs=None,
    ):
        """provide a field, tile (as integer) and band.
        eg.

        >>> prep_kwargs = {}
        >>> builder = mosaic_builder.from_dxs_spec("EN", 4, "K", hdu_prep_kwargs=prep_kwargs)
        >>> print(builder.mosaic_path) # mosaic will be saved here.
        >>> builder.build(n_cpus=4)      

        Parameters
        ----------
        field
        tile
        band
        prefix
            prefix.suffix give mosaic name eg. [prefix]EN04K[suffix].fits
        suffix
        extension
            defaults to ".fits" - eg. EN04K.fits    
        add_flux_scale
            whether to scale the flux of each input stack to a common zeropoint.
        swarp_config
            dict overwrites values in swarp_config_file. c
            eg swarp_config={"detect_sigma": 5.0", "size": (1000,1000)}
            floats, ints, tuples are formatted correctly internally.
        swarp_config_file
            path to a swarp config file. by default .../dxs/configuration/mosaic.swarp
        """
        mosaic_dir = paths.get_mosaic_dir(field, tile, band)
        mosaic_stem = paths.get_mosaic_stem(field, tile, band, prefix=prefix, suffix=suffix)
        extension = f".{extension}" if extension is not None else ""
        mosaic_path = mosaic_dir / f"{mosaic_stem}{extension}.fits"
        swarp_config = swarp_config or {}
        hdu_prep_kwargs = hdu_prep_kwargs or {}
        logger.info(f"mosaic for {field} {tile} {band}")

        # Get a list of all the stacks in this tile (ignoring band) for geometry.
        geom_mosaic_stacks = get_stack_data(field, tile, band=None)
        if len(geom_mosaic_stacks) == 0:
            logger.info(f"No stacks in {field}, {tile}, in any band")
            return None
        geom_stack_list = [
            paths.stack_data_path / f"{x}.fit" for x in geom_mosaic_stacks["filename"]
        ]
        border = survey_config["mosaics"].get("border", None)
        factor = survey_config["mosaics"].get("factor", None)
        if "center" not in swarp_config or "image_size" not in swarp_config:
            center, size = calculate_mosaic_geometry(
                geom_stack_list, ccds=survey_config["ccds"],
                pixel_scale=swarp_config.get("pixel_scale", None),
                border=border,
                factor=factor,
            )
            swarp_config["center"] = center #f"{center[0]:.6f},{center[1]:.6f}"
            swarp_config["image_size"] = size #f"{size[0]},{size[1]}"
        swarp_config["center_type"] = "MANUAL"
        swarp_config["pixelscale_type"] = "MANUAL"
        # which stacks should go into the swarp?
        mosaic_stacks = get_stack_data(field, tile, band)
        if len(mosaic_stacks) == 0:
            logger.info(f"No stacks in {field} {tile} {band} to build.")
            return None
        if include_neighbors:
            mosaic_stacks["ccds"] = [
                survey_config["ccds"] for _ in range(len(mosaic_stacks))
            ]
            neighbor_stacks = get_neighbor_stacks(field, tile, band)
            logger.info(f"including neighbors {len(neighbor_stacks)}")
        else:
            neighbor_stacks = None

        #=========== sort some things for preparing ============#
        normalise_exptime = hdu_prep_kwargs.get("normalise_exptime", False) # check default in HDUPreparer
        if normalise_exptime:
            magzpt_inc_exptime = False # ie, if counts/s img, no need to account!
        else:
            magzpt_inc_exptime = True
        hdu_prep_kwargs["reference_magzpt"] = cls.calc_magzpt(
            mosaic_stacks, magzpt_inc_exptime=magzpt_inc_exptime
        )
        hdu_prep_kwargs["add_flux_scale"] = add_flux_scale
        return cls(
            mosaic_stacks, 
            mosaic_path, 
            neighbor_stacks=neighbor_stacks,
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file,
            hdu_prep_kwargs=hdu_prep_kwargs
        )

    @classmethod
    def coverage_from_dxs_spec(
        cls, field, tile, band, 
        pixel_scale, 
        include_neighbors=True,
        prefix=None,
        suffix=None,
        swarp_config=None, 
        swarp_config_file=None, 
        hdu_prep_kwargs=None,
    ):
        """
        similar to from_dxs_spec but uses coverage.swarp config, 
        option to include pixel scale
        fill_value should be set to 1
        """
        swarp_config = swarp_config or {}
        coverage_config = {
            "pixel_scale": pixel_scale,
        }
        swarp_config_file = swarp_config_file or paths.config_path / "swarp/coverage.swarp"
        swarp_config.update(coverage_config)
        hdu_prep_kwargs = hdu_prep_kwargs or {}
        hdu_prep_kwargs["fill_value"] = 1.0
        return cls.from_dxs_spec(
            field, tile, band, 
            prefix=prefix, 
            suffix=suffix, 
            include_neighbors=include_neighbors,
            extension="cov", 
            add_flux_scale=False,
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file,
            hdu_prep_kwargs=hdu_prep_kwargs,
        )

    def build(self, prepare_hdus=True, n_cpus=None, **kwargs):
        """
        Parameters
        ----------
        stack_list
            list of stacks to build mosaic from
        prepare_hdus
            bool, whether to prepare the hdus or not.
        n_cpus
            defaults to whatever is passed at init (or None).
        kwargs
            kwargs that are passed to HDUPreparer.prepare_stack() 
        """
        check_modules("swarp") # do we have swarp available?
        if len(kwargs) > 0:
            logger.warn(
                "please pass all hdu_prep_kwargs at initialisation as "
                "hdu_prep_kwargs=<dict of prep kwargs>"
            )
            self.hdu_prep_kwargs.update(kwargs)
        if prepare_hdus:
            hdu_list = self.prepare_all_hdus(
                self.stack_list, n_cpus=n_cpus, **self.hdu_prep_kwargs
            )
            self.write_swarp_list(hdu_list)
        else:
            self.write_swarp_list(self.stack_list)
        self.initialise_astromatic(n_cpus=n_cpus)
        swarp_list_name = "@"+str(self.swarp_list_path)
        self.swarp.run(swarp_list_name)
        logger.info(f"Mosaic written to {self.mosaic_path}")
        with open(self.swarp_run_parameters_path, "w+") as f:
            json.dump(self.cmd_kwargs, f, indent=2)
       
    def prepare_all_hdus(self, stack_list=None, n_cpus=None, **kwargs):
        """
        Parameters
        ----------
        stack_list
            a list of (Path-like) paths to stacks to prepare HDUs for.
        kwargs
            key-word arguments accepted by HduPreparer(), and "prefix".
        """
        if stack_list is None:
            stack_list = self.stack_list
        if n_cpus is None:    
            hdu_list = []
            for ii, stack_path in enumerate(stack_list):
                results = HDUPreparer.prepare_stack(
                    stack_path, ccds=self.ccds_list[ii], **kwargs
                )
                hdu_list.extend(results)
        else:
            # can do many threads of stacks...
            kwarg_list = [dict(**kwargs, ccds=ccds) for ccds in self.ccds_list]            
            arg_list = [(sp, kw) for sp, kw in zip(stack_list, kwarg_list)]
            # This is a bit ugly, but essentially we need to make a single object (tuple)
            # that we can give to a single-arg function for pool.map(). 
            with Pool(n_cpus) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.map(_hdu_prep_wrapper, arg_list), total=len(arg_list)
                    )    
                )
                #results = pool.map(_hdu_prep_wrapper, arg_list)
                hdu_list = [p for result in results for p in result]
        for hdu_path in hdu_list:
            assert hdu_path.exists()
        return hdu_list

    def write_swarp_list(self, hdu_list):
        """
        SWARP does something very odd. it seems to IGNORE the first input file.
        so we add a tiny file with not much data as the first file
        """
        tiny_header = build_mosaic_header(
            center=self.swarp_config["center"], size=(2,2), pixel_scale=0.2
        )
        tiny_data = np.random.uniform(0, 1, (2,2))
        tiny_hdu = fits.PrimaryHDU(data=tiny_data, header=tiny_header)
        tiny_hdu_path = paths.temp_hdus_path / f"{self.mosaic_path.stem}_tiny_data.fits"
        tiny_hdu.writeto(tiny_hdu_path, overwrite=True)
        
        with open(self.swarp_list_path, "w+") as f:
            hdu_path_list = (
                [str(tiny_hdu_path)+"\n"] + [str(hdu_path)+"\n" for hdu_path in hdu_list]
            )
            f.writelines(hdu_path_list)        
    
    def initialise_astromatic(self, n_cpus=None):
        config = self.build_swarp_config()
        config["nthreads"] = n_cpus or 1    
        config.update(self.swarp_config)
        #if not os.isatty(0):
        #    config["verbose_type"] = "quiet" # If running on a batch, prevent spewing output.
        config = format_flags(config)
        self.swarp = Astromatic(
            "SWarp", 
            str(paths.temp_swarp_path), # I think this is ignored by Astromatic() anyway?!
            config=config, 
            config_file=str(self.swarp_config_file),
            store_output=True,
        )
        swarp_list_name = "@"+str(self.swarp_list_path)
        logger.info("build - starting swarp")
        cmd, cmd_kwargs = self.swarp.build_cmd(swarp_list_name)
        self.cmd_kwargs = cmd_kwargs
        self.cmd_kwargs["cmd"] = cmd

    def build_swarp_config(self,):
        config = {}
        config["imageout_name"] = self.mosaic_path
        weightout_name = self.mosaic_dir / f"{self.mosaic_path.stem}.weight.fits"
        config["weightout_name"] = weightout_name
        config["resample_dir"] = paths.temp_swarp_path
        config["pixel_scale"] = survey_config["mosaics"].get("pixel_scale", 0.2) #f"{pixel_scale:.6f}" 
        config["pixelscale_type"] = "manual"       
        config["xml_name"] = self.swarp_xml_path
        return config

    def add_extra_keys(self, keys=None, magzpt_inc_exptime=True):
        keys = keys or {}
        if self.mosaic_stacks is not None:
            seeing_val = self.calc_seeing(self.mosaic_stacks)
            keys["seeing"] = (seeing_val, "median seeing of stacks, in arcsec")
            includes = "does" if magzpt_inc_exptime else "does not"
            magzpt_val = self.calc_magzpt(
                self.mosaic_stacks, magzpt_inc_exptime=magzpt_inc_exptime
            )
            keys["magzpt"] = (magzpt_val,  f"median; {includes} inc. 2.5log(t_exp)")
        branch, local_sha = get_git_info()
        keys["branch"] = (branch, "pipeline branch")
        keys["localSHA"] = (local_sha.replace("'",""), "local pipeline SHA")
        keys["pipevers"] = (0.0, "pipeline version")
        add_keys(self.mosaic_path, keys, hdu=0, verbose=True)

    @staticmethod
    def calc_seeing(stack_data):
        """
        Median seeing value 
        this DOESN'T include any seeing values for neighbor stacks, as they're
        only on the very edges
        """
        seeing_cols = [f"seeing_{ccd}" for ccd in survey_config["ccds"]]
        seeing_df = stack_data[ seeing_cols ]
        seeing = np.median(seeing_df.stack().values)
        return seeing

    @staticmethod
    def calc_magzpt(stack_data, magzpt_inc_exptime=True):
        """
        If counts per second (ie, exptime = 1), DON't include exptime.
        Else: we'll have T times more flux per obeject. so need to add 2.5log(expT)
        """
        magzpt_cols = [f"magzpt_{ccd}" for ccd in survey_config["ccds"]]
        magzpt_df = pd.DataFrame() # Try to avoid setting with copy warning...!
        if magzpt_inc_exptime:
            exptime_col = 2.5*np.log10(stack_data["exptime"])
        else:
            exptime_col = 0.
        for col in magzpt_cols:
            magzpt_df[col] = stack_data[col] + exptime_col
        magzpt = np.median(magzpt_df.stack().values)
        logger.info(f"magzpt = {magzpt}; inc. exptime: {magzpt_inc_exptime}")
        return magzpt

class HDUPreparer:
    """
    Class for preparing a single HDU ready for SWarp.

    eg.

    >>> hdup = HDUPreparer(hdu, "./hdu_out.fits")
    >>> hdup.prepare_hdu()

    Parameters
    ----------
    hdu
        a single astropy hdu object
    hdu_path
        str or PathLike - where will this hdu be saved? after prep
    fill_value
        float- fill the entire data array with a value close to this (+/- small ~1e-4 random noise)
        bc. swarp doesn't like array of the same value. useful for coverage maps( value = 1.0)
        or exposure time maps.
    resize
        bool - if true, removes `edges` pixels from each hdu in prep. default False.
    edges
        float or int - removes this many pixels from each edge, if `resize=True`
    mask_sources
        bool - if True, ignore parts of `hdu.data` when estimating background. default False
    mask_header
        hdu header object. use header rather than directly passing `astropy.wcs` object  
        as `wcs` seems to go weird in combination with multiprocessing Pool.
    mask_map
        array with shape equal to `hdu.shape`. Non-zeros values are sources to mask.
        (ie, can use SExtractor segmentation check image as mask).
    normalize_exptime
        bool - divide data array by the value provided in `exptime`. Default False.
        Recommend only use with `subtract_bgr=True`...!
    exptime
        float - removed soon... should read exptime from `hdu.header`
    subtract_bgr
        bool
    bgr_size
        int - box size to estimate bgr in. Used as `photutils.Background2d(box_size=(bgr_size, bgr_size))`
    filter_size
        int - Used as `photutils.Background2d(filter_size=filter_size)`
    sigma
        float - Used as for sigma clipping in bgr_estimation. see `photutils`.
    add_flux_scale
        bool - if True, add `flxscale` to output header, calculated as 
        10**(-0.4*(magzpt-reference_magzpt)), if `reference_magzpt` is provided. 
    reference_magzpt
        used to calculate flxscale. add 2.5*log10(exptime) if `normalise_exptime` is True. 
        default None. 
    """



    def __init__(
        self, hdu, hdu_path,
        fill_value=None, fill_value_var=1e-3,
        resize=False, edges=25.0, 
        mask_sources=False, mask_header=None, mask_map=None,
        normalise_exptime=False, exptime=None, # check from_dxs_spec hdu_prep_kwargs...
        subtract_bgr=False, bgr_size=None, filter_size=1, sigma=3.0,
        add_flux_scale=True, reference_magzpt=None
    ):
        self.data = hdu.data
        if fill_value is not None:
            self.data = np.random.normal(
                fill_value, fill_value_var, hdu.data.shape
            )
            # slight variations from fill_value - swarp doesn't like HDU full of constant.
        self.header = hdu.header
        self.hdu_path = Path(hdu_path)
        self.resize = resize
        self.edges = edges
        self.mask_sources = mask_sources
        if mask_header is not None:
            self.mask_wcs = WCS(mask_header)
        else:
            self.mask_wcs = None
        # construct wcs inside init func, else it goes weird with Pool?
        self.mask_map = mask_map
        self.normalise_exptime = normalise_exptime
        self.magzpt = self.header.get("MAGZPT", None)
        if self.magzpt is None:
            logger.warn(f"MAGZPT for {self.hdu_path.stem} not found - set to reference {reference_magzpt}")
            self.magzpt = reference_magzpt
        if normalise_exptime is False:
            self.magzpt = self.magzpt + 2.5*np.log10(exptime) # log10!!!!
        self.exptime = exptime
        self.subtract_bgr = subtract_bgr
        self.bgr_size = bgr_size
        self.sigma = sigma
        self.add_flux_scale = add_flux_scale
        self.reference_magzpt = reference_magzpt

        self.hdu_wcs = WCS(hdu.header)
        self.hdu_wcs.fix()
        self.xlen = hdu.header["NAXIS1"]
        self.ylen = hdu.header["NAXIS2"]

        self.center = (self.ylen // 2, self.xlen // 2)
        self.center_coords = SkyCoord.from_pixel(
            xp=self.center[1], yp=self.center[0], wcs=self.hdu_wcs, origin=1
        )

    def modify_header(self):
        if self.add_flux_scale:
            if self.reference_magzpt is None:
                raise ValueError(
                    "Can't add flux scale if no reference_magzpt to scale to!\n"
                    "Either provide reference_magzpt= or set add_flux_scale=False"
                )
            else:
                mag_diff = self.magzpt - self.reference_magzpt
                flux_scaling = 10**(-0.4*mag_diff)
                self.header["FLXSCALE"] = flux_scaling
                logger.info(f"flxscl {self.hdu_path.stem} {flux_scaling:.2f}")
        self.header["EXP_TIME"] = self.exptime

    def prepare_hdu(self):
        if self.mask_wcs is not None:
            hdu_footprint = SkyCoord(self.hdu_wcs.calc_footprint(), unit="degree")
            contains = self.mask_wcs.footprint_contains(hdu_footprint)
            if not any(contains):
                logger.info(f"{self.hdu_path.stem} not in mask")
        if self.mask_sources:
            try:
                source_mask = self.get_source_mask()
            except Exception as e:
                logger.warn(e)
                return None
        else:
            source_mask = None
        if self.subtract_bgr:
            bgr = self.get_background(source_mask=source_mask)
            self.data = self.data - bgr.background
        if self.normalise_exptime:
            if self.subtract_bgr is False:
                warn_msg = (
                    "are you sure it's a good idea to normalise exposure time "
                    + "without subtracting the background...?!"
                )
                logger.warn(warn_msg)
            self.data = self.data / self.exptime
        if self.resize:
            cutout_size = (int(self.ylen-2*self.edges), int(self.xlen-2*self.edges))
            cutout = Cutout2D(
                self.data, position=self.center, size=cutout_size, wcs=self.hdu_wcs
            )
            self.data = cutout.data
            self.header.update(cutout.wcs.to_header())
        self.modify_header()
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        if self.hdu_path.exists():
            logger.warn(f"{self.hdu_path.stem} is being overwritten!")
        hdu.writeto(self.hdu_path, overwrite=True)
        return True

    def get_source_mask(self):
        approx_size = (self.ylen + 50, self.xlen + 50)
        # TODO: do better by taking the max, min of the stack footprint in mask wcs pix?
        apx_map = Cutout2D(
            self.mask_map, 
            position=self.center_coords,
            size=approx_size,
            wcs=self.mask_wcs,
            mode="partial", # definitely want PARTIAL. 
            fill_value=0 # ensures that outside values will be not masked.
        )
        reprojected_map, reprojected_footprint = rpj.reproject_interp(
            (apx_map.data, apx_map.wcs), 
            output_projection=self.header, 
            order="nearest-neighbor"
        )
        assert reprojected_map.shape == self.data.shape
        mask = reprojected_map
        mask = mask.astype(bool) # Background2D expects True for masked pixels.
        return mask

    def get_background(self, source_mask=None):
        sigma_clip = SigmaClip(sigma=self.sigma)
        estimator = SExtractorBackground(sigma_clip)
        bgr_map = Background2D(
            self.data, 
            mask=source_mask, 
            sigma_clip=sigma_clip, 
            bkg_estimator=estimator,
            box_size=(self.bgr_size, self.bgr_size),
            filter_size=(1,1)
        )
        return bgr_map

    @classmethod
    def prepare_stack(
        cls, stack_path, overwrite=False, hdu_prefix=None, ccds=None, **kwargs
    ):
        """
        class method - prepare a whole stack (ie, multi-extension fits) of HDUs.
        returns a list of paths to prepared HDUs.

        Parameters
        ----------
        stack_path
            path to [multi-]extension fits that will be prep'ed.
        overwrite
            bool: if False, and this HDU has already been prepared, skip it.
        hdu_prefix
            name of hdu will be <hdu_prefix><stack_path.stem>_<ccd>.fits
        ccds
            list: which ccds (ie, fits extensions) in the stack to prepare? defaults to
            whatever is in survey_config ccds.
        **kwargs
            any kwargs which go into initialisation of HDUPreparer.
        """

        stack_path = Path(stack_path)
        results = []
        ccds = ccds or survey_config["ccds"]
        with fits.open(stack_path) as f:
            try:
                objsplit = f[0].header["OBJECT"].split()
                spec = objsplit[2] + f[0].header["FILTER"] + objsplit[3]
            except:
                spec = ""
            for ii, ccd in enumerate(ccds):
                stack_str = stack_path.stem + spec
                hdu_name = get_hdu_name(stack_str, ccd, prefix=hdu_prefix)
                hdu_path = paths.temp_hdus_path / hdu_name # includes ".fits" already...
                logger.info(f"prep {spec} {hdu_path.stem}")
                if not overwrite and hdu_path.exists():
                    results.append(hdu_path)
                    continue
                p = cls(f[ccd], hdu_path, exptime=f[0].header["EXP_TIME"], **kwargs)
                result = p.prepare_hdu()
                if result is not None:
                    results.append(hdu_path)
        return results


##======== Aux. functions for HDUPreparer

def _hdu_prep_wrapper(arg):
    args, kwargs = arg
    return HDUPreparer.prepare_stack(args, **kwargs)    

def get_hdu_name(stack_str, ccd, weight=False, prefix=None):
    weight = ".weight" if weight else ""
    prefix = prefix or ""
    try:
        stack_str = stack_str.stem
    except:
        stack_str = str(stack_str)
    return f"{prefix}{stack_str}_{ccd:02d}{weight}.fits"


##======== Finding out which stacks are important for a mosaic.

def get_stack_data(field, tile, band, pointing=None):
    """
    Data frame of info for stacks in a given field/tile/band [optionally pointing].
    Must provide, field, tile, band as args - although tile and band can be None, or a list.
    """
    if header_data is None:
        raise IOError("Run dxs/setup_scripts/extract_header_info.py first")
    queries = [f"(field=='{field}')"]
    if tile is not None:
        if not isinstance(tile, list):
            tile = [tile]
        queries.append(f"(tile in @tile)")
    if band is not None:
        if not isinstance(band, list):
            band = [band]
        queries.append(f"(band in @band)")
    if pointing is not None:
        if not isinstance(pointing, list):
            pointing = [pointing]
        queries.append(f"(pointing in @pointing)")    
    query = "&".join(q for q in queries)
    return header_data.query(query)

def get_neighbor_stacks(field, tile, band):
    """
    Which stacks are on the border of the mosaic we're SWarping? 
    We should add these in to make extracting the catalog nicer.
    """
    neighbors = get_neighbor_tiles(field, tile)
    df_list = []
    for neighbor_tile, cardinal in neighbors.items():
        hdus_for_cardinal = neighbors_config["border_hdus"][cardinal]
        for pointing, ccds in hdus_for_cardinal.items():
            stacks = get_stack_data(field, neighbor_tile, band, pointing=pointing)
            stacks["ccds"] = [ccds for _ in range(len(stacks))]
            df_list.append(stacks)
    neighbor_stacks = pd.concat(df_list)
    return neighbor_stacks

def get_neighbor_tiles(field, tile):
    pos = neighbors_config["arrangement"][field][tile]
    neighbor_lookup = {}
    for cardinal, relative in neighbors_config["directions"].items():
        exact_neighbor = get_new_position(pos, relative)
        field_arrangement = neighbors_config["arrangement_to_tile"][field]
        neighbor_tile = field_arrangement.get(exact_neighbor, None)
        if neighbor_tile is not None:
            # ie, if it has an exact neighbor.
            neighbor_lookup[neighbor_tile] = cardinal
            continue
        for neighbor_tile, offset_neighbor in neighbors_config["arrangement"][field].items(): 
            correct_NS = offset_neighbor[1] == exact_neighbor[1]
            correct_EW = offset_neighbor[0] == exact_neighbor[0]
            offset_NS = abs(offset_neighbor[0] - exact_neighbor[0]) < 1
            offset_EW = abs(offset_neighbor[0] - exact_neighbor[0]) < 1
    
            if cardinal in ["N", "S"] and correct_NS and offset_EW:
                neighbor_lookup[neighbor_tile] = cardinal
            elif cardinal in ["E", "W"] and correct_EW and offset_NS:
                neighbor_lookup[neighbor_tile] = cardinal
        
    return neighbor_lookup

def get_new_position(start, move):
    return (start[0] + move[0], start[1] + move[1])


###=================== geometry ==================###

def calculate_mosaic_geometry(
    stack_list, ccds=None, factor=None, border=None, pixel_scale=None
):
    """
    How big should the mosaic be to fit all the stacks in stack_list?
    
    Parameters
    ----------
    stack_list
        list of fits files
    ccds
        hdu extensions to consider (default [1,2,3,4] for dxs)
    factor
        scale the mosaic size (in each dimension) by this factor (default 1.0)
    border
        add/remove this many border pixels (default 0)
    pixel_scale
        in arcsec/pix (defaults reads from configuration/survey_config.yaml)
    
    Returns two tuples: centre (ra, dec) in degrees, and mosaic_size (x, y) in pxiels.
    """
    
    ## TODO fix wrap-around 360 values.

    logger.info("Calculating mosaic geometry")
    ccds = ccds or [0]
    ra_values = []
    dec_values = []
    for ii, stack_path in enumerate(stack_list):
        with fits.open(stack_path) as f:
            for ccd in ccds:
                hdu_wcs = WCS(f[ccd].header)
                footprint = hdu_wcs.calc_footprint()
                ra_values.extend(footprint[:,0])
                dec_values.extend(footprint[:,1])
    ra_limits = (np.min(ra_values), np.max(ra_values))
    dec_limits = (np.min(dec_values), np.max(dec_values))

    # center is easy.
    center = SkyCoord(ra=np.mean(ra_limits), dec=np.mean(dec_limits), unit="degree")
    # image size takes a bit more thought because of spherical things.
    cos_dec = np.cos(center.dec * np.pi / 180.)
    pixel_scale = pixel_scale or survey_config["mosaics"]["pixel_scale"]
    plate_factor = 3600. / pixel_scale
    x_size = abs(ra_limits[1] - ra_limits[0]) * plate_factor * cos_dec
    y_size = abs(dec_limits[1] - dec_limits[0]) * plate_factor
    mosaic_size = (int(x_size), int(y_size))
    if factor is not None:
        mosaic_size = (mosaic_size[0]*factor, mosaic_size[1]*factor)
    if border is not None:
        mosaic_size = (mosaic_size[0]+border, mosaic_size[1]+border)
    max_size = [int(x) for x in survey_config["mosaics"]["max_size"]]
    if mosaic_size[0] > max_size[0] or mosaic_size[1] > max_size[1]:
        raise MosaicBuilderError(
            f"Mosaic too large: {mosaic_size[0]},{mosaic_size[1]}"
            f" larger than {max_size[0],max_size[1]}"
            f" \n check configuration/survey_config.yaml"
        )
    logger.info(f"geom - size {mosaic_size[0]},{mosaic_size[1]}")
    logger.info(f"geom - center {center.ra:.3f}, {center.dec:.3f}")
    return center, mosaic_size




###============= misc? ================###

def add_keys(mosaic_path, data, hdu=0, verbose=False):
    with fits.open(mosaic_path, mode="update") as mosaic:
        for key, val in data.items():
            if verbose:
                logger.info(f"add keys - {mosaic_path.name} update {key} to {val}")
            #mosaic[hdu].header[key.upper()] = val
            if not isinstance(val, tuple):
                val = (val, "")
            try:
                mosaic[hdu].header.set(key.upper(), *val)
            except:
                logger.info(f"Could not write {key}, {val}")
                pass
        mosaic.flush()

if __name__ == "__main__":
    print("no main block")

    









