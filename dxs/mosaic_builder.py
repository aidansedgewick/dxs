import json
import logging
import warnings
import yaml
from functools import partial
from multiprocessing import Pool
from pathlib import Path

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

#from dxs.utils.image import 
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

neighbors_path = paths.config_path / "dxs_positions.yaml"
with open(neighbors_path) as f:
    neighbors_config = yaml.load(f, Loader=yaml.FullLoader)
    neighbors_config["arrangement_to_tile"] = {}
    for field, arrangement in neighbors_config["arrangement"].items():
        neighbors_config["arrangement_to_tile"][field] = {
            tuple(v): k for k, v in arrangement.items()
        }

stack_data = pd.read_csv(paths.header_data_path)

import matplotlib.pyplot as plt

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
        swarp_config_file= None,
    ):
        check_modules("swarp") # do we have swarp available?
        if isinstance(mosaic_stacks, pd.DataFrame):
            self.mosaic_stacks = mosaic_stacks
            if neighbor_stacks is not None:
                assert isinstance(neighbor_stacks, pd.DataFrame)
                relevant_stacks = pd.concat([mosaic_stacks, neighbor_stacks])
            else:
                relevant_stacks = mosaic_stacks
            print(relevant_stacks.groupby(["tile", "pointing"]).size())
            self.stack_list = [
                paths.stack_data_path / f"{x}.fit" for x in relevant_stacks["filename"]
            ]
            if "ccds" in relevant_stacks.columns:
                self.ccds_list = [x for x in relevant_stacks["ccds"]] # Unnecessary?
            else:
                self.ccds_list = [None for _ in range(len(relevant_stacks))]
            
        elif isinstance(mosaic_stacks, list):
            self.stack_list = mosaic_stacks
            if neighbor_stacks is not None:
                raise MosaicBuilderError(
                    "to use BOTH, mosaic_stacks and neighbor_stacks must be pd.DataFrame (with \"filename\" column"
                )
            self.ccds_list = [None for _ in range(len(mosaic_stacks))]                

        self.mosaic_path = Path(mosaic_path)
        self.swarp_config = swarp_config or {}
        self.swarp_config_file = swarp_config_file or paths.config_path / "swarp/mosaic.swarp"
    
        self.mosaic_dir = self.mosaic_path.parent
        self.mosaic_dir.mkdir(exist_ok=True, parents=True)

        string_name = str(self.mosaic_path.stem).replace(".", "_")
        self.swarp_list_path = self.mosaic_dir / f"{string_name}_swarp_list.txt"
        self.swarp_run_parameters_path = (
            self.mosaic_dir / f"{string_name}_swarp_run_parameters.json"
        )

    @classmethod
    def from_dxs_spec(
        cls, field, tile, band, 
        include_neighbors=True,
        prefix=None, 
        extension=None, 
        swarp_config=None, 
        swarp_config_file=None, 
    ):
        mosaic_dir = paths.get_mosaic_dir(field, tile, band)
        mosaic_stem = paths.get_mosaic_stem(field, tile, band, prefix=prefix)
        extension = f".{extension}" if extension is not None else ""
        mosaic_path = mosaic_dir / f"{mosaic_stem}{extension}.fits"
        swarp_config = swarp_config or {}

        # Get a list of all the stacks in this tile (ignoring band) for geometry.
        geom_mosaic_stacks = get_stack_data(field, tile, band=None)
        geom_stack_list = [
            paths.stack_data_path / f"{x}.fit" for x in geom_mosaic_stacks["filename"]
        ]
        border = survey_config["mosaics"].get("border", None)
        factor = survey_config["mosaics"].get("factor", None)
        center, size = calculate_mosaic_geometry(
            geom_stack_list, ccds=survey_config["ccds"],
            pixel_scale=swarp_config.get("pixel_scale", None),
            border=border,
            factor=factor,
        )
        swarp_config["center_type"] = "MANUAL"
        swarp_config["center"] = center #f"{center[0]:.6f},{center[1]:.6f}"
        swarp_config["image_size"] = size #f"{size[0]},{size[1]}"
        swarp_config["pixelscale_type"] = "MANUAL"
        
        mosaic_stacks = get_stack_data(field, tile, band)
        if len(mosaic_stacks) == 0:
            logger.info("No stacks to build.")
            return None
        if include_neighbors:
            mosaic_stacks["ccds"] = [
                survey_config["ccds"] for _ in range(len(mosaic_stacks))
            ]
            neighbor_stacks = get_neighbor_stacks(field, tile, band)
            logger.info(f"including neighbors {len(neighbor_stacks)}")
        else:
            neighbor_stacks = None
        return cls(
            mosaic_stacks, 
            mosaic_path, 
            neighbor_stacks=neighbor_stacks,
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file,
        )

    @classmethod
    def coverage_from_dxs_spec(
        cls, field, tile, band, 
        pixel_scale, 
        include_neighbors=True,
        prefix=None, 
        swarp_config=None, 
        swarp_config_file=None, 
    ):
        """
        similar to from_dxs_spec but uses coverage.swarp config, option to include pixel scale
        Value should be set to 1
        """
        swarp_config = swarp_config or {}
        coverage_config = {
            "pixel_scale": pixel_scale,
        }
        swarp_config_file = swarp_config_file or paths.config_path / "swarp/coverage.swarp"
        swarp_config.update(coverage_config)
        return cls.from_dxs_spec(
            field, tile, band, prefix=prefix, include_neighbors=True,
            extension="cov", swarp_config=swarp_config, swarp_config_file=swarp_config_file,
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
        if prepare_hdus:
            if self.swarp_config_file == paths.config_path / "swarp/coverage.swarp":
                if "value" not in kwargs:
                    kwargs["value"] = 1.0
                    logger.info("map values for HDUs set value to 1.0")
            hdu_list = self.prepare_all_hdus(self.stack_list, n_cpus=n_cpus, **kwargs)
            self.write_swarp_list(hdu_list)
        config = self.build_swarp_config()
        config["nthreads"] = n_cpus or 1    
        config.update(self.swarp_config)
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
        kwargs = self.swarp.run(swarp_list_name)
        logger.info(f"Mosaic written to {self.mosaic_path}")
        with open(self.swarp_run_parameters_path, "w+") as f:
            json.dump(kwargs, f, indent=2)
       
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
                    stack_path, ccds=self.ccds_list[ii], **kwargs)
                hdu_list.extend(results)
        else:
            # can do many threads of stacks...
            kwarg_list = [dict(**kwargs, ccds=ccds) for ccds in self.ccds_list]            
            arg_list = [(sp, kw) for sp, kw in zip(stack_list, kwarg_list)]
            # This is a bit ugly, but essentially we need to make a single object (tuple)
            # that we can give to a single-arg function for pool.map(). 
            with Pool(n_cpus) as pool:
                results = pool.map(_hdu_prep_wrapper, arg_list)
                hdu_list = [p for result in results for p in result]
        for hdu_path in hdu_list:
            assert hdu_path.exists()
        return hdu_list

    def write_swarp_list(self, hdu_list):
        with open(self.swarp_list_path, "w+") as f:
            f.writelines([str(hdu_path)+"\n" for hdu_path in hdu_list])        
    
    def build_swarp_config(self,):
        config = {}
        config["imageout_name"] = self.mosaic_path
        weightout_name = self.mosaic_dir / f"{self.mosaic_path.stem}.weight.fits"
        config["weightout_name"] = weightout_name
        config["resample_dir"] = paths.temp_swarp_path
        config["pixel_scale"] = survey_config["mosaics"].get("pixel_scale", 0.2) #f"{pixel_scale:.6f}"        
        return config

    def add_extra_keys(self, keys=None, magzpt_inc_exptime=False):
        keys = keys or {}
        keys["seeing"] = (self.calc_seeing(), "median seeing of stacks, in arcsec")
        includes = "does" if magzpt_inc_exptime else "does not"
        keys["magzpt"] = (
            self.calc_magzpt(magzpt_inc_exptime=magzpt_inc_exptime),
            f"median; {includes} inc. 2.5log(t_exp)"
        )
        branch, local_sha = get_git_info()
        keys["branch"] = (branch, "pipeline branch")
        keys["localSHA"] = (local_sha.replace("'",""), "pipeline SHA")
        add_keys(self.mosaic_path, keys, hdu=0, verbose=True)

    def calc_seeing(self,):
        """
        this DOESN'T include any seeing values for neighbor stacks, as they're
        only on the very edges
        """
        seeing_cols = [f"seeing_{ccd}" for ccd in survey_config["ccds"]]
        seeing_df = self.mosaic_stacks[ seeing_cols ]
        seeing = np.median(seeing_df.stack().values)
        return seeing

    def calc_magzpt(self, magzpt_inc_exptime=True):
        """
        If counts per second (ie, exptime = 1), DON't include exptime.
        Else: we'll have T times more flux per obeject. so need to add 
        """
        magzpt_cols = [f"magzpt_{ccd}" for ccd in survey_config["ccds"]]
        if magzpt_inc_exptime:
            exptime_factor = 2.5*np.log10(self.mosaic_stacks["exptime"])
            magzpt_df = self.mosaic_stacks[ magzpt_cols ] + exptime_factor
        else:
            magzpt_df = self.mosaic_stacks[ magzpt_cols ]
        magzpt = np.median(magzpt_df.stack().values)
        return magzpt

class HDUPreparer:
    def __init__(
        self, hdu, hdu_path,
        value=None,
        resize=False, edges=25.0, 
        mask_sources=False, mask_wcs=None, mask_map=None,
        normalise_exptime=False, exptime=None,
        subtract_bgr=False, bgr_size=None, filter_size=1, sigma=3.0,
    ):
        self.data = hdu.data
        if value is not None:
            self.data = np.random.uniform(0.9999*value, 1.0001*value, hdu.data.shape)
            #self.data = np.full(hdu.data.shape, value)
        self.header = hdu.header
        self.hdu_path = hdu_path
        self.resize = resize
        self.edges = edges
        self.mask_sources = mask_sources
        self.mask_wcs = mask_wcs
        self.mask_map = mask_map
        self.normalise_exptime = normalise_exptime
        self.exptime = exptime
        self.subtract_bgr = subtract_bgr
        self.bgr_size = bgr_size
        self.sigma = sigma

        self.fwcs = WCS(hdu.header)
        self.fwcs.fix()
        self.xlen = hdu.header["NAXIS1"]
        self.ylen = hdu.header["NAXIS2"]

        self.center = (self.ylen // 2, self.xlen // 2)
        self.center_coords = SkyCoord.from_pixel(
            xp=self.center[1], yp=self.center[0], wcs=self.fwcs, origin=1
        )

    def prepare_hdu(self):
        if self.mask_sources:
            source_mask = self.get_source_mask()
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
                self.data, position=self.center, size=cutout_size, wcs=self.fwcs
            )
            self.data = cutout.data
            self.header.update(cutout.wcs.to_header())

        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        if self.hdu_path.exists():
            logger.warn(f"{self.hdu_path.stem} is being overwritten!")
        hdu.writeto(self.hdu_path, overwrite=True)

    def get_source_mask(self):
        approx_size = (self.ylen + 50, self.xlen + 50)
        # do better by taking the max, min of the stack footprint in mask wcs pix.
        apx_map = Cutout2D(
            self.mask_map, 
            position=self.center_coords,
            size=approx_size,
            wcs=self.mask_wcs,
            mode="partial", # definitely want PARTIAL. 
            fill_value=0
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
        stack_path = Path(stack_path)
        results = []
        ccds = ccds or survey_config["ccds"]
        with fits.open(stack_path) as f:
            for ii, ccd in enumerate(ccds):
                hdu_name = get_hdu_name(stack_path, ccd, prefix=hdu_prefix)
                logger.info(f"prep - {hdu_name}")
                hdu_path = paths.temp_hdus_path / hdu_name # includes ".fits" already...
                results.append(hdu_path)
                if not overwrite and hdu_path.exists():
                    continue
                p = cls(f[ccd], hdu_path, exptime=f[0].header["EXP_TIME"], **kwargs)
                p.prepare_hdu()
        return results

##======== Aux. functions for HDUPreparer

def _hdu_prep_wrapper(arg):
    args, kwargs = arg
    return HDUPreparer.prepare_stack(args, **kwargs)    

def get_hdu_name(stack_path, ccd, weight=False, prefix=None):
    weight = ".weight" if weight else ""
    prefix = prefix or ""
    return f"{prefix}{stack_path.stem}_{ccd:02d}{weight}.fits"


##======== Finding out which stacks are important for this mosaic.

def get_stack_data(field, tile, band, pointing=None):
    """
    
    """
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
    return stack_data.query(query)

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

def calculate_mosaic_geometry(
    stack_list, ccds=None, factor=None, border=None, pixel_scale=None
):
    """
    How big should the mosaic be?
    """
    logger.info("Calculating mosaic geometry")
    ccds = ccds or [0]
    ra_values = []
    dec_values = []
    for ii, stack_path in enumerate(stack_list):
        with fits.open(stack_path) as f:
            for ccd in ccds:
                fwcs = WCS(f[ccd].header)
                footprint = fwcs.calc_footprint()
                ra_values.extend(footprint[:,0])
                dec_values.extend(footprint[:,1])
    ra_limits = (np.min(ra_values), np.max(ra_values))
    dec_limits = (np.min(dec_values), np.max(dec_values))

    # center is easy.
    center = (np.mean(ra_limits), np.mean(dec_limits))
    # image size takes a bit more thought because of spherical things.
    cos_dec = np.cos(center[1]*np.pi / 180.)
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
    logger.info(f"geom - center {center[0]:.3f}, {center[1]:.3f}")
    return center, mosaic_size



def build_mosaic_header(center, size, pixel_scale, proj="TAN"):
    """
    NOT for use in FITS files. Useful for cropping input stacks down to size.
    """
    w = build_mosaic_wcs(center, size, pixel_scale, proj=proj)
    h = w.to_header()
    h.insert(0, "SIMPLE", "T")
    h.insert(1, "BITPIX", -32)
    h.insert(2, "NAXIS", 2)
    h.insert(3, "NAXIS1", size[0])
    h.insert(4, "NAXIS2", size[1])
    return h
    

def build_mosaic_wcs(center, size, pixel_scale, proj="TAN"):
    
    w = WCS(naxis=2)
    w.wcs.crpix = [size[0]/2, size[1]/2]
    w.wcs.cdelt = [pixel_scale / 3600., pixel_scale / 3600.]
    w.wcs.crval = list(center)
    w.wcs.ctype = [
        "RA" + "-" * (6-len(proj)) + proj, "DEC" + "-" * (5-len(proj)) + proj
    ]
    w.fix()
    return w
    

def add_keys(mosaic_path, data, hdu=0, verbose=False):
    with fits.open(mosaic_path, mode="update") as mosaic:
        for key, val in data.items():
            if verbose:
                print(f"Update {key} to {val}")
            #mosaic[hdu].header[key.upper()] = val
            if not isinstance(data, tuple):
                data = (data,)
            try:
                mosaic[hdu].header.set(key.upper(), *val)
            except:
                pass
        mosaic.flush()

if __name__ == "__main__":
    print("no main block")

    









