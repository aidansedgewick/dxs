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
from astropy.wcs import WCS, FITSFixedWarning

from dxs.hdu_preprocessor import HDUPreprocessor

from astromatic_wrapper.api import Astromatic

from dxs.utils.image import build_mosaic_wcs
from dxs.utils.misc import (
    check_modules, format_flags, get_git_info, AstropyFilter
)

from dxs import paths

logger = logging.getLogger("mosaic_builder")
astropy_logger = logging.getLogger("astropy")
astropy_logger.addFilter(AstropyFilter())

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

default_neighbors_config_path = paths.config_path / "dxs_arrangement.yaml"
def load_neighbors_config(neighbors_config_path=default_neighbors_config_path):
    with open(neighbors_config_path) as f:
        neighbors_config = yaml.load(f, Loader=yaml.FullLoader)
        neighbors_config["arrangement_to_tile"] = {}
        for field, arrangement in neighbors_config["arrangement"].items():
            neighbors_config["arrangement_to_tile"][field] = {
                tuple(v): k for k, v in arrangement.items()
            }
    return neighbors_config
neighbors_config = load_neighbors_config()

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
        header_keys=None,
        hdu_prep_kwargs=None,
        stack_data_dir=None,
        ext=".fit",
    ):
        if isinstance(mosaic_stacks, pd.DataFrame):
            self.mosaic_stacks = mosaic_stacks
            if neighbor_stacks is not None:
                assert isinstance(neighbor_stacks, pd.DataFrame)
                relevant_stacks = pd.concat([mosaic_stacks, neighbor_stacks])
            else:
                relevant_stacks = mosaic_stacks
            try:
                logger.info("stacks to use:")
                print(relevant_stacks.groupby(["field", "tile", "pointing"]).size()) 
            except:
                pass
        
            # make a list of stacks from the dataframe...
            stack_data_dir = stack_data_dir or paths.stack_data_path    
            self.stack_list = [
                stack_data_dir  / f"{x}{ext}" for x in relevant_stacks["filename"]
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
                    "to use BOTH `mosaic_stacks` and `neighbor_stacks` must be pd.DataFrame (with \"filename\" column)"
                )
            self.ccds_list = [None for _ in range(len(mosaic_stacks))]

        self.mosaic_path = Path(mosaic_path)
        logger.info(f"output at {self.mosaic_path.name}")
        self.swarp_config = swarp_config or {}
        self.swarp_config_file = swarp_config_file or paths.config_path / "swarp/mosaic.swarp"

        self.hdu_prep_kwargs = hdu_prep_kwargs or {}
        self.header_keys = header_keys or {}

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
        convert_vega_to_AB=True,
        include_deprecated_stacks=True,
        prefix=None,
        suffix=None,
        extension=None,
        add_flux_scale=True,
        swarp_config=None, 
        swarp_config_file=None, 
        hdu_prep_kwargs=None,
        header_keys=None,
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
        geom_mosaic_stacks = get_stack_data(
            field, tile, band=None, include_deprecated_stacks=True # always inc. for geom...
        )
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
        logger.info(f"include deprec stacks? {include_deprecated_stacks}")

        mosaic_stacks = get_stack_data(
            field, tile, band, include_deprecated_stacks=include_deprecated_stacks
        )
        if len(mosaic_stacks) == 0:
            logger.info(f"No stacks in {field} {tile} {band} to build.")
            return None
            
        if include_neighbors:
            if "ccds" not in mosaic_stacks.columns:
                mosaic_stacks["ccds"] = [
                    survey_config["ccds"] for _ in range(len(mosaic_stacks))
                ]
            neighbor_stacks = get_neighbor_stacks(
                field, tile, band, include_deprecated_stacks=include_deprecated_stacks
            )
            logger.info(f"{len(neighbor_stacks)} stacks (including neighbors)")
        else:
            neighbor_stacks = None

        #=========== sort some things for preparing ============#
        #normalise_exptime = hdu_prep_kwargs.get("normalise_exptime", False) # check default in HDUPreprocessor
        #if normalise_exptime:
        #    magzpt_inc_exptime = False # ie, if counts/s img, no need to account!
        #else:
        #    magzpt_inc_exptime = True

        if convert_vega_to_AB:
            AB_conversion = survey_config["ab_vega_offset"].get(band, None)
            if AB_conversion is None:
                logger.warning("No AB offset for {band} - check survey_config.yaml...")
                AB_conversion = 0.
            logger.info(f"Convert to AB: add {AB_conversion:.4f} to magzpt")
        else:
            AB_conversion = 0.

        header_keys = header_keys or {}
        header_keys["seeing"] = (
            cls.calc_seeing(mosaic_stacks), "median seeing of stacks, in arcsec"
        )
        magzpt_value = cls.calc_magzpt(mosaic_stacks, magzpt_inc_exptime=True)
        magzpt_value = magzpt_value + AB_conversion
        header_keys["magzpt"] = (
            magzpt_value, f"median; inc. 2.5log(t_exp), dAB={AB_conversion:.4f}"
        )
        header_keys["usedeprec"] = (
            include_deprecated_stacks, f"mosaic include deprecated stacks?"
        )
        logger.info("header_keys:")
        for k, v in header_keys.items():
            print(f"     {k} = {v}")

        hdu_prep_kwargs["add_flux_scale"] = add_flux_scale
        hdu_prep_kwargs["AB_conversion"] = AB_conversion
        hdu_prep_kwargs["reference_magzpt"] = magzpt_value # NOW INCLUDES AB.
        
        return cls(
            mosaic_stacks, 
            mosaic_path, 
            neighbor_stacks=neighbor_stacks,
            header_keys=header_keys,
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file,
            hdu_prep_kwargs=hdu_prep_kwargs
        )

    @classmethod
    def coverage_from_dxs_spec(
        cls, field, tile, band, 
        include_neighbors=True,
        include_deprecated_stacks=True,
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
        #swarp_config["pixel_scale"] = pixel_scale
        swarp_config_file = swarp_config_file or paths.config_path / "swarp/coverage.swarp"
        hdu_prep_kwargs = hdu_prep_kwargs or {}
        hdu_prep_kwargs["fill_value"] = 1.0
        
        return cls.from_dxs_spec(
            field, tile, band, 
            prefix=prefix, 
            suffix=suffix, 
            include_neighbors=include_neighbors,
            include_deprecated_stacks=include_deprecated_stacks,
            extension="cov", 
            add_flux_scale=False,
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file,
            hdu_prep_kwargs=hdu_prep_kwargs,
        )

    @classmethod
    def exptime_from_dxs_spec(
        cls, field, tile, band, 
        include_neighbors=True,
        include_deprecated_stacks=True,
        prefix=None,
        suffix=None,
        swarp_config=None, 
        swarp_config_file=None, 
        hdu_prep_kwargs=None,
    ):
        """
        similar to from_dxs_spec but uses coverage.swarp config, 
        option to include pixel scale
        fill_value should be set to exptime
        """
        swarp_config = swarp_config or {}
        #swarp_config["pixel_scale"] = pixel_scale
        swarp_config_file = swarp_config_file or paths.config_path / "swarp/coverage.swarp"
        hdu_prep_kwargs = hdu_prep_kwargs or {}
        hdu_prep_kwargs["fill_value"] = "exptime"
        
        return cls.from_dxs_spec(
            field, tile, band, 
            prefix=prefix, 
            suffix=suffix, 
            include_neighbors=include_neighbors,
            include_deprecated_stacks=include_deprecated_stacks,
            extension="exp", 
            add_flux_scale=False,
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file,
            hdu_prep_kwargs=hdu_prep_kwargs,
        )

    def build(self, prepare_hdus=True, n_cpus=None, **build_hdu_prep_kwargs):
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
            kwargs that are passed to HDUPreprocessor.prepare_stack() 
        """
        check_modules("swarp") # do we have swarp available?
        if len(build_hdu_prep_kwargs) > 0:
            logger.warn(
                "please pass all hdu_prep_kwargs at initialisation as "
                "hdu_prep_kwargs=<dict of prep kwargs>. Deprecated soon..."
            )
            self.hdu_prep_kwargs.update(build_hdu_prep_kwargs)
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
        self.add_extra_keys()
       
    def prepare_all_hdus(self, stack_list=None, n_cpus=None, **kwargs):
        """
        Parameters
        ----------
        stack_list
            a list of (Path-like) paths to stacks to prepare HDUs for.
        kwargs
            key-word arguments accepted by HDUPreprocessor(), and "prefix".
        """
        if stack_list is None:
            stack_list = self.stack_list

        print("prepare stacks with:")
        for k, v in kwargs.items():
            try:                
                print(f"    {k}: {v}")
            except:
                print(f"    {k}: [CAN'T PRINT]")
        if n_cpus is None:    
            hdu_list = []
            for ii, stack_path in enumerate(stack_list):
                results = HDUPreprocessor.prepare_stack(
                    stack_path, ccds=self.ccds_list[ii], **kwargs
                )
                hdu_list.extend(results)
        else:
            # can do many threads of stacks...
            kwarg_list = [dict(**kwargs, ccds=ccds) for ccds in self.ccds_list]            
            arg_list = [(sp, kw) for sp, kw in zip(stack_list, kwarg_list)]
            # This is a ugly, but essentially we need to make a single object (tuple)
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
        SWARP does something very odd. 
        it seems to IGNORE the first input file.
        so we add a tiny file with not much data as the first file (2x2 pixels).
        """
        center = self.swarp_config.get("swarp_config", None)
        if center is None:
            with fits.open(hdu_list[0]) as f:
                h = f[0].header
                center = SkyCoord(ra=h["CRVAL1"], dec=h["CRVAL2"], unit="deg")
                logger.info(f"small, fake img center at {center}")
        
        tiny_hdu_path = self.build_tiny_hdu(center)
        with open(self.swarp_list_path, "w+") as f:
            hdu_path_list = (
                [str(tiny_hdu_path)+"\n"] + [str(hdu_path)+"\n" for hdu_path in hdu_list]
            )
            f.writelines(hdu_path_list)

    def build_tiny_hdu(self, center):  
        tiny_wcs = build_mosaic_wcs(
            center=center, size=(2,2), pixel_scale=0.2
        )
        tiny_header = tiny_wcs.to_header()
        tiny_data = np.random.uniform(0, 1, (2,2))
        tiny_hdu = fits.PrimaryHDU(data=tiny_data, header=tiny_header)
        tiny_hdu_path = paths.scratch_hdus_path / f"{self.mosaic_path.stem}_tiny_data.fits"
        tiny_hdu.writeto(tiny_hdu_path, overwrite=True) 
        return tiny_hdu_path
    
    def initialise_astromatic(self, n_cpus=None):
        config = self.build_swarp_config()
        config["nthreads"] = n_cpus or 1    
        config.update(self.swarp_config)
        #if not os.isatty(0):
        #    config["verbose_type"] = "quiet" # If running on a batch, prevent spewing output.
        config = format_flags(config)
        self.swarp = Astromatic(
            "SWarp", 
            str(paths.scratch_swarp_path), # I think this is ignored by Astromatic() anyway?!
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
        config["resample_dir"] = paths.scratch_swarp_path
        config["pixel_scale"] = survey_config["mosaics"].get("pixel_scale", 0.2) #f"{pixel_scale:.6f}" 
        config["pixelscale_type"] = "manual"       
        config["xml_name"] = self.swarp_xml_path
        return config

    def add_extra_keys(self, extra_keys=None, magzpt_inc_exptime=True):
        extra_keys = extra_keys or {}
        print(self.header_keys)
        keys = self.header_keys
        keys.update(extra_keys)
        print("these are the keys")
        print(keys)
        
        keys["do_flxsc"] = (
            self.hdu_prep_kwargs.get("add_flux_scale", False),
            "use 10**(-0.4*hdu_magzpt - refmag)"
        )
        keys["refmag"] = (
            self.hdu_prep_kwargs.get("reference_magzpt", None),
                       
        )
        keys["aboffset"] = (
            self.hdu_prep_kwargs.get("AB_conversion", 0.),
            "added onto vals found in stacks"
        )
        keys["fill_val"] = (
            self.hdu_prep_kwargs.get("fill_value", None),
            "fill HDUs with this for SWARP"
        )
        keys["trimedge"] = (
            self.hdu_prep_kwargs.get("edges", 0),
            "trimmed pix from edges of stacks"
        )
        keys["bgr_sub"] = (
            self.hdu_prep_kwargs.get("subtract_bgr", False),
            "did we subtract the bgr in hduprep"
        )
        keys["bgrfiltr"] = (
            self.hdu_prep_kwargs.get("filter_size", "default"),
            "filter size in sextractor bgr"
        )
        keys["bgrsigma"] = (
            self.hdu_prep_kwargs.get("sigma", "default"),
            "sigma clip for sextractor bgr"
        )
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
            exptime_col = 2.5 * np.log10(stack_data["exptime"])
        else:
            exptime_col = 0.
        for col in magzpt_cols:
            magzpt_df[col] = stack_data[col] + exptime_col
        magzpt = np.median(magzpt_df.stack().values)
        logger.info(f"magzpt={magzpt:.3f} (+2.5log(expT): {magzpt_inc_exptime})")
        return magzpt

##======== Aux. functions for HDUPreprocessor

def _hdu_prep_wrapper(arg):
    args, kwargs = arg
    return HDUPreprocessor.prepare_stack(args, **kwargs)    

##======== Finding out which stacks are important for a mosaic.

def get_stack_data(field, tile, band, pointing=None, include_deprecated_stacks=True):
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
    stacks = header_data.query(query)
    ccds = survey_config["ccds"]
    if not include_deprecated_stacks:
        deprecated_query = "deprec_mf==0"
        stacks = stacks.query(deprecated_query)
        ccd_lists = []
        for ii, row in stacks.iterrows():
            ccds_ii = [ccd for ccd in ccds if row[f"deprec_{ccd}"] == 0]
            ccd_lists.append(ccds_ii)
        ccds_series = pd.Series(ccd_lists, index=stacks.index, dtype=object)
        assert sum(stacks["deprec_mf"]) == 0
    else:
        ccds_series = pd.Series(
            [ccds for _ in range(len(stacks))], index=stacks.index, dtype=object
        )
    assert ccds_series.index.equals(stacks.index)
    stacks.insert(len(stacks.columns), "ccds", ccds_series)
    return stacks

def get_neighbor_stacks(field, tile, band, include_deprecated_stacks=True):
    """
    Which stacks are on the border of the mosaic we're SWarping? 
    We should add these in, to make extracting the catalog nicer.
    """
    neighbors = get_neighbor_tiles(field, tile)
    df_list = []
    for neighbor_tile, cardinal in neighbors.items():
        hdus_for_cardinal = neighbors_config["border_hdus"][cardinal]
        for pointing, relevant_ccds in hdus_for_cardinal.items():
            stacks = get_stack_data(
                field, neighbor_tile, band, 
                pointing=pointing, 
                include_deprecated_stacks=include_deprecated_stacks
            )
            if "ccds" in stacks.columns:
                stacks.drop("ccds", inplace=True, axis=1) # axis=1 for column drop.
            if not include_deprecated_stacks:
                assert sum(stacks["deprec_mf"]) == 0
                ccd_lists = []
                for ii, row in stacks.iterrows():
                    ccds_ii = [ccd for ccd in relevant_ccds if row[f"deprec_{ccd}"] == 0]
                    ccd_lists.append(ccds_ii)
                ccds_series = pd.Series(ccd_lists, index=stacks.index, dtype=object)
            else:
                ccds_series = pd.Series(
                    [relevant_ccds for _ in range(len(stacks))], index=stacks.index, dtype=object
                )
            assert stacks.index.equals(ccds_series.index)
            stacks.insert(len(stacks.columns), "ccds", ccds_series)
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
    stack_list, ccds=None, factor=None, border=None, pixel_scale=None, n_cpus=None
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
    if n_cpus is None:
        for stack_path in tqdm.tqdm(stack_list):
            with fits.open(stack_path) as f:
                for ccd in ccds:
                    hdu_wcs = WCS(f[ccd].header)
                    footprint = hdu_wcs.calc_footprint()
                    ra_values.extend(footprint[:,0])
                    dec_values.extend(footprint[:,1])
    ra_limits = (np.min(ra_values), np.max(ra_values))
    dec_limits = (np.min(dec_values), np.max(dec_values))

    # center is easy.
    center = SkyCoord(ra=np.mean(ra_limits), dec=np.mean(dec_limits), unit="deg")
    # image size takes a bit more thought because of spherical things.
    cos_dec = np.cos(center.dec) # an astropy Angle...
    print(center.dec)
    pixel_scale = pixel_scale or survey_config["mosaics"]["pixel_scale"]
    plate_factor = 3600. / pixel_scale
    print(abs(ra_limits[1] - ra_limits[0]) * plate_factor, cos_dec)
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
            f" larger than {max_size[0],max_size[1]}q"
            f" \n check configuration/survey_config.yaml"
        )
    logger.info(f"geom - size {mosaic_size[0]},{mosaic_size[1]}")
    logger.info(f"geom - center {center.ra:.3f}, {center.dec:.3f}")
    return center, mosaic_size

def _get_ra_dec_wrapper():
    pass


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

    









