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

stack_data = pd.read_csv(paths.header_data_path)

import matplotlib.pyplot as plt

class MosaicBuilderError(Exception):
    pass

class MosaicBuilder:

    """
    Class for building mosiacs. Calls SWarp.
    """

    def __init__(
        self, 
        relevant_stacks, 
        mosaic_path, 
        swarp_config=None, 
        swarp_config_file= None, 
        n_cpus=None
    ):
        check_modules("swarp") # do we have swarp available?
        if isinstance(relevant_stacks, pd.DataFrame):
            self.relevant_stacks = relevant_stacks
            self.stack_list = [
                paths.stack_data_path / f"{x}.fit" for x in self.relevant_stacks["filename"]
            ]
        elif isinstance(relevant_stacks, list):
            self.stack_list = relevant_stacks
        self.mosaic_path = Path(mosaic_path)
        self.swarp_config = swarp_config or {}
        self.swarp_config_file = swarp_config_file or paths.config_path / "swarp/mosaic.swarp"
        self.n_cpus = n_cpus
    
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
        prefix=None, extension=None, swarp_config=None, swarp_config_file=None, n_cpus=None,
    ):
        mosaic_dir = paths.get_mosaic_dir(field, tile, band)
        mosaic_stem = paths.get_mosaic_stem(field, tile, band, prefix=prefix)
        extension = f".{extension}" if extension is not None else ""
        mosaic_path = mosaic_dir / f"{mosaic_stem}{extension}.fits"
        swarp_config = swarp_config or {}

        geom_relevant_stacks = get_stack_data(field, tile, band=None)
        geom_stack_list = [
            paths.stack_data_path / f"{x}.fit" for x in geom_relevant_stacks["filename"]
        ]
        center, size = calculate_mosaic_geometry(
            geom_stack_list, ccds=survey_config["ccds"],
            pixel_scale=swarp_config.get("pixel_scale", None)
        )
        swarp_config["center_type"] = "MANUAL"
        swarp_config["center"] = center #f"{center[0]:.6f},{center[1]:.6f}"
        swarp_config["image_size"] = size #f"{size[0]},{size[1]}"
        swarp_config["pixelscale_type"] = "MANUAL"
        
        relevant_stacks = get_stack_data(field, tile, band)
        if len(relevant_stacks) == 0:
            print("No stacks to build.")
            return None
        return cls(
            relevant_stacks, 
            mosaic_path, 
            swarp_config=swarp_config, 
            swarp_config_file=swarp_config_file, 
            n_cpus=n_cpus
        )

    @classmethod
    def coverage_from_dxs_spec(
        cls, field, tile, band, pixel_scale,
        prefix=None, swarp_config=None, swarp_config_file=None, n_cpus=None,
    ):
        swarp_config = swarp_config or {}
        coverage_config = {
            "pixel_scale": pixel_scale,
        }
        swarp_config_file = swarp_config_file or paths.config_path / "swarp/coverage.swarp"
        swarp_config.update(coverage_config)
        return cls.from_dxs_spec(
            field, tile, band, prefix=prefix, 
            extension="cov", swarp_config=swarp_config, swarp_config_file=swarp_config_file,
            n_cpus=n_cpus
        )

    def build(self, prepare_hdus=True, **kwargs):
        """
        Parameters
        ----------
        stack_list
            list of stacks to build mosaic from
        kwargs
            kwargs that are passed to HDUPreparer.prepare_stack() 
        """
        if prepare_hdus:
            hdu_list = self.prepare_all_hdus(self.stack_list, **kwargs)
            self.write_swarp_list(hdu_list)
        config = self.build_swarp_config()
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
       
    def prepare_all_hdus(self, stack_list=None, **kwargs):
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
        if self.n_cpus is None:    
            hdu_list = []        
            for stack_path in stack_list:
                results = HDUPreparer.prepare_stack(stack_path, **kwargs)
                hdu_list.extend(results)
        else:
            # can do many threads of stacks...
            arg_list = [(stack_path, kwargs) for stack_path in stack_list]
            with Pool(self.n_cpus) as pool:
                results = pool.map(_hdu_prep_wrapper, arg_list)
                hdu_list = [p for result in results for p in result]
        for hdu_path in hdu_list:
            assert hdu_path.exists()
        return hdu_list

    def write_swarp_list(self, hdu_list):
        with open(self.swarp_list_path, "w+") as f:
            f.writelines([str(hdu_path)+"\n" for hdu_path in hdu_list])        
    
    def build_swarp_config(self):
        config = {}
        config["imageout_name"] = self.mosaic_path
        weightout_name = self.mosaic_dir / f"{self.mosaic_path.stem}.weight.fits"
        config["weightout_name"] = weightout_name
        config["resample_dir"] = paths.temp_swarp_path
        config["pixel_scale"] = survey_config["pixel_scale"] #f"{pixel_scale:.6f}"        
        config["nthreads"] = self.n_cpus
        return config

    def add_extra_keys(self, keys=None, normalise_exptime=False):
        keys = keys or {}
        keys["seeing"] = self.relevant_stacks["seeing"].median()
        if normalise_exptime:
            keys["magzpt"] = self.relevant_stacks["magzpt"].median()
        else:
            exptime_factor = 2.5*np.log10(self.relevant_stacks["exptime"])
            magzpt_col = self.relevant_stacks["magzpt"] + exptime_factor
            keys["magzpt"] = magzpt_col.median() # this should work, as still pd.Series...
        branch, local_sha = get_git_info()
        keys["branch"] = branch
        keys["localSHA"] = local_sha.replace("'","")
        add_keys(self.mosaic_path, keys, hdu=0, verbose=True)


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
        hdu.writeto(self.hdu_path)

    def get_source_mask(self):
        approx_size = (self.ylen + 50, self.xlen + 50)
        apx_map = Cutout2D(
            self.mask_map, 
            position=self.center_coords,
            size=approx_size,
            wcs=self.mask_wcs,
            mode="partial",
            fill_value=0
        )
        reprojected_map, reprojected_footprint = rpj.reproject_interp(
            (apx_map.data, apx_map.wcs), 
            output_projection=self.header, 
            order="nearest-neighbor"
        )
        assert True # check reprojected_footprint isclose self.data?
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
    def prepare_stack(cls, stack_path, overwrite=False, hdu_prefix=None, **kwargs):
        stack_path = Path(stack_path)
        results = []
        with fits.open(stack_path) as f:
            for ii, ccd in enumerate(survey_config["ccds"]):
                hdu_name = get_hdu_name(stack_path, ccd, prefix=hdu_prefix)
                print(hdu_name)
                hdu_path = paths.temp_hdus_path / hdu_name # includes ".fits" already...
                results.append(hdu_path)
                if not overwrite and hdu_path.exists():
                    continue
                p = cls(f[ccd], hdu_path, exptime=f[0].header["EXP_TIME"], **kwargs)
                p.prepare_hdu()
        return results


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

def _hdu_prep_wrapper(arg):
    args, kwargs = arg
    return HDUPreparer.prepare_stack(args, **kwargs)    

def get_hdu_name(stack_path, ccd, weight=False, prefix=None):
    weight = ".weight" if weight else ""
    prefix = prefix or ""
    return f"{prefix}{stack_path.stem}_{ccd:02d}{weight}.fits"

def calculate_mosaic_geometry(
    stack_list, ccds=None, factor=None, border=None, pixel_scale=None
):
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
    pixel_scale = pixel_scale or survey_config["pixel_scale"]
    plate_factor = 3600. / pixel_scale
    x_size = abs(ra_limits[1] - ra_limits[0]) * plate_factor * cos_dec
    y_size = abs(dec_limits[1] - dec_limits[0]) * plate_factor
    mosaic_size = (int(x_size), int(y_size))
    if factor is not None:
        mosaic_size = (mosaic_size[0]*factor, mosaic_size[1]*factor)
    if border is not None:
        mosaic_size = (mosaic_size[0]+border, mosaic_size[1]+border)
    max_size = [int(x) for x in survey_config["max_mosaic_size"]]
    if mosaic_size[0] > max_size[0] or mosaic_size[1] > max_size[1]:
        raise MosaicBuilderError(
            f"Mosaic too large: {mosaic_size[0]},{mosaic_size[1]}"
            f" larger than {max_size[0],max_size[1]}"
            f" \n check configuration/survey_config.yaml"
        )
    logger.info(f"geom - size {mosaic_size[0]},{mosaic_size[1]}")
    logger.info(f"geom - center {center[0]:.3f}, {center[1]:.3f}")
    return center, mosaic_size    

def add_keys(mosaic_path, data, hdu=0, verbose=False):
    with fits.open(mosaic_path, mode="update") as mosaic:
        for key, val in data.items():
            if verbose:
                print(f"Update {key} to {val}")
            mosaic[hdu].header[key.upper()] = val
        mosaic.flush()

if __name__ == "__main__":
    builder = MosaicBuilder("EN", 4, "K") # , n_cpus=4)
    builder.build()
    builder.add_extra_keys()

    









