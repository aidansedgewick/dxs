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
from photutils.background import SExtractorBackground, Background2D

from dxs import paths

logger = logging.getLogger("hdu_prep")

class HDUPreprocessorError(Exception):
    pass

class HDUPreprocessor:
    """
    Class for preparing a single HDU ready for SWarp.

    eg.

    >>> hdup = HDUPreprocessor(hdu, "./hdu_out.fits")
    >>> hdup.prepare_hdu()

    Parameters
    ----------
    hdu
        a single astropy hdu object
    hdu_output_path
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
        used to calculate flxscale.  
        default None.
    overwrite_magzpt
        bool - if True, replace the MAGZPT entry in the header, with the version with
         +2.5log10(exptime) and +AB_conversion (if provided). default False
    """

    def __init__(
        self, 
        hdu: fits.hdu.image._ImageBaseHDU, 
        hdu_output_path: str,
        fill_value: float = None, 
        fill_value_var: float = 1e-3,
        edges: int = None, 
        subtract_bgr: bool = False, 
        bgr_size: int = None, 
        filter_size: int = 1, 
        sigma: float = 3.0,
        mask_sources: bool = False, 
        mask_header: fits.Header = None, 
        mask_map: np.ndarray = None,
        exptime: float = None, 
        add_flux_scale: bool = False, 
        reference_magzpt: float = None, 
        AB_conversion: float = 0.0,
        overwrite_magzpt: bool = False,
    ):
        self.data = hdu.data
        self.header = hdu.header

        self.hdu_output_path = Path(hdu_output_path)

        self.hdu_wcs = WCS(hdu.header)
        self.hdu_wcs.fix()
        self.xlen = hdu.header["NAXIS1"]
        self.ylen = hdu.header["NAXIS2"]

        self.center = (self.ylen // 2, self.xlen // 2)
        self.center_coords = SkyCoord.from_pixel(
            xp=self.center[1], yp=self.center[0], wcs=self.hdu_wcs, origin=1
        )

        if fill_value is not None:
            if isinstance(fill_value, str):
                if fill_value == "exptime":
                    if exptime is None:
                        raise HDUPreprocessorError(
                            "Can't fill with exptime if exptime not given!"
                        )
                    fill_value = exptime
                else:
                    raise HDUPreprocessorError("fill_value is float, int or 'exptime'")
            if subtract_bgr:
                raise HDUPreprocessorError("Don't fill with a value, and then subtract bgr!")
            if mask_sources:
                raise HDUPreprocessorError("Don't fill with a value, and then mask sources!")

        self.edges = edges

        self.fill_value = fill_value
        self.fill_value_var = fill_value_var
        if self.fill_value is not None and self.fill_value == "exptime":
            self.fill_value = exptime
        
        if subtract_bgr and bgr_size is None:
            raise HDUPreprocessorError("Must provide bgr_size to subtract background")
        if bgr_size is not None:
            if not (isinstance(bgr_size, int) or isinstance(bgr_size, tuple)):
                raise ValueError("bgr_size should be int or tuple(int, int)")
            if isinstance(bgr_size, int):
                bgr_size = (bgr_size, bgr_size)
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        self.subtract_bgr = subtract_bgr
        self.bgr_size = bgr_size
        self.filter_size = filter_size
        self.sigma = sigma

        if mask_sources:
            if mask_header is None or mask_map is None:
                raise HDUPreprocessorError(
                    "to mask sources, must provide mask_header and mask_map (np array)"
                )
            mask_wcs = WCS(mask_header)
        else:
            mask_wcs = None
        self.mask_sources = mask_sources
        self.mask_wcs = mask_wcs
        self.mask_map = mask_map

        hdu_stem = self.hdu_output_path.stem
        
        if exptime is None:
            logger.warning(f"{hdu_stem}: no exptime provided") 
        self.exptime = exptime

        magzpt = self.header.get("MAGZPT", None)
        if magzpt is None:
            logger.warning(f"{hdu_stem}: MAGZPT not found; set to ref={reference_magzpt}")
            magzpt = reference_magzpt
        if add_flux_scale:
            if reference_magzpt is None:
                raise HDUPreprocessorError(
                    "Can't add flux scale if no reference_magzpt to scale to!\n"
                    "Either provide reference_magzpt= or set add_flux_scale=False"
                )

        if magzpt is not None:
            if self.exptime is not None:
                magzpt = magzpt + 2.5 * np.log10(exptime) # log10!!!!
            magzpt = magzpt + AB_conversion

        self.add_flux_scale = add_flux_scale
        self.magzpt = magzpt
        self.reference_magzpt = reference_magzpt
        self.overwrite_magzpt = overwrite_magzpt       
      
    def modify_header(self):
        if self.add_flux_scale:
            mag_diff = self.magzpt - self.reference_magzpt
            flux_scaling = 10**(-0.4*mag_diff)
            self.header["FLXSCALE"] = flux_scaling
            logger.info(f"flxscl {self.hdu_output_path.stem} {flux_scaling:.2f}")
        if self.overwrite_magzpt:
            self.header["MAGZPT"] = self.magzpt
        if self.exptime is not None:
            self.header["EXP_TIME"] = self.exptime

    def prepare_hdu(self):
        if self.fill_value is not None:
            self.data = np.random.normal(
                self.fill_value, self.fill_value_var, self.data.shape
            )

        if self.mask_wcs is not None:
            hdu_footprint = SkyCoord(self.hdu_wcs.calc_footprint(), unit="degree")
            contains = self.mask_wcs.footprint_contains(hdu_footprint)
            if not any(contains):
                logger.info(f"{self.hdu_output_path.stem} not in mask")

        # If we've given a source mask, get the relevant parts.
        if self.mask_sources:
            try:
                source_mask = self.get_source_mask()
            except Exception as e:
                logger.warn(e)
                return None
        else:
            source_mask = None

        ## Estimate and subtract the background.
        if self.subtract_bgr:
            bgr = self.get_background(source_mask=source_mask)
            self.data = self.data - bgr.background
            self.bgr_map = bgr.background

        ## Trim to an appropriate size.
        if self.edges is not None:
            cutout_size = (int(self.ylen-2*self.edges), int(self.xlen-2*self.edges))
            cutout = Cutout2D(
                self.data, position=self.center, size=cutout_size, wcs=self.hdu_wcs
            )
            self.data = cutout.data
            self.header.update(cutout.wcs.to_header())
        
        # Finally, fix some extra bits and write out...
        self.modify_header()
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        if self.hdu_output_path.exists():
            logger.warning(f"{self.hdu_output_path.stem} is being overwritten!")
        hdu.writeto(self.hdu_output_path, overwrite=True)
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

    def get_background(self, source_mask: np.ndarray = None):
        """
        Returns a photutils Background2D object.
        >>> hdu_proc = HDUPreprocessor(hdu, output_path)
        >>> bgr = hdu_proc.get_background()
        >>> bgr_map = bgr.background.
    
        Arguments
        ---------
        source_mask
            a (boolean) 2D numpy array, where True elements are sources are to 
            be ignored during background estimation. If provided, must be the same
            shape as the hdu data.
            Defaults to None (ie, no pixels ignorred)
        
        """
        sigma_clip = SigmaClip(sigma=self.sigma)
        estimator = SExtractorBackground(sigma_clip)
        bgr = Background2D(
            self.data, 
            mask=source_mask, 
            sigma_clip=sigma_clip, 
            bkg_estimator=estimator,
            box_size=self.bgr_size, #(self.bgr_size, self.bgr_size),
            filter_size=self.filter_size
        )
        return bgr # This is a photutils.Background2D

    @classmethod
    def prepare_stack(
        cls, stack_path, overwrite=False, hdu_prefix=None, ccds=None, output_dir=None, **kwargs
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
            any kwargs which go into initialisation of HDUPreprocessor.
        """

        stack_path = Path(stack_path)
        results = []
        ccds = ccds or survey_config["ccds"]
        output_dir = Path(output_dir) or paths.scratch_hdus_path
        with fits.open(stack_path) as f:
            try:
                objsplit = f[0].header["OBJECT"].split()
                spec = objsplit[2] + f[0].header["FILTER"] + objsplit[3]
            except:
                spec = ""
            for ii, ccd in enumerate(ccds):
                stack_str = stack_path.stem + spec
                hdu_name = get_hdu_name(stack_str, ccd, prefix=hdu_prefix)
                hdu_output_path = output_dir / hdu_name # includes ".fits" already...
                logger.info(f"prp {hdu_output_path.stem}")
                if not overwrite and hdu_output_path.exists():
                    results.append(hdu_output_path)
                    continue
                exptime = f[0].header["EXP_TIME"]
                if exptime is None:
                    raise ValueError(f"{stack_path.name} EXP_TIME is NONE")
                p = cls(f[ccd], hdu_output_path, exptime=exptime, **kwargs)
                result = p.prepare_hdu()
                if result is not None:
                    results.append(hdu_output_path)
        return results

def get_hdu_name(stack_str, ccd, weight=False, prefix=None):
    weight = ".weight" if weight else ""
    prefix = prefix or ""
    try:
        stack_str = stack_str.stem
    except:
        stack_str = str(stack_str)
    return f"{prefix}{stack_str}_{ccd:02d}{weight}.fits"


