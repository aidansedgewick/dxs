import json
import logging
import os
import sys
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from easyquery import Query

from dxs import (
    MosaicBuilder, 
    BrightStarProcessor,
    CatalogExtractor, 
    CatalogMatcher, 
    CatalogPairMatcher, 
    CrosstalkProcessor
)
from dxs.utils.misc import check_modules, print_header
from dxs.utils.table import fix_column_names
from dxs.utils.image import scale_mosaic, make_good_coverage_map, mask_regions_in_mosaic
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("mosaic_pipeline")

def mosaic_pipeline(
    field, tile, band, n_cpus=1, initial=False, coverage=False, exptime=False, masked=False
):
    if not any([initial, coverage, exptime, masked]):
        initial, coverage, exptime, masked = True, True, True, True
    mosaic_types = {
        "initial": initial, "coverage": coverage, "exptime": exptime, "masked": masked
    }
    spec = (field, tile, band)

    field_name = survey_config["code_to_field"].get(field, None)
    if field_name is not None:
        bright_star_catalog_path = (
            paths.input_data_path / f"external/tmass/tmass_{field_name}_stars.csv"
        )
        query = Query("j_m - k_m < 1.0") | Query("j_m < 8.0") | Query("k_m < 8.0")
        bright_star_processor = BrightStarProcessor.from_file(
            bright_star_catalog_path, query=query, format="ascii"
        )
        extreme_star_processor = BrightStarProcessor.from_file(
            paths.config_path / "very_bright_stars.csv", format="ascii"
        ) # A hand-written file to deal with MIRA in XM06.

    tmass_mag = f"{band.lower()}_m"

    logger.info(f"Mosaics for {spec}, use {n_cpus} threads")
    logger.info(f"create mosaics: {[k for k,v in mosaic_types.items() if v]}")
    check_modules("swarp", "sex")

    stem = paths.get_mosaic_stem(*spec)
    prep_kwargs = {"hdu_prefix": f"{stem}_i"}
    builder = MosaicBuilder.from_dxs_spec(
        *spec, 
        suffix="_init",
        include_deprecated_stacks=survey_config["mosaics"]["include_deprecated_stacks"],
        hdu_prep_kwargs=prep_kwargs,
    )

    if builder is None: # ie, if there are no stacks to build
        logger.info("Builder is None. Exiting")
        return None
    if initial:
        print_header("make initial mosaic")
        builder.build(n_cpus=n_cpus)
        #builder.add_extra_keys()
        try:
            with fits.open(builder.mosaic_path) as f:
                header = f[0].header # check we can open the file!
            with open(builder.swarp_list_path) as f:
                hdu_list = f.read().splitlines()
                for hdu_path_str in hdu_list:
                    hdu_path = Path(hdu_path_str)
                    assert hdu_path.exists()
                    os.remove(hdu_path)
                    assert not hdu_path.exists()
        except Exception as e:
            logger.warning(f"during delete temp: {e}")
            pass

    if coverage:
        print_header("make coverage mosaic")
        coverage_swarp_config = {
            "center": builder.swarp_config["center"], # don't calculate twice!
            "image_size": builder.swarp_config["image_size"], # don't calculate twice!
        }
        coverage_prep_kwargs = {"hdu_prefix": f"{stem}_u"}
        #pixel_scale = survey_config["mosaics"]["pixel_scale"]# * 10.0
        cov_builder = MosaicBuilder.coverage_from_dxs_spec(
            *spec, 
            #pixel_scale=pixel_scale,
            include_deprecated_stacks=survey_config["mosaics"]["include_deprecated_stacks"],
            swarp_config=coverage_swarp_config, 
            hdu_prep_kwargs=coverage_prep_kwargs,
        )
        try:
            df = cov_builder.mosaic_stacks.copy()
            df = df.explode("ccds")
            summary = df.query("tile==@tile").groupby(["pointing", "ccds"]).size()
            minimum_coverage = summary.min()
        except Exception as e:
            logger.warning(f"error in min coverage:\n{e}")
            summary = cov_builder.mosaic_stacks.groupby(["pointing"]).size()
            minimum_coverage = summary.min()
        print(f"minimum_coverage is {minimum_coverage} ")
        cov_builder.build(n_cpus=n_cpus)
        #cov_builder.add_extra_keys()
        good_cov_path = cov_builder.mosaic_path.with_suffix(".good_cov.fits")
        make_good_coverage_map(
            cov_builder.mosaic_path, output_path=good_cov_path, minimum_coverage=minimum_coverage       
        )
        if bright_star_processor is not None:
            region_masks = bright_star_processor.process_region_masks(mag_col=tmass_mag)
            extreme_region_masks = extreme_star_processor.process_region_masks(mag_col=tmass_mag)
            mask_regions_in_mosaic(good_cov_path, region_masks)
            mask_regions_in_mosaic(good_cov_path, extreme_region_masks, expand=10000)

        try:
            weight_path = cov_builder.mosaic_path.with_suffix(".weight.fits")
            logger.info(f"Removing {weight_path}")
            os.remove(weight_path)
        except:
            logger.warn(f"Can't remove {weight_path}")
        ### A little bit of clean up.
        try:
            with fits.open(cov_builder.mosaic_path) as f:
                header = f[0].header # check we can open the file!
            with open(cov_builder.swarp_list_path) as f:
                hdu_list = f.read().splitlines()
                for hdu_path_str in hdu_list:
                    hdu_path = Path(hdu_path_str)
                    assert hdu_path.exists()
                    os.remove(hdu_path)
                    assert not hdu_path.exists()
        except Exception as e:
            logger.warn(f"during delete temp: {e}")
            pass

    if exptime:
        print_header("make exptime map")
        exptime_swarp_config = {
            "center": builder.swarp_config["center"], # don't calculate twice!
            "image_size": builder.swarp_config["image_size"], # don't calculate twice!
        }
        exptime_prep_kwargs = {"hdu_prefix": f"{stem}_e"}
        pixel_scale = survey_config["mosaics"]["pixel_scale"]# * 10.0
        exptime_builder = MosaicBuilder.exptime_from_dxs_spec(
            *spec, 
            #pixel_scale=pixel_scale * 10.,
            include_deprecated_stacks=survey_config["mosaics"]["include_deprecated_stacks"],
            swarp_config=exptime_swarp_config, 
            hdu_prep_kwargs=exptime_prep_kwargs,
        )
        exptime_builder.build(n_cpus=n_cpus)
        #exptime_builder.add_extra_keys()

        try:
            weight_path = exptime_builder.mosaic_path.with_suffix(".weight.fits")
            logger.info(f"Removing {weight_path}")
            os.remove(weight_path)
        except:
            logger.warn(f"Can't remove {weight_path}")
        ### A little bit of clean up.
        try:
            with fits.open(exptime_builder.mosaic_path) as f:
                header = f[0].header # check we can open the file!
            with open(exptime_builder.swarp_list_path) as f:
                hdu_list = f.read().splitlines()
                for hdu_path_str in hdu_list:
                    hdu_path = Path(hdu_path_str)
                    assert hdu_path.exists()
                    os.remove(hdu_path)
                    assert not hdu_path.exists()
        except Exception as e:
            logger.warn(f"during delete temp: {e}")
            pass
                
    if masked:
        ### get segementation image to use as mask.
        print_header("segment to get mask")
        seg_name = builder.mosaic_path.stem
        mosaic_dir = paths.get_mosaic_dir(*spec)
        catalog_dir = paths.get_mosaic_dir(*spec) # so we can store the mask with the mosaic.
        seg_catalog_name = catalog_dir / f"{seg_name}_segmentation.cat.fits"
        seg_config = {"checkimage_name": mosaic_dir / f"{seg_name}_mask.seg.fits"}
        extractor = CatalogExtractor(
            builder.mosaic_path,
            catalog_path=seg_catalog_name,
            sextractor_config=seg_config, 
            sextractor_config_file=paths.config_path / f"sextractor/segmentation.sex",
            sextractor_parameter_file=paths.config_path / f"sextractor/segmentation.param",
        )
        extractor.extract()

        ### make masked mosaic
        logger.info(f"mask at {extractor.segmentation_mosaic_path.stem}")
        with fits.open(extractor.segmentation_mosaic_path) as f:
            mask_map = f[0].data
            mask_header = f[0].header.copy()

        print_header("make source-masked mosaic")
        masked_swarp_config = {
            "center": builder.swarp_config["center"], # don't calculate twice!
            "image_size": builder.swarp_config["image_size"], # don't calculate twice!
            "subtract_back": "N", # This is for the SWARP
        }        
        masked_prep_kwargs = {
            "hdu_prefix": f"{stem}_m",
            #resize=True, edges=25.0,
            "mask_sources": True, "mask_header": mask_header, "mask_map": mask_map,
            "subtract_bgr": True, # This is for the HDUPREP - see masked_swarp_config.
            "bgr_size": survey_config["mosaics"]["masked_bgr_size"],
            "overwrite": False,
        }
        masked_builder = MosaicBuilder.from_dxs_spec(
            *spec, 
            swarp_config=masked_swarp_config,
            include_deprecated_stacks=survey_config["mosaics"]["include_deprecated_stacks"],
            hdu_prep_kwargs=masked_prep_kwargs
        )
        if masked_builder is None:
            print("masked builder is None?!")
            sys.exit()
        masked_builder.build(n_cpus=n_cpus,)
        #masked_builder.add_extra_keys(magzpt_inc_exptime=True)

        try:
            with fits.open(masked_builder.mosaic_path) as f:
                header = f[0].header # check we can open the file!
            with open(masked_builder.swarp_list_path) as f:
                hdu_list = f.read().splitlines()
                for hdu_path_str in hdu_list[20:]:
                    hdu_path = Path(hdu_path_str)
                    assert hdu_path.exists()
                    os.remove(hdu_path)
                    assert not hdu_path.exists()
        except Exception as e:
            logger.warn(f"during delete temp: {e}")
            pass
                

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("band")
    parser.add_argument("--n_cpus", type=int)
    parser.add_argument("--initial", action="store_true", default=False)
    parser.add_argument("--coverage", action="store_true", default=False)
    parser.add_argument("--exptime", action="store_true", default=False)
    parser.add_argument("--masked", action="store_true", default=False)

    args = parser.parse_args()

    mosaic_pipeline(
        args.field, args.tile, args.band, n_cpus=args.n_cpus, 
        initial=args.initial, coverage=args.coverage, exptime=args.exptime, masked=args.masked
    )
    print("Done!")

    
