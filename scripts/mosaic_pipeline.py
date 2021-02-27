import json
import logging
import os
import sys
import yaml
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from dxs import (
    MosaicBuilder, 
    CatalogExtractor, 
    CatalogMatcher, 
    CatalogPairMatcher, 
    CrosstalkProcessor
)
from dxs.utils.misc import check_modules, print_header
from dxs.utils.table import fix_column_names
from dxs.utils.image import scale_mosaic, make_good_coverage_map
from dxs.quick_plotter import QuickPlotter
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("main")

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("band")
    parser.add_argument("--n_cpus", type=int)
    parser.add_argument("--initial", action="store_true", default=False)
    parser.add_argument("--coverage", action="store_true", default=False)
    parser.add_argument("--masked", action="store_true", default=False)

    args = parser.parse_args()
    spec = (args.field, args.tile, args.band)

    logger.info(f"Mosaic for {spec}, use {args.n_cpus} threads")
    check_modules("swarp", "sex")

    builder = MosaicBuilder.from_dxs_spec(*spec, prefix="s")
    prep_kwargs = {"hdu_prefix": f"{builder.mosaic_path.stem}_"}
    if builder is None: # ie, if there are no stacks to build
        logger.info("Builder is None. Exiting")
        sys.exit()
    if args.initial:
        print_header("make initial mosaic")
        builder.build(n_cpus=args.n_cpus)
        builder.add_extra_keys()

    if args.coverage:
        print_header("make coverage mosaic")
        coverage_swarp_config = {
            "center": builder.swarp_config["center"], # don't calculate twice!
            "image_size": builder.swarp_config["image_size"], # don't calculate twice!
        }
        coverage_prep_kwargs = {"hdu_prefix": f"{builder.mosaic_path.stem}_u"}
        pixel_scale = survey_config["mosaics"]["pixel_scale"]# * 10.0
        cov_builder = MosaicBuilder.coverage_from_dxs_spec(*spec, pixel_scale=pixel_scale)
        cov_builder.build(n_cpus=args.n_cpus)
        cov_builder.add_extra_keys() 
        scale_mosaic(
            cov_builder.mosaic_path, value=1., save_path=cov_builder.mosaic_path, round_val=0
        )
        good_cov_path = cov_builder.mosaic_path.with_suffix(".good_cov.fits")
        mask_good_coverage_map(cov_builder.mosaic_path, output_path=good_cov_path)
        try:
            weight_path = cov_builder.mosaic_path.with_suffix(".weight.fits")
            logger.info(f"Removing {weight_path}")
            os.remove(weight_path)
        except:
            logger.warn(f"Can't remove {weight_path}")

    if args.masked:
        ### get segementation image to use as mask.
        print_header("segment to get mask")
        seg_name = paths.get_mosaic_stem(*spec)
        mosaic_dir = paths.get_mosaic_dir(*spec)
        catalog_dir = paths.get_mosaic_dir(*spec) # so we can store the mask with the mosaic.
        seg_config = {
            "checkimage_name": mosaic_dir / f"{seg_name}_mask.seg.fits",
            "catalog_name": catalog_dir / f"{seg_name}_segmenation.fits",
        }
        extractor = CatalogExtractor.from_dxs_spec(
            *spec, 
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
        }        
        masked_prep_kwargs = {
            "hdu_prefix": paths.get_mosaic_stem(*spec, prefix="m"),
            #resize=True, edges=25.0,
            "mask_sources": True, "mask_header": mask_header, "mask_map": mask_map,
            "normalise_exptime": True,
            "subtract_bgr": True, "bgr_size": 32,
            "overwrite": False,
        }
        masked_builder = MosaicBuilder.from_dxs_spec(
            *spec, prefix="sm", swarp_config=masked_swarp_config, prep_kwargs=masked_prep_kwargs
        )
        if masked_builder is None:
            sys.exit()
        masked_builder.build(n_cpus=args.n_cpus,)
        masked_builder.add_extra_keys(magzpt_inc_exptime=False)
    
