import json
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
from dxs.utils.misc import check_modules
from dxs.utils.table import fix_column_names, fix_sextractor_column_names
from dxs.utils.image import scale_mosaic
from dxs.quick_plotter import QuickPlotter, Plot
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("band")
    parser.add_argument("--n_cpus", type=int)

    args = parser.parse_args()

    spec = (args.field, args.tile, args.band)

    print(f"mosaic for {spec} with {args.n_cpus}")
    
    builder = MosaicBuilder.from_dxs_spec(
        args.field, args.tile, args.band, n_cpus=args.n_cpus
    )
    builder.build()
    builder.add_extra_keys()

    pixel_scale = survey_config["pixel_scale"] * 10.0
    cov_builder = MosaicBuilder.coverage_from_dxs_spec(
        args.field, args.tile, args.band, pixel_scale=pixel_scale, n_cpus=args.n_cpus
    )
    cov_builder.build(value=1.0, hdu_prefix="ones_")
    cov_builder.add_extra_keys() 
    scale_mosaic(
        cov_builder.mosaic_path, value=1., save_path=cov_builder.mosaic_path, round_val=0
    )
    

    seg_name = paths.get_mosaic_stem(*spec)
    mosaic_dir = paths.get_mosaic_dir(*spec)
    catalog_dir = paths.get_catalog_dir(*spec)
    seg_config = {
        "checkimage_name": mosaic_dir / f"{seg_name}_mask.seg.fits"
        "catalog_name": catalog_dir / f"{seg_name}_segmenation.cat"
    }
    
    extractor = CatalogExtractor.from_dxs_spec(
        *spec, 
        sextractor_config=seg_config, 
        sextractor_config_file=paths.config_path / f"segmentation.sex"
        sextractor_parameter_file=paths.config_path / f"segmentation.param"
    )
    extractor.extract()

    with fits.open(extractor.segmentation_mosaic_path) as f:
        mask_map = f[0].data
        mask_wcs = WCS(f[0].header)
    
    masked_builder = MosaicBuilder.from_dxs_spec(
        args.field, args.tile, args.band, prefix="m", n_cpus=args.n_cpus
    )
    masked_builder.build(
        hdu_prefix="m",
        #resize=True, edges=25.0,
        mask_sources=True, mask_wcs=mask_wcs, mask_map=mask_map,
        normalise_exptime=True,
        subtract_bgr=True, bgr_size=32,           
    )
    masked_builder.add_extra_keys(normalise_exptime=True)
    
