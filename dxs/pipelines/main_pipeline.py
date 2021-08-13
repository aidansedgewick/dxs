import json
import logging
import os
import sys
import yaml
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dxs.utils.misc import tile_parser
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("main")

from dxs.pipelines.mosaic_pipeline import mosaic_pipeline
from dxs.pipelines.photometry_pipeline import photometry_pipeline
from dxs.pipelines.merge_pipeline import merge_pipeline
from dxs.pipelines.field_mask_pipeline import field_mask_pipeline

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("--n_cpus", type=int)

    args = parser.parse_args()

    mosaic_pipeline(args.field, args.tile, "J", n_cpus=args.n_cpus)
    mosaic_pipeline(args.field, args.tile, "K", n_cpus=args.n_cpus)
    mosaic_pipeline(args.field, args.tile, "H", n_cpus=args.n_cpus)
    photometry_pipeline(args.field, args.tile, n_cpus=args.n_cpus)
    
    #tiles_in_field = survey_config["tiles_per_field"][args.field]
    tiles = survey_config["merge"]["default_tiles"].get(field, None)
    #[x for x in range(1, tiles_in_field+1)]
    try:
        merge_pipeline(args.field, tiles, ["J", "K"], require_all=True)
    except Exception as e:
        logger.info(f"Can't merge J,K:\n    {e}")
    try:
        field_mask_pipeline(args.field, tiles, "J")
    except Exception as e:
        logger.info(f"Can't make J field mask:\n    {e}")
    try:
        field_mask_pipeline(args.field, tiles, "K")
    except Exception as e:
        logger.info(f"Can't make K field mask:\n    {e}")

    print("Done!")
