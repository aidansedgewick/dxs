import json
import logging
import os
import sys
import yaml
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("main")

from dxs.pipelines.mosaic_pipeline import mosaic_pipeline
from dxs.pipelines.photometry_pipeline import photometry_pipeline


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("--n_cpus", type=int)

    args = parser.parse_args()


    mosaic_pipeline(args.field, args.tile, "J", n_cpus=args.n_cpus)
    mosaic_pipeline(args.field, args.tile, "K", n_cpus=args.n_cpus)
    mosaic_pipeline(args.field, args.tile, "H", n_cpus=args.n_cpus)
    photometry_pipeline(args.field, args.tile, prefix="sm", n_cpus=args.n_cpus)
