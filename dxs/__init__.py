import logging.config
import yaml

from .bright_star_processor import BrightStarProcessor #, stack_bright_stars
from .catalog_builder import (
    CatalogExtractor, 
    CatalogMatcher, 
    CatalogPairMatcher
)
from .catalog_merge import merge_catalogs
from .crosstalk_processor import CrosstalkProcessor
from .hdu_preprocessor import HDUPreprocessor
from .mosaic_builder import (
    MosaicBuilder, 
    calculate_mosaic_geometry

)
#from .photoz_processor import PhotozProcessor
#from .quick_plotter import QPlot, QuickPlotter

from dxs import paths

default_logging_config = paths.config_path / "default_logging_config.yaml"
if default_logging_config.exists():
    with open(default_logging_config, "rt") as f:
        log_config = yaml.safe_load(f.read())
    logging.config.dictConfig(log_config)
