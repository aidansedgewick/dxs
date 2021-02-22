import logging.config
import yaml

from .catalog_builder import (
    CatalogExtractor, 
    CatalogMatcher, 
    CatalogPairMatcher, 
    merge_catalogs
)
from .crosstalk_processor import CrosstalkProcessor
from .mosaic_builder import (
    MosaicBuilder, 
    HDUPreparer, 
    calculate_mosaic_geometry
)
from .pystilts import Stilts
from .quick_plotter import QPlot, QuickPlotter

from dxs import paths

default_logging_config = paths.config_path / "default_logging_config.yaml"
if default_logging_config.exists():
    with open(default_logging_config, "rt") as f:
        log_config = yaml.safe_load(f.read())
    logging.config.dictConfig(log_config)
