import json
import yaml
from pathlib import Path

import astropy.io.fits as fits
from astromatic_wrapper.api import Astromatic

from dxs import MosaicBuilder, Stilts
from dxs.utils.misc import check_modules, format_astromatic_flags

from dxs import paths

class CatalogExtractor:
    """
    Class for building catalogs. Calls SExtractor.

    """
    def __init__(
        self, field, tile, detection_band, measurement_band=None, prefix=None,
        sextractor_config=None, sextractor_config_file=None, sextractor_parameter_file=None,
    ):
        check_modules("sex") # do we have sextractor available?
        self.field = field
        self.tile = tile
        self.detection_band = detection_band
        self.measurement_band = measurement_band
        # work out where the mosaics will live
        detection_mosaic_dir = MosaicBuilder.get_mosaic_dir(field, tile, detection_band)
        detection_mosaic_stem = MosaicBuilder.get_mosaic_stem(field, tile, detection_band, prefix)
        self.detection_mosaic_path = detection_mosaic_dir / f"{detection_mosaic_stem}.fits"
        if measurement_band is not None:
            measurement_mosaic_dir = MosaicBuilder.get_mosaic_dir(field, tile, measurement_band)
            measurement_mosaic_stem = MosaicBuilder.get_mosaic_name(field, tile, measurement_band)
            self.measurement_mosaic_path = measurement_mosaic_dir / f"{measurement_mosaic_stem}.fits"
        else:
            self.measurement_mosaic_path = None
        # workout where the catalogs should live.
        self.catalog_dir = self.get_catalog_dir(field, tile, detection_band)
        self.catalog_dir.mkdir(exist_ok=True, parents=True)
        self.catalog_stem = self.get_catalog_stem(
            field, tile, detection_band, measurement_band=measurement_band
        )
        self.catalog_path = self.catalog_dir / f"{self.catalog_stem}.cat"
        self.segmentation_mosaic_path = (
            detection_mosaic_dir / f"{detection_mosaic_stem}.seg.fits"
        )
        # keep the parameter files.
        self.sextractor_config = sextractor_config or {}
        self.sextractor_run_parameters_path = self.catalog_dir / "sextractor_run_parameters.json"
        self.sextractor_config_file = (
            sextractor_config_file or paths.config_path / "sextractor/indiv.sex"
        )
        self.sextractor_parameter_file = (
            sextractor_parameter_file or paths.config_path / "sextractor/indiv.param"
        )

    @staticmethod
    def get_catalog_dir(field, tile, detection_band):
        return paths.catalogs_path / CatalogExtractor.get_catalog_stem(field, tile, detection_band)

    @staticmethod
    def get_catalog_stem(field, tile, detection_band, measurement_band=None, prefix=None):
        prefix = prefix or ''
        measurement_band = measurement_band or ''
        return f"{prefix}{field}{tile:02d}{detection_band}{measurement_band}"

    def extract(self):
        config = self.build_sextractor_config()
        config.update(self.sextractor_config)
        config = format_astromatic_flags(config) # this capitalises stuff too.
        print(config)
        self.sextractor = Astromatic(
            "SExtractor", 
            str(paths.temp_sextractor_path), # I think Astromatic ignores anyway?!
            config=config, # command line flags.
            config_file=str(self.sextractor_config_file),
        )
        filenames = [str(self.detection_mosaic_path)]
        if self.measurement_mosaic_path is not None:
            filenames.append(str(self.detection_mosaic_path))
        self.sextractor.run(filenames)
        with open(self.sextractor_run_parameters_path, "w+") as f:
            json.dump(kwargs, f, indent=2)

    def build_sextractor_config(self,):
        config = {}
        config["catalog_name"] = self.catalog_path
        config["parameters_name"] = self.sextractor_parameter_file
        config["checkimage_type"] = "SEGMENTATION"
        config["checkimage_name"] = self.segmentation_mosaic_path
        with fits.open(self.detection_mosaic_path) as mosaic:
            header = mosaic[0].header
            config["seeing_fwhm"] = header["SEEING"]
            config["mag_zeropoint"] = header["MAGZPT"]
            config["starnnw_name"] = paths.config_path / "sextractor/default.nnw"
            config["filter_name"] = paths.config_path / "sextractor/gauss_5.0_9x9.conv"
        if self.measurement_mosaic_path is not None:
            with fits.open(self.measurement_mosaic_path) as mosaic:
                header = mosaic[0].header
                config["mag_zeropoint"] = header["MAGZPT"]
        return config

#class CatalogMatcher:
#    def __init__(self,):
#        pass
#    def 

if __name__ == "__main__":
    extractor = CatalogExtractor("EN", 4, "J")
    extractor.extract()







