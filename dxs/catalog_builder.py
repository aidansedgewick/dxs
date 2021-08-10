import json
import logging
import shutil
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column, vstack
from astropy.wcs import WCS

from astromatic_wrapper.api import Astromatic
from easyquery import Query

from stilts_wrapper import Stilts

#from dxs.mosaic_builder import get_mosaic_dir, get_mosaic_stem
from dxs.utils.misc import check_modules, format_flags, create_file_backups
from dxs.utils.table import (
    fix_column_names, add_map_value_to_catalog, add_column_to_catalog
)

from dxs import paths

logger = logging.getLogger("catalog_builder")

class CatalogExtractorError:
    pass

class CatalogExtractor:
    """
    Class for building catalogs. Calls SExtractor.
    Parameters
    ----------
    detection_mosaic_path
    measurement_mosaic_path
        for use in SExtractor dual-image mode.
    catalog_path
        where to store the output? defaults to config's "catalog_dir" / [dectection_image].fits
    sextractor_config
        command line arguments to overwrite whatever is in config file.
    sextractor_config_file
        
    """

    def __init__(
        self, 
        detection_mosaic_path, 
        measurement_mosaic_path=None, 
        weight_path=None,
        use_weight=True,
        catalog_path=None,
        sextractor_config=None, 
        sextractor_config_file=None, 
        sextractor_parameter_file=None,
    ):
        self.detection_mosaic_path = Path(detection_mosaic_path)
        d_stem = self.detection_mosaic_path.stem
        detection_mosaic_dir = self.detection_mosaic_path.parent
        if measurement_mosaic_path is not None:
            self.measurement_mosaic_path = Path(measurement_mosaic_path)
            m_stem = self.measurement_mosaic_path.stem
        else:
            self.measurement_mosaic_path = None
            m_stem = ''
        if catalog_path is None:
            catalog_dir = paths.catalogs_path / f"{d_stem}"
            catalog_path = catalog_dir / f"{d_stem}{m_stem}.cat.fits"
        else:
            catalog_path = Path(catalog_path)
            catalog_dir = catalog_path.parent
        catalog_dir.mkdir(exist_ok=True, parents=True)
        self.catalog_path = Path(catalog_path)
        self.use_weight = use_weight
        if weight_path is None:
            weight_path = self.detection_mosaic_path.with_suffix(".weight.fits")
        self.weight_path = weight_path

        self.aux_dir = catalog_dir / "aux"
        self.aux_dir.mkdir(exist_ok=True, parents=True)
        
        # keep the parameter files.
        self.sextractor_config = sextractor_config or {}
        self.sextractor_run_parameters_path = (
            self.aux_dir / f"{catalog_path.stem}_se_run_parameters.json"
        )
        self.sextractor_config_file = (
            sextractor_config_file or paths.config_path / "sextractor/indiv.sex"
        )
        self.sextractor_parameter_file = (
            sextractor_parameter_file or paths.config_path / "sextractor/indiv.param"
        )

        self.segmentation_mosaic_path = self.sextractor_config.get(
            "checkimage_name",
            detection_mosaic_dir / f"{d_stem}.seg.fits"
        )

    @classmethod
    def from_dxs_spec(
        cls, 
        field, 
        tile, 
        detection_band, 
        measurement_band=None, 
        prefix=None, 
        catalog_stem=None,
        sextractor_config=None, 
        sextractor_config_file=None, 
        sextractor_parameter_file=None,
    ):
        """
        Build a catalog directly from field name, tile number, band. (eg EN 4 K)

        Parameters
        ----------
        field
        tile
        detection_band
            band for to act as SExtractor measurement image.
        measurement_band

        eg.
        >>> extractor = CatalogExtractor.from_dxs_spec("EN", 4, "K")
        >>> extractor.extract()
        
        """
        
        # work out where the mosaics will live
        detection_mosaic_dir = paths.get_mosaic_dir(field, tile, detection_band)
        detection_mosaic_stem = paths.get_mosaic_stem(field, tile, detection_band, prefix)
        detection_mosaic_path = detection_mosaic_dir / f"{detection_mosaic_stem}.fits"
        if measurement_band is not None:
            measurement_mosaic_dir = paths.get_mosaic_dir(field, tile, measurement_band)
            measurement_mosaic_stem = paths.get_mosaic_stem(field, tile, measurement_band, prefix=prefix)
            measurement_mosaic_path = (
                measurement_mosaic_dir / f"{measurement_mosaic_stem}.fits"
            )
        else:
            measurement_mosaic_path = None
        # work out where the catalogs should live.
        catalog_dir = paths.get_catalog_dir(field, tile, detection_band)
        catalog_dir.mkdir(exist_ok=True, parents=True)
        if catalog_stem is None:
            catalog_stem = paths.get_catalog_stem(
                field, tile, detection_band, measurement_band=measurement_band, prefix=prefix
            )
        catalog_path = catalog_dir / f"{catalog_stem}.cat.fits"
        return cls(
            detection_mosaic_path, 
            measurement_mosaic_path=measurement_mosaic_path,
            catalog_path=catalog_path,
            sextractor_config=sextractor_config,
            sextractor_config_file=sextractor_config_file,
            sextractor_parameter_file=sextractor_parameter_file,
        )

    def extract(self):
        check_modules("sex")
        logger.info(f"extract catalog to {self.catalog_path}")
        config = self.build_sextractor_config()
        self.sextractor = Astromatic(
            "SExtractor", 
            str(paths.scratch_sextractor_path), # I think Astromatic() ignores anyway?!
            config=config, # command line flags.
            config_file=str(self.sextractor_config_file),
        )
        filenames = [str(self.detection_mosaic_path)]
        if self.measurement_mosaic_path is not None:
            filenames.append(str(self.measurement_mosaic_path))
        filenames = ",".join(f for f in filenames)
        cmd, cmd_kwargs = self.sextractor.build_cmd(filenames)
        cmd_kwargs["cmd"] = cmd
        self.sextractor.run(filenames)   
        with open(self.sextractor_run_parameters_path, "w+") as f:
            json.dump(cmd_kwargs, f, indent=2)
        
    def build_sextractor_config(self,):
        config = {}
        config["catalog_name"] = self.catalog_path
        config["parameters_name"] = self.sextractor_parameter_file
        config["checkimage_type"] = "SEGMENTATION"
        config["checkimage_name"] = self.segmentation_mosaic_path
        config["weight_type"] = "background"
        if self.use_weight:
            config["weight_image"] = self.weight_path
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

        config.update(self.sextractor_config) # overwrite the inbuilt stuff with the input.
        config = format_flags(config) # this capitalises stuff too.
        return config

    def add_snr(self, flux_columns, err_columns, snr_columns, nan_value=0.0):
        if not isinstance(flux_columns, list):
            flux_columns = [flux_columns]
        if not isinstance(err_columns, list):
            err_columns = [err_columns]
        if not isinstance(snr_columns, list):
            snr_columns = [snr_columns]
        if len(flux_columns) != len(err_columns):
            raise ValueError("provide as many flux_columns as err_columns")
        if len(flux_columns) != len(snr_columns):
            raise ValueError("provide as many flux_columns as snr_columns")
    
        catalog = Table.read(self.catalog_path)
        for fl, err, snr in zip(flux_columns, err_columns, snr_columns):
            if snr in catalog.columns:
                raise ValueError(f"{snr} already exists.")
            catalog[snr] = catalog[fl] / catalog[err]
            
            nan_mask = np.isnan(catalog[snr])
            if sum(nan_mask) > 0:
                logger.warning(f"{sum(nan_mask)} NaN values. set to {nan_value}")
                catalog[snr][nan_mask] = nan_value
        catalog.write(self.catalog_path, overwrite=True)

    def add_map_value(
        self, mosaic_path, column_name, ra=None, dec=None, xpix=None, ypix=None, hdu=0,
    ):
        logger.info("adding map value")
        info = add_map_value_to_catalog(
            self.catalog_path, mosaic_path, column_name, 
            ra=ra, dec=dec, xpix=xpix, ypix=ypix, hdu=hdu,        
        )
        logger.info(info)

    def add_column(self, column_data: Dict):
        info = add_column_to_catalog(self.catalog_path, column_data)
        logger.info(info)

class CatalogMatcher:
    """
    Class for matching one catalog to another. Calls stilts.
    Start with a "main" catalog, and match extras to it.

    Default to best (left) match for the main catalog.

    Parameters
    ----------
    catalog_path
        path to main catalog
    ra, dec
        the columns for the coordinates to use in the main catalog when doing matches.

    >>> cat_matcher = CatalogMatcher(
            "./catalog.cat.fits", output_path="./catalog_with_matches.cat.fits", 
            ra="best_ra", dec="best_ra",
        )
    >>> cat_matcher.match_catalog("./gaia_data.fits", ra="gaia_ra", dec="gaia_dec")

    """
    
    def __init__(self, catalog_path, ra="ra", dec="dec"):
        self.catalog_path = Path(catalog_path)
        self.ra = ra
        self.dec = dec

        self.summary_info = []

    def match_catalog(
        self, 
        extra_catalog, 
        output_path, 
        ra="ra", 
        dec="dec", 
        engine="stilts", 
        set_output_as_input=False,
        **kwargs
    ):
        extra_catalog = Path(extra_catalog)
        output_path = Path(output_path)       
        if self.catalog_path == output_path:
             logger.warning("catalog_path is the same as output_path!!")
        if engine == "stilts":
            flags = {"join": "all1", "find": "best"}
            flags.update(kwargs)
            self.stilts = Stilts.tskymatch2(
                in1=self.catalog_path, 
                in2=extra_catalog, 
                out=output_path,
                ra1=self.ra, dec1=self.dec,
                ra2=ra, dec2=dec,
                all_formats="fits",
                **flags
            )
            self.stilts.run()
            info = f"{self.stilts.parameters['join']} match {extra_catalog.name}"
        else:
            raise ValueError("currently only engine='stilts' implemented...")
        logger.info(info)
        self.summary_info.append(info)
        if set_output_as_input:
            self.catalog_path = output_path

    def print_summary(self):
        summary = "Summary:"
        for info in self.summary_info:
            summary += f"\n  -{info}"
        print(summary)


class CatalogPairMatcher(CatalogMatcher):
    """
    Class for matching catalogs. Calls stilts. Inherits (?) from CatalogMatcher.
    This means that once you've done the pair matching for the two catalogs, you
    can treat it like a CatalogMatcher and add extra catalogs on easily with the
    same methods, etc.

    Parameters
    ----------
    catalog1_path, catalog2_path
        catalogs to match
    output_path
        where to store the output?
    ra1, dec1
        name of ra, dec column in catalog 1
    ra2, dec2
        name of ra, dec column in catalog 1
    output_ra, output_dec
        the name of the column to be created when selecting the best coordinate pair 
        value to use from the above column.
    
    >>> pair_matcher = CatalogPairMatcher(
    ...       "J.cat.fits", "K.cat.fits", "./matched.cat.fits", ra1="Jra", dec1="Jdec"
    ... )
    """

    def __init__(
        self, catalog1_path, catalog2_path, output_path,
        ra1="ra", dec1="dec", ra2="ra", dec2="dec", output_ra="ra", output_dec="dec", 
    ):
        self.catalog1_path = Path(catalog1_path)
        self.catalog2_path = Path(catalog2_path)
        self.catalog_path = Path(output_path)
        self.output_path = Path(output_path)
        self.ra1 = ra1
        self.dec1 = dec1
        self.ra2 = ra2
        self.dec2 = dec2
        # Somewhere to keep the best values when we need them.
        self.output_ra = output_ra
        self.output_dec = output_dec
        self.summary_info = []
        logger.info(f"pair matcher - {self.catalog1_path.name} and {self.catalog2_path.name}")

    def best_pair_match(self, engine="stilts", **kwargs):
        if engine == "stilts":
            self.stilts = Stilts.tskymatch2(
                in1=self.catalog1_path, in2=self.catalog2_path, out=self.catalog_path,
                ra1=self.ra1, dec1=self.dec1, ra2=self.ra2, dec2=self.dec2,
                join="1or2", find="best",
                all_formats="fits",
                **kwargs                
            )
            self.stilts.run()
        else:
            raise ValueError("currently only engine='stilts' implemeneted...")
        info = f"best match {self.catalog1_path.name} and {self.catalog2_path.name}"
        logger.info(info)
        self.summary_info.append(info)
        
    def select_best_coords(
        self, snr1, snr2, output_ra=None, output_dec=None, 
        ra1=None, dec1=None, ra2=None, dec2=None, 
    ):
        """
        Choose ra1, dec1 as the "output_ra", "output_dec" if value of  snr1 > snr2
        snr_1
        Set this value 
        """
        catalog = Table.read(self.catalog_path)
        logger.info(f"pair cat has {len(catalog.colnames)} cols")
        output_ra = output_ra or self.output_ra
        output_dec = output_dec or self.output_dec
        catalog.add_column(-99.0, name=output_ra)
        catalog.add_column(-99.0, name=output_dec)

        ra1 = ra1 or self.ra1
        dec1 = dec1 or self.dec1
        ra2 = ra2 or self.ra2
        dec2 = dec2 or self.dec2
        
        keep_coord1 = (catalog[snr1] >= catalog[snr2]) | (np.isnan(catalog[snr2]))
        keep_coord2 = (catalog[snr2] >  catalog[snr1]) | (np.isnan(catalog[snr1]))
        assert np.sum(keep_coord1) + np.sum(keep_coord2) == len(catalog)
        # Select the right coordinates
        catalog[output_ra][ keep_coord1 ] = catalog[ra1][ keep_coord1 ]
        catalog[output_dec][ keep_coord1 ] = catalog[dec1][ keep_coord1 ]
        catalog[output_ra][ keep_coord2 ] = catalog[ra2][ keep_coord2 ]
        catalog[output_dec][ keep_coord2 ] = catalog[dec2][ keep_coord2 ]
        assert len(catalog[ catalog[output_ra] < -90.0 ]) == 0
        assert len(catalog[ catalog[output_dec] < -90.0 ]) == 0
        catalog.write(self.catalog_path, overwrite=True)
        self.ra = output_ra
        self.dec = output_dec







if __name__ == "__main__":
    extractor = CatalogExtractor("EN", 4, "J")
    extractor.extract()

    star_table_path = (
        paths.input_data_path / "external/tmass/tmass_ElaisN1_stars.csv"
    )








