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

#from dxs.mosaic_builder import get_mosaic_dir, get_mosaic_stem
from dxs.pystilts import Stilts
from dxs.utils.misc import check_modules, format_flags, create_file_backups
from dxs.utils.table import fix_column_names, table_to_numpynd

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
        check_modules("sex")
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
            catalog_path = Path.cwd() / f"{d_stem}{m_stem}.fits"
        self.catalog_path = Path(catalog_path)
        self.use_weight = use_weight
        if weight_path is None:
            weight_path = self.detection_mosaic_path.with_suffix(".weight.fits")
        self.weight_path = weight_path

        self.aux_dir = catalog_path.parent / "aux"
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
        cls, field, tile, detection_band, measurement_band=None, prefix=None, catalog_stem=None,
        sextractor_config=None, sextractor_config_file=None, sextractor_parameter_file=None,
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
        catalog_path = catalog_dir / f"{catalog_stem}.fits"
        return cls(
            detection_mosaic_path, 
            measurement_mosaic_path=measurement_mosaic_path,
            catalog_path=catalog_path,
            sextractor_config=sextractor_config,
            sextractor_config_file=sextractor_config_file,
            sextractor_parameter_file=sextractor_parameter_file,
        )

    def extract(self):
        logger.info(f"extract catalog to {self.catalog_path}")
        config = self.build_sextractor_config()
        config.update(self.sextractor_config) # overwrite the inbuilt stuff with the input.
        config = format_flags(config) # this capitalises stuff too.
        self.sextractor = Astromatic(
            "SExtractor", 
            str(paths.temp_sextractor_path), # I think Astromatic() ignores anyway?!
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
        return config

    def add_snr(
        self, flux, flux_err=None, catalog_path=None, snr_name="snr", 
        flux_format=None, flux_err_format=None, snr_format=None, nan_value=0.0
    ):
        # TODO: this is unpleasant - fix!
        """
        fix
        """

        if catalog_path is None:
            catalog_path = self.catalog_path
        if not isinstance(flux, list):
            flux = [flux]
        if not isinstance(flux_err, list):
            if flux_err is not None:
                flux_err = [flux_err]
            else:
                flux_err = [None for _ in flux]
        if len(flux_err) != len(flux):
            lf, lfe = len(flux), len(flux_err)
            raise ValueError(
                f"Provide same number of flux columns ({lf}) as flux_err columns ({lfe})"
            )
        if not isinstance(snr_name, list):
            snr_name = [f"{fl}_{snr_name}" for fl in flux]
        if len(snr_name) != len(flux):
            lf, lsnr = len(flux), len(snr_name)
            raise ValueError(
                f"Provide same number of flux columns ({lf}) as flux_err columns ({lsnr})"
            )            
        catalog = Table.read(catalog_path)
        for fl, fl_err, snr in zip(flux, flux_err, snr_name):
            if flux_format is not None:
                flux_col = flux_format.format(**{"flux": fl, "flux_err": fl_err})
            else:
                flux_col = fl
            if flux_err_format is not None:
                flux_err_col = flux_err_format.format(**{"flux": fl, "flux_err": fl_err})
            else:
                flux_err_col = fl_err
            if snr_format is not None:
                snr_col = snr_format.format(
                    **{"flux": fl, "flux_err": fl_err, "snr_name": snr_name}
                )
            else:
                snr_col = snr
            catalog[snr_col] = catalog[flux_col] / catalog[flux_err_col]
            nan_mask = np.isnan(catalog[snr_col])
            n_nans = nan_mask.sum()
            if n_nans > 0:
                logger.warn(f"add_snr - {flux_col} give {n_nans} nan values")
            catalog[snr_col][ nan_mask ] = nan_value
        catalog.write(catalog_path, overwrite=True)

    def add_map_value(
        self, mosaic_path, column_name, ra=None, dec=None, xpix=None, ypix=None, hdu=0,
    ):
        logger.info("adding map value")
        info = _add_map_value(
            self.catalog_path, mosaic_path, column_name, 
            ra=ra, dec=dec, xpix=xpix, ypix=ypix, hdu=hdu,        
        )
        logger.info(info)

    def add_column(self, column_data: Dict):
        info = _add_column(self.catalog_path, column_data)
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
    output_path
        path to output after matchers are made
    ra, dec
        the columns for the coordinates to use in the main catalog when doing matches.

    >>> cat_matcher = CatalogMatcher(
            "./catalog.fits", output_path="./catalog_with_matches.fits", 
            ra="best_ra", dec="best_ra",
        )
    >>> cat_matcher.match_catalog("./gaia_data.fits", ra="gaia_ra", dec="gaia_dec")

    """
    
    def __init__(self, catalog_path, output_path=None, ra="ra", dec="dec"):
        self.catalog_path = Path(catalog_path)
        output_path = output_path or self.catalog_path
        self.output_path = Path(output_path)
        self.ra = ra
        self.dec = dec

        self.summary_info = []

    @classmethod
    def from_dxs_spec(
        cls, field, tile, band, prefix=None, suffix=None, output_path=None, ra="ra", dec="dec"
    ):
        catalog_dir = paths.get_catalog_dir(field, tile, band)
        catalog_stem = paths.get_catalog_stem(field, tile, band, prefix=prefix)
        catalog_path = catalog_dir / f"{catalog_stem}.fits"
        suffix = suffix or ''
        prefix = prefix or ''
        if output_path is None:
            output_stem = f"{prefix}{catalog_stem}{suffix}"
            output_path = catalog_dir / f"{output_stem}.fits"
        return cls(catalog_path, output_path, ra=ra, dec=dec)

    def match_catalog(
        self, extra_catalog, ra="ra", dec="dec", prefix=None, output_path=None, **kwargs
    ):
        prefix = prefix or extra_catalog.stem
        extra_catalog = Path(extra_catalog)
        output_path = output_path or self.output_path
        stilts = Stilts.tskymatch2_fits(
            self.catalog_path, extra_catalog, output_path=output_path,
            flags={"join": "all1", "find": "best"},
            ra1=self.ra, dec1=self.dec,
            ra2=ra, dec2=dec,
            **kwargs
        )
        stilts.run()
        info = f"{stilts.flags['join']} match {extra_catalog.name}"
        logger.info(info)
        self.summary_info.append(info)

    def join_catalog(
        self, extra_catalog, values1, values2, engine="stilts", **kwargs
    ):
        """remember kwarg values overwrite flags value..."""
        if engine=="stilts":
            stilts = Stilts.tskymatch2_fits(
                self.output_catalog, extra_catalog, values1=values1, values2=values2,
                flags={"join": "all1", "find": "best"}, **kwargs
            )
            stilts.run()
        elif engine=="astropy":
            raise NotImplementedError("DIY for now - or just use tskymatch2_fits.")

    def add_map_value(
        self, mosaic_path, column_name, ra=None, dec=None, xpix=None, ypix=None, hdu=0,
    ):
        info = _add_map_value(
            self.output_path, mosaic_path, column_name, 
            ra=ra, dec=dec, xpix=xpix, ypix=ypix, hdu=hdu,        
        )
        logger.info(info)
        self.summary_info.append(info)

    def add_column(self, column_data: Dict):
        _add_column(self.output_path, column_data)

    def fix_column_names(self, **kwargs):
        fix_column_names(self.output_path, **kwargs)

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
    ra1, dec1
        name of ra, dec column in catalog 1
    ra2, dec2
        name of ra, dec column in catalog 1
    output_ra, output_dec
        the name of the column to be created when selecting the best coordinate pair 
        value to use from the above column.
    
    >>> pair_matcher = CatalogPairMatcher(
            "Jcat.fits", "Kcat.fits", "./matched_cat.fits", ra1="Jra", dec1="Jdec"

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

    @classmethod
    def from_dxs_spec(
        cls, field, tile, output_path, prefix=None, suffix=None,
    ):
        catalog1_dir = paths.get_catalog_dir(field, tile, "J")
        catalog1_stem = paths.get_catalog_stem(field, tile, "J", prefix=prefix)
        catalog1_path = catalog1_dir / f"{catalog1_stem}.fits"
        catalog2_dir = paths.get_catalog_dir(field, tile, "K")
        catalog2_stem = paths.get_catalog_stem(field, tile, "K", prefix=prefix)
        catalog2_path = catalog2_dir / f"{catalog2_stem}.fits"
        return cls(
            catalog1_path, catalog2_path, output_path,
            ra1="J_ra", dec1="J_dec", prefix1="J", suffix1=suffix,
            ra2="K_ra", dec2="K_dec", prefix2="K", suffix2=suffix,
        )

    def best_pair_match(self, **kwargs):
        stilts = Stilts.tskymatch2_fits(
            self.catalog1_path, self.catalog2_path, self.catalog_path,
            ra1=self.ra1, dec1=self.dec1, ra2=self.ra2, dec2=self.dec2,
            flags={"join": "1or2", "find": "best"},
            **kwargs
        )
        stilts.run()
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
        assert len(catalog[ catalog["ra"] < -90.0 ]) == 0
        assert len(catalog[ catalog["dec"] < -90.0 ]) == 0
        catalog.write(self.catalog_path, overwrite=True)
        self.ra = output_ra
        self.dec = output_dec

def _add_map_value(
    catalog_path, mosaic_path, column_name, ra=None, dec=None, xpix=None, ypix=None, hdu=0
):
    catalog = Table.read(catalog_path)
    with fits.open(mosaic_path) as mosaic:
        logger.info(f"open {mosaic_path}")
        mosaic_data = mosaic[hdu].data
        header = mosaic[hdu].header
    use_coords = all([ra, dec])
    use_pixels = all([xpix, ypix])
    err_msg = "Must provide xpix, ypix column names OR ra, dec column names -- not both."
    if use_coords and use_pixels:
        raise ValueError(err_msg)
    if use_coords:
        mosaic_wcs = WCS(header)
        image_positions = table_to_numpynd( catalog[[ra, dec]] )  # TODO: use SkyCoord?
        pixels = mosaic_wcs.wcs_world2pix(image_positions, 0)
        x_values = pixels[:,0].astype(int)
        y_values = pixels[:,1].astype(int)
    elif xpix is not None and ypix is not None:
        x_values = catalog[xpix].astype(int)
        y_values = catalog[ypix].astype(int)
    else:
        raise ValueError(err_msg)
    map_values = mosaic_data[ y_values, x_values ]
    col = Column(map_values, column_name)
    catalog.add_column(col)
    catalog.write(catalog_path, overwrite=True)
    info = f"added map data {column_name} from {mosaic_path.stem}"
    return info

def _add_column(catalog_path, column_data: Dict):
    catalog = Table.read(catalog_path)
    for column_name, column_values in column_data.items():
        catalog.add_column(column_values, name=column_name)
    catalog.write(catalog_path, overwrite=True)
    return f"add {len(column_data)} columns: " + " ".join(c for c in column_data.keys())





if __name__ == "__main__":
    extractor = CatalogExtractor("EN", 4, "J")
    extractor.extract()

    star_table_path = (
        paths.input_data_path / "external/tmass/tmass_ElaisN1_stars.csv"
    )








