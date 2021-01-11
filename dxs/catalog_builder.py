import json
import logging
import shutil
import yaml
from pathlib import Path

import astropy.io.fits as fits
from astropy.table import Table, Column
from astropy.wcs import WCS

from astromatic_wrapper.api import Astromatic

from dxs.crosstalk_processor import CrosstalkProcessor
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
    field
    tile
    detection_band
        band for to act as SExtractor measurement image.
    measurement_band

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
        detection_mosaic_dir = paths.get_mosaic_dir(field, tile, detection_band)
        detection_mosaic_stem = paths.get_mosaic_stem(field, tile, detection_band, prefix)
        self.detection_mosaic_path = detection_mosaic_dir / f"{detection_mosaic_stem}.fits"
        if measurement_band is not None:
            measurement_mosaic_dir = paths.get_mosaic_dir(field, tile, measurement_band)
            measurement_mosaic_stem = paths.get_mosaic_stem(field, tile, measurement_band)
            self.measurement_mosaic_path = (
                measurement_mosaic_dir / f"{measurement_mosaic_stem}.fits"
            )
        else:
            self.measurement_mosaic_path = None
        # workout where the catalogs should live.
        self.catalog_dir = paths.get_catalog_dir(field, tile, detection_band)
        self.catalog_dir.mkdir(exist_ok=True, parents=True)
        self.catalog_stem = paths.get_catalog_stem(
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
    """
    def __init__(
        self, detection_mosaic_path, measurement_mosaic_path=None, catalog_path=None,
        sextractor_config=None, sextractor_config_file=None, sextractor_parameter_file=None,
    ):
        self.detection_mosaic_path = Path(detection_mosaic_path)
        self.measurment_mosaic_path = Path(measurement_mosaic_path)
        if catalog_path is None:
            d = detection_mosaic_path.stem
            if measurement_mosaic_path is not None:
                m = f"_{measurement_mosaic_path.stem}
            else:
                m = ''
        
            catalog_path = Path.cwd() / f"{m}{d}.cat"
        self.catalog_path = Path(catalog_path)
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

    @classmethod
    def from_dxs_spec(
        cls, field, tile, detection_band, measurement_band=None, prefix=None,
        sextractor_config=None, sextractor_config_file=None, sextractor_parameter_file=None,
    ):
        # work out where the mosaics will live
        detection_mosaic_dir = paths.get_mosaic_dir(field, tile, detection_band)
        detection_mosaic_stem = paths.get_mosaic_stem(field, tile, detection_band, prefix)
        detection_mosaic_path = detection_mosaic_dir / f"{detection_mosaic_stem}.fits"
        if measurement_band is not None:
            measurement_mosaic_dir = paths.get_mosaic_dir(field, tile, measurement_band)
            measurement_mosaic_stem = paths.get_mosaic_name(field, tile, measurement_band)
            measurement_mosaic_path = (
                measurement_mosaic_dir / f"{measurement_mosaic_stem}.fits"
            )
        else:
            self.measurement_mosaic_path = None
        # workout where the catalogs should live.
        catalog_dir = paths.get_catalog_dir(field, tile, detection_band)
        catalog_dir.mkdir(exist_ok=True, parents=True)
        catalog_stem = paths.get_catalog_stem(
            field, tile, detection_band, measurement_band=measurement_band
        )
        catalog_path = self.catalog_dir / f"{self.catalog_stem}.cat"
        return cls(
            detection_mosaic_path, 
            measurement_mosaic_path=measurement_mosaic_path,
            catalog_path=catalog_path,
            sextractor_config=sextractor_config,
            sextractor_config_file=sextractor_config_file
            sextractor_parameter_file=sextractor_parameter_file,
        )
    """

    def extract(self):
        config = self.build_sextractor_config()
        config.update(self.sextractor_config)
        config = format_flags(config) # this capitalises stuff too.
        print(config)
        self.sextractor = Astromatic(
            "SExtractor", 
            str(paths.temp_sextractor_path), # I think Astromatic() ignores anyway?!
            config=config, # command line flags.
            config_file=str(self.sextractor_config_file),
        )
        filenames = [str(self.detection_mosaic_path)]
        if self.measurement_mosaic_path is not None:
            filenames.append(str(self.detection_mosaic_path))
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


class CatalogMatcher:

    def __init__(self, catalog_path, output_path, ra="ra", dec="dec"):
        self.catalog_path = Path(catalog_path)
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
        catalog_path = catalog_dir / f"{catalog_stem}.cat"
        suffix = suffix or ''
        prefix = prefix or ''
        if output_path is None:
            output_stem = f"{prefix}{catalog_stem}{suffix}"
            output_path = catalog_dir / f"{output_stem}.cat"
        return cls(catalog_path, output_path, ra=ra, dec=dec)

    def match_catalog(
        self, extra_catalog, ra="ra", dec="dec", prefix=None, output_path=None, **kwargs
    ):
        prefix = prefix or extra_catalog.stem
        extra_catalog = Path(extra_catalog)
        output_path = output_path or self.output_path
        stilts = Stilts.tskymatch2_fits(
            self.output_path, extra_catalog, output=output_path,
            flags={"join": "all1", "find": "best"},
            ra1=self.ra, dec1=self.dec,
            ra2=ra, dec2=dec,
            **kwargs
        )
        stilts.run()
        fix_column_names(
            self.output_path, 
            input_columns=f"Separation", 
            output_columns=f"{prefix}_separation"
        )
        info = f"{stilts.flags['join']} match {extra_catalog.name}"
        logger.info(info)
        self.summary_info.append(info)

    def join_catalog(
        self, extra_catalog, values1, values2, engine="stilts", **kwargs
    ):
        if engine=="stilts":
            stilts = Stilts.tskymatch2_fits(
                self.output_catalog, extra_catalog, values1=values1, values2=values2,
                flags={"join": "all1", "find": "best"}, **kwargs
            )
            stilts.run()
        elif engine=="astropy":
            raise NotImplementedError("DIY for now.")

    def add_map_value(self, mosaic_path, column_name, ra=None, dec=None, xpix=None, ypix=None, hdu=0):
        catalog = Table.read(self.output_path)
        with fits.open(mosaic_path) as mosaic:
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
            pixels = mosaic_wcs.wcs_world2pix(image_positions)
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
        catalog.writeto(self.output_path, format="fits", overwrite=True)
        info = f"added map data {column_name} from {mosaic_path.stem}"
        logger.info(info)
        self.summary_info.append(info)

    def print_summary(self):
        summary = "Summary:"
        for info in self.summary_info:
            summary += f"\n  -{info}"
        print(summary)


class CatalogPairMatcher(CatalogMatcher):
    """
    Class for matching catalogs. Calls stilts.
    """

    def __init__(
        self, catalog1_path, catalog2_path, output_path,
        ra1="ra", dec1="dec", prefix1=None, suffix1=None,
        ra2="ra", dec2="dec", prefix2=None, suffix2=None, 
    ):
        self.catalog1_path = Path(catalog1_path)
        self.catalog2_path = Path(catalog2_path)
        self.output_path = Path(output_path)
        self.ra1 = ra1
        self.dec1 = dec1
        self.ra2 = ra2
        self.dec2 = dec2
        # Somewhere to keep the best values when we need them.
        self.ra = None
        self.dec = None
        self.summary_info = []

    @classmethod
    def from_dxs_spec(
        cls, field, tile, output_path, prefix=None
    ):
        catalog1_dir = paths.get_catalog_dir(field, tile, "J")
        catalog1_stem = paths.get_catalog_stem(field, tile, "J", prefix=prefix)
        catalog1_path = catalog1_dir / f"{catalog1_stem}.cat"
        catalog2_dir = paths.get_catalog_dir(field, tile, "K")
        catalog2_stem = paths.get_catalog_stem(field, tile, "K", prefix=prefix)
        catalog2_path = catalog2_dir / f"{catalog2_stem}.cat"
        return cls(catalog1_path, catalog2_path, output_path, prefix)

    def best_pair_match(self, **kwargs):
        stilts = Stilts.tskymatch2_fits(
            self.catalog1_path, self.catalog2_path, self.output,
            ra1=self.ra1, dec1=self.dec1, ra2=self.ra2, dec2=self.dec2,
            flags={"join": "1or2", "find": "best"},
            **kwargs
        )
        stilts.run()
        fix_column_names(
            self.output, 
            input_columns="Separation", 
            output_columns=f"{self.prefix1}{self.prefix2}_separation"
        )
        info = f"best match {self.catalog_path1.name} and {self.catalog2_path.name}"
        logger.info(info)
        self.summary_info.append(info)
        
    def select_best_coords(
        self, ra1, dec1, snr1, ra2, dec2, snr2, best_ra="ra", best_dec="dec"
    ):
        catalog = Table.read(self.output_path)
        catalog.add_column(-99.0, "ra")
        catalog.add_column(-99.0, "dec")
        
        keep_coord1 = (catalog[snr1] >= catalog[snr2]) | (np.isnan(catalog[snr2]))
        keep_coord2 = (catalog[snr2] >  catalog[snr1]) | (np.isnan(catalog[snr1]))
        assert np.sum(keep_coord1) + np.sum(keep_coord2) == len(catalog)
        # Select the right coordinates
        catalog[best_ra][ keep_coord1 ] = catalog[ra1][ keep_coord1 ]
        catalog[best_dec][ keep_coord1 ] = catalog[dec1][ keep_coord1 ]
        catalog[best_ra][ keep_coord2 ] = catalog[ra2][ keep_coord2 ]
        catalog[best_dec][ keep_coord2 ] = catalog[dec2][ keep_coord2 ]
        assert len(catalog[ catalog["ra"] < -90.0 ]) == 0
        assert len(catalog[ catalog["dec"] < -90.0 ]) == 0
        catalog.write(self.output_path, format="fits", overwrite=True)
        self.ra = best_ra
        self.dec = best_dec

def combine_catalogs(
    self, catalog_list, output_path, id_col, ra_col, dec_col, snr_col, error=1.0
):
    catalog_list = create_file_backups(catalog_list, paths.temp_sextractor_path)
    catalog_path1 = catalog_list[0]
    output_path = Path(output_path)
    temp_overlap_path = paths.temp_sextractor_path / "{output_path.stem}_overlap.cat"
    temp_output_path = paths.temp_sextractor_path / "{output_path.stem}_combined.cat"
    for ii, catalog_path in enumerate(catalog_list):
        id_modifier = int("1{ii+1:02d}")*1_000_000
        _modify_id_value(catalog_path, id_modifier)
    for catalog_path2 in catalog_list:
        if catalog_path == result_catalog_path:
            continue         
        stilts = Stilts.tskymatch2_fits(
            output_catalog_path, catalog_path, output=temp_overlap_path,
            ra=ra_col, dec=dec_col, error=error, join="1and2", find="best"
        )
        stilts.run() # Find objects which appear in both (and only both) catalogs.

        catalog1 = Table.read(catalog_path1)
        overlap = Table.read(temp_overlap_path)
        catalog2 = Table.read(catalog_path2)
        catalog1_unique_mask = np.isin(
            catalog1[id_col], overlap[id_col+"_1"], invert=True
        )
        catalog2_unique_mask = np.isin(
            catalog2[id_col], overlap[id_col+"_2"], invert=True
        )
        catalog1 = catalog1[ catalog1_unique_mask ]
        catalog2 = catalog2[ catalog2_unique_mask ]
        columns = catalog1.columns
        overlap1 = overlap[ overlap[snr_col+"_1"] > overlap[snr_col+"_2"] ][columns]
        overlap2 = overlap[ overlap[snr_col+"_2"] > overlap[snr_col+"_1"] ][columns]
        assert len(overlap1) + len(overlap2) == len(overlap)
        combined_catalog = vstack(
            [catalog1, overlap1, overlap2, catalog2], join_type="exact"
        )
        combined_catalog.write(temp_output_path)
        catalog1_path = temp_output_path
    shutil.copy2(temp_output_path, output_path)

def _modify_id_value(catalog_path, id_modifier, id_col="id",):
    catalog = Table.read(catalog_path)
    catalog[id_col] = id_modifier + catalog[id_col]
    catalog.write(catalog_path, format="fits", overwrite=True)


if __name__ == "__main__":
    extractor = CatalogExtractor("EN", 4, "J")
    extractor.extract()

    star_table_path = (
        paths.input_data_path / "external_catalogs/tmass/tmass_ElaisN1_stars.csv"
    )
    star_catalog = Table.read(star_table_path, format="ascii")
    star_catalog = star_catalog[ star_catalog["k_m"] < 12.0 ]

    processor = CrosstalkProcessor.from_dxs_spec("EN", 4, "K", star_catalog)
    crosstalk_catalog_path = paths.temp_data_path / "EN04K_crosstalks.cat"
    #processor.collate_crosstalks(save_path=crosstalk_catalog_path)
    processor.match_crosstalks_to_catalog(
        extractor.catalog_path, ra="X_WORLD", dec="Y_WORLD", error=1.0,
        crosstalk_catalog_path=crosstalk_catalog_path
    )
    







