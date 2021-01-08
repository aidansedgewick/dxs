import json
import shutil
import yaml
from pathlib import Path

import astropy.io.fits as fits
from astropy.table import Table, Column
from astromatic_wrapper.api import Astromatic

from dxs.crosstalk_processor import CrosstalkProcessor
from dxs.mosaic_builder import get_mosaic_dir, get_mosaic_stem
from dxs.pystilts import Stilts
from dxs.utils.misc import check_modules, format_flags, create_file_backups

from dxs import paths

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
        detection_mosaic_dir = get_mosaic_dir(field, tile, detection_band)
        detection_mosaic_stem = get_mosaic_stem(field, tile, detection_band, prefix)
        self.detection_mosaic_path = detection_mosaic_dir / f"{detection_mosaic_stem}.fits"
        if measurement_band is not None:
            measurement_mosaic_dir = get_mosaic_dir(field, tile, measurement_band)
            measurement_mosaic_stem = get_mosaic_name(field, tile, measurement_band)
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
    """
    def __init__(
        self, detection_mosaic_path, measurement_mosaic_path=None, prefix=None,
        sextractor_config=None, sextractor_config_file=None, sextractor_parameter_file=None,
    ):
        self.detection_mosaic_path = detection_mosaic_path
        self.measurment_mosaic_path = measurement_mosaic_path


    @classmethod
    def from_dxs_spec(
        cls, field, tile, detection_band, measurement_band=None, prefix=None,
        sextractor_config=None, sextractor_config_file=None, sextractor_parameter_file=None,
    ):
    """

    def extract(self):
        config = self.build_sextractor_config()
        config.update(self.sextractor_config)
        config = format_flags(config) # this capitalises stuff too.
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

def get_catalog_dir(field, tile, detection_band):
    return paths.catalogs_path / get_catalog_stem(field, tile, detection_band)

def get_catalog_stem(field, tile, detection_band, measurement_band=None, prefix=None):
    prefix = prefix or ''
    measurement_band = measurement_band or ''
    return f"{prefix}{field}{tile:02d}{detection_band}{measurement_band}"

class CatalogMatcher:
    """
    Class for matching catalogs. Calls stilts.
    """

    def __init__(self,):
        pass

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
    crosstalk_cat_path = paths.temp_data_path / "EN4K_crosstalks.fits"
    processor.collate_crosstalks(save_path=crosstalk_path)
    processor.match_crosstalks_to_catalog(
        extractor.catalog_path, ra="X_WORLD", dec="Y_WORLD"
    )
    







