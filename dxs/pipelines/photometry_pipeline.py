import json
import logging
import time
import yaml
from argparse import ArgumentParser
from pathlib import Path

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
    CrosstalkProcessor,
)
from dxs.utils.misc import check_modules, print_header
from dxs.utils.table import fix_column_names, explode_columns_in_fits #, remove_objects_in_bad_coverage
from dxs.utils.image import calc_survey_area
from dxs import paths

logger = logging.getLogger("photometry_pipeline")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)


###============ get some info about column names ==============###

default_lookup_path = paths.config_path / "sextractor/column_name_lookup.yaml"

aperture_suffixes = survey_config["catalogs"].get("apertures", None)
flux_radii_suffixes = survey_config["catalogs"].get("flux_radii", None)

snr_fluxes = ["AUTO"] #, "ISO", "APER"]
flux_cols = [f"FLUX_{flux}" for flux in snr_fluxes]
fluxerr_cols = [f"FLUXERR_{flux}" for flux in snr_fluxes]
snr_cols = [f"SNR_{flux}" for flux in snr_fluxes]

crosstalk_columns = (
    [f"crosstalk_{x}" for x in ["ra", "dec", "direction", "order", "flag"]]
    + [f"parent_{x}" for x in ["id", "ra", "dec", "mag"]]
)
crosstalk_column_lookup = {
    "GroupID": "crosstalk_group_id",
    "GroupSize": "crosstalk_group_size",
    "Separation": "crosstalk_separation",
}

def fix_crosstalk_column_names(catalog_path, band=None, prefix=None, suffix=None):
    fix_column_names(
        catalog_path,
        input_columns=crosstalk_columns,
        column_lookup=crosstalk_column_lookup,
        band=band,
        prefix=prefix, 
        suffix=suffix,
    )

def _load_sextractor_column_lookup(lookup_path=default_lookup_path):
    with open(lookup_path, "r") as f:
        sextractor_lookup = yaml.load(f, Loader=yaml.FullLoader)
    return sextractor_lookup

def fix_sextractor_column_names(catalog_path, band=None, prefix=None, suffix=None):
    sextractor_lookup = _load_sextractor_column_lookup()
    fix_column_names(
        catalog_path, 
        input_columns="catalog",
        column_lookup=sextractor_lookup, 
        band=band, 
        prefix=prefix, 
        suffix=suffix
    )

def photometry_pipeline(
    field, tile, 
    prefix="", 
    extract=True, 
    collate_crosstalks=True,
    match_crosstalks=True,
    match_fp=True,
    #match_pair=True,
    #match_extras=True,
    n_cpus=None,
):
    field_name = survey_config["code_to_field"][field]
    star_catalog_path = (
        paths.input_data_path / f"external/tmass/tmass_{field_name}_stars.csv"
    )
    star_catalog = Table.read(star_catalog_path, format="ascii")
    print(f"Match crosstalk stars in {star_catalog_path.name}")

    ##============================Extract catalog from J-image.
    Jspec = (field, tile, "J")
    Jcat_stem = paths.get_catalog_stem(*Jspec, prefix=prefix, suffix="_init")
    J_ex = CatalogExtractor.from_dxs_spec(*Jspec, catalog_stem=Jcat_stem, prefix=prefix)
    J_coverage_map_path = paths.get_mosaic_path(*Jspec).with_suffix(".cov.good_cov.fits")
    J_weight_map_path = paths.get_mosaic_path(*Jspec).with_suffix(".weight.fits")

    if extract:
        print_header("J photom")
        J_ex.extract()
        J_ex.add_snr(flux_cols, fluxerr_cols, snr_cols)
        fix_sextractor_column_names(J_ex.catalog_path, band="J")
        J_ex.add_map_value(J_coverage_map_path, "J_coverage", ra="J_ra", dec="J_dec")
        logger.info(f"initial J cat at {J_ex.catalog_path}")
        J_ex.add_column({"J_tile": tile})

    ## Match its crosstalks
    J_xproc = CrosstalkProcessor.from_dxs_spec(*Jspec, star_catalog.copy())
    assert "j_m" in J_xproc.star_catalog.colnames
    if collate_crosstalks:
        J_xproc.collate_crosstalks(mag_column="j_m", mag_limit=15.0, n_cpus=n_cpus)
    Jout_stem = paths.get_catalog_stem(*Jspec, prefix=prefix)
    J_output_path = J_ex.catalog_path.parent / f"{Jout_stem}.cat.fits"
    if match_crosstalks:
        print_header("J crosstalks")
        J_xproc.match_crosstalks_to_catalog(
            J_ex.catalog_path, ra="J_ra", dec="J_dec", output_path=J_output_path
        )
        fix_crosstalk_column_names(J_output_path, band="J")
        explode_columns_in_fits(J_output_path, "J_flux_radius", suffixes=flux_radii_suffixes)
        J_aper_cols = ["J_mag_aper", "J_magerr_aper"] #"J_flux_aper", "J_fluxerr_aper", 
        explode_columns_in_fits(
            J_output_path, J_aper_cols, suffixes=aperture_suffixes, remove=True
        )
        #remove_objects_in_bad_coverage(J_output_path, coverage_column="J_coverage")
        logger.info(f"xtalk-matched J cat at {J_output_path}")

    ## Now do K-forced photometry from J image.
    JKfp_ex = CatalogExtractor.from_dxs_spec(
        field, tile, "J", measurement_band="K", prefix=prefix
    )
    if extract:
        print_header("K forced from J apers")
        JKfp_ex.extract()
        fix_sextractor_column_names(JKfp_ex.catalog_path, band="K", suffix="_Jfp")
        logger.info(f"K forced from J apers at {JKfp_ex.catalog_path}")
    # stick them together.
    Jfp_output_dir = paths.get_catalog_dir(*Jspec)
    Jfp_combined_stem = paths.get_catalog_stem(field, tile, "JK", prefix=prefix)
    Jfp_output_path =  Jfp_output_dir / f"{Jfp_combined_stem}.cat.fits"
    Jfp_matcher = CatalogMatcher(J_output_path, ra="J_ra", dec="J_dec")
    if match_fp:
        print_header("match J and K forced")
        Jfp_matcher.match_catalog(
            JKfp_ex.catalog_path, output_path=Jfp_output_path, ra="K_ra_Jfp", dec="K_dec_Jfp", error=1.0
        )
        fix_column_names(Jfp_output_path, column_lookup={"Separation": "Jfp_separation"})
        logger.info(f"joined J and K forced at {Jfp_output_path}")



    ##=========================Extract catalog from K-image.
    Kspec = (field, tile, "K")
    Kcat_stem = paths.get_catalog_stem(*Kspec, prefix=prefix, suffix="_init") 
    K_ex = CatalogExtractor.from_dxs_spec(
        *Kspec, catalog_stem=Kcat_stem, prefix=prefix
    )
    K_coverage_map_path = paths.get_mosaic_path(*Kspec).with_suffix(".cov.good_cov.fits")
    K_weight_map_path = paths.get_mosaic_path(*Kspec).with_suffix(".weight.fits")

    if extract:
        print_header("K photom")
        K_ex.extract()
        K_ex.add_snr(flux_cols, fluxerr_cols, snr_cols)
        fix_sextractor_column_names(K_ex.catalog_path, band="K")
        K_ex.add_map_value(K_coverage_map_path, "K_coverage", ra="K_ra", dec="K_dec")
        K_ex.add_column({"K_tile": tile})
        logger.info(f"initial K cat at {K_ex.catalog_path}")

    ## Match its crosstalks
    K_xproc = CrosstalkProcessor.from_dxs_spec(field, tile, "K", star_catalog.copy())
    assert "k_m" in K_xproc.star_catalog.colnames
    if collate_crosstalks:
        K_xproc.collate_crosstalks(mag_column="k_m", mag_limit=15.0, n_cpus=n_cpus)
    Kout_stem = paths.get_catalog_stem(field, tile, "K", prefix=prefix)
    K_output_path = K_ex.catalog_path.parent / f"{Kout_stem}.cat.fits"
    if match_crosstalks:
        print_header("K crosstalks")
        K_xproc.match_crosstalks_to_catalog(
            K_ex.catalog_path, ra="K_ra", dec="K_dec", output_path=K_output_path, 
        )
        fix_crosstalk_column_names(K_output_path, band="K")
        explode_columns_in_fits(K_output_path, "K_flux_radius", suffixes=flux_radii_suffixes)
        K_aper_cols = ["K_mag_aper", "K_magerr_aper"] #"K_flux_aper", "K_fluxerr_aper", 
        explode_columns_in_fits(
            K_output_path, K_aper_cols, suffixes=aperture_suffixes, remove=True
        )
        #remove_objects_in_bad_coverage(K_output_path, coverage_column="K_coverage")
        logger.info(f"xtalk-matched K cat at {K_output_path}")

    ## Now do K-forced photometry from J image.
    KJfp_ex = CatalogExtractor.from_dxs_spec(*Kspec, measurement_band="J", prefix=prefix)
    if extract:
        print_header("J forced from K apers")
        KJfp_ex.extract()
        fix_sextractor_column_names(KJfp_ex.catalog_path, band="J", suffix="_Kfp")
        logger.info(f"J forced from K apers at {KJfp_ex.catalog_path}")
    ## stick them together.
    Kfp_output_dir = paths.get_catalog_dir(*Kspec)
    Kfp_combined_stem = paths.get_catalog_stem(field, tile, "KJ", prefix=prefix)
    Kfp_output_path =  Kfp_output_dir / f"{Kfp_combined_stem}.cat.fits"
    Kfp_matcher = CatalogMatcher(K_output_path, ra="K_ra", dec="K_dec")
    if match_fp:
        Kfp_matcher.match_catalog(
            KJfp_ex.catalog_path, output_path=Kfp_output_path, 
            ra="J_ra_Kfp", dec="J_dec_Kfp", error=1.0
        )
        fix_column_names(Kfp_output_path, column_lookup={"Separation": "Kfp_separation"})
        logger.info(f"joined J and K forced at {Kfp_output_path}")

    ##======================== extract H

    ## Extract catalog from H-image.
    H_ex = CatalogExtractor.from_dxs_spec(field, tile, "H", prefix=prefix)
    if H_ex.detection_mosaic_path.exists() and extract:
        print_header("H-band")
        H_ex.extract()
        H_ex.add_snr(flux_cols, fluxerr_cols, snr_cols)
        fix_sextractor_column_names(H_ex.catalog_path, band="H")
        H_cov_map_path = K_ex.detection_mosaic_path.with_suffix(".cov.good_cov.fits")
        H_ex.add_map_value(H_cov_map_path, "H_coverage", ra="H_ra", dec="H_dec")
        remove_objects_in_bad_coverage(H_ex.catalog_path, coverage_column="H_coverage")
        H_ex.add_column({"H_tile": tile})
        H_aper_cols = ["H_mag_aper", "H_magerr_aper"] #"K_flux_aper", "K_fluxerr_aper", 
        explode_columns_in_fits(
            H_output_path, H_aper_cols, suffixes=aperture_suffixes, remove=True
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("--prefix", action="store", default=None)
    parser.add_argument("--extract", action="store_true", default=False)
    parser.add_argument("--collate_crosstalks", action="store_true", default=False)
    parser.add_argument("--match_all", action="store_true", default=False)
    parser.add_argument("--match_crosstalks", action="store_true", default=False)
    parser.add_argument("--match_fp", action="store_true", default=False)
    #parser.add_argument("--match_pair", action="store_true", default=False)
    #parser.add_argument("--match_extras", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--n_cpus", action="store", default=None, type=int)

    args = parser.parse_args()

    field = args.field
    tile = args.tile

    if args.match_all:
        args.match_crosstalks = True
        args.match_fp = True
        #args.match_pair = True
        #args.match_extras = True        

    if args.full:
        args.extract = True
        args.collate_crosstalks = True
        args.match_crosstalks = True
        args.match_fp = True
        #args.match_pair = True
        #args.match_extras = True

    check_modules("swarp", "sex", "stilts")
    
    photometry_pipeline(
        field, tile, prefix=args.prefix, 
        extract=args.extract, 
        collate_crosstalks=args.collate_crosstalks,
        match_crosstalks=args.match_crosstalks,
        match_fp=args.match_fp,
        #match_pair=args.match_pair,
        #match_extras=args.match_extras,
        n_cpus=args.n_cpus
    )

    t1 = time.time()
    print("done!")
    t2 = time.time()

    if t2 - t1 < 5.:
        path = Path(__file__).absolute()
        try:
            print_path = path.relative_to(Path.cwd())
        except:
            print_path = path
        cmd = f"python3 {print_path} {args.field} {args.tile} --full"
        print(f"try:\n    {cmd}")

