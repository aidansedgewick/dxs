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
    QuickPlotter
)
from dxs.utils.misc import check_modules, print_header
from dxs.utils.table import fix_column_names, explode_column, remove_objects_in_bad_coverage
from dxs.utils.image import calc_survey_area, make_normalised_weight_map
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

default_lookup_path = paths.config_path / "sextractor/column_name_lookup.yaml"

snr_fluxes = ["AUTO", "ISO", "APER"]
snr_flux_formats = {
    "flux_format": "FLUX_{flux}", 
    "flux_err_format": "FLUXERR_{flux}",
    "snr_format": "SNR_{flux}",
}
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
    parser.add_argument("--match_pair", action="store_true", default=False)
    parser.add_argument("--match_extras", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--n_cpus", action="store", default=None, type=int)

    args = parser.parse_args()

    field = args.field
    tile = args.tile

    if args.match_all:
        args.match_crosstalks = True
        args.match_fp = True
        args.match_pair = True
        args.match_extras = True        

    if args.full:
        args.extract = True
        args.collate_crosstalks = True
        args.match_crosstalks = True
        args.match_fp = True
        args.match_pair = True
        args.match_extras = True

    check_modules("swarp", "sex", "stilts")

    field_name = survey_config["code_to_field"][field]
    star_catalog_path = (
        paths.input_data_path / f"external/tmass/tmass_{field_name}_stars.csv"
    )
    star_catalog = Table.read(star_catalog_path, format="ascii")
    print(f"Match crosstalk stars in {star_catalog_path.name}")

    t1 = time.time()

    ##============================Extract catalog from J-image.
    
    Jcat_stem = paths.get_catalog_stem(field, tile, "Ji", prefix=args.prefix)
    J_ex = CatalogExtractor.from_dxs_spec(
        field, tile, "J", catalog_stem=Jcat_stem, prefix=args.prefix
    )
    J_coverage_map_path = paths.get_mosaic_path(field, tile, "J").with_suffix(".cov.good_cov.fits")
    J_weight_map_path = paths.get_mosaic_path(field, tile, "J").with_suffix(".weight.fits")
    #J_norm_weight_map = J_ex.detection_mosaic_path.with_suffix(".norm_weight.fits")
    if args.extract:
        print_header("J photom")
        J_ex.extract()
        J_ex.add_snr(snr_fluxes, **snr_flux_formats)     
        fix_sextractor_column_names(J_ex.catalog_path, band="J")
        J_ex.add_map_value(J_coverage_map_path, "J_coverage", ra="J_ra", dec="J_dec")
        #make_normalised_weight_map(
        #    J_weight_map_path, J_coverage_map_path, J_norm_weight_map
        #)
        #J_ex.add_map_value(J_norm_weight_map, "J_norm_weight", ra="J_ra", dec="J_dec")
        J_ex.add_column({"J_tile": tile})
    ## Match its crosstalks
    J_xproc = CrosstalkProcessor.from_dxs_spec(
        field, tile, "J", star_catalog.copy(), n_cpus=args.n_cpus
    )
    assert "j_m" in J_xproc.star_catalog.colnames
    if args.collate_crosstalks:
        J_xproc.collate_crosstalks(mag_column="j_m", mag_limit=13.0)
    Jout_stem = paths.get_catalog_stem(field, args.tile, "J", prefix=args.prefix)
    J_output_path = J_ex.catalog_path.parent / f"{Jout_stem}.fits"
    if args.match_crosstalks:
        print_header("J crosstalks")
        J_xproc.match_crosstalks_to_catalog(
            J_ex.catalog_path, ra="J_ra", dec="J_dec", 
            output_path=J_output_path, band="J"
        )
        fix_crosstalk_column_names(J_output_path, band="J")
        explode_column(J_output_path, "J_flux_radius", suffixes=[20, 50, 90])
        for col in ["J_flux_aper", "J_fluxerr_aper", "J_mag_aper", "J_magerr_aper"]:
            explode_column(J_output_path, col, suffixes=[10, 18, 20, 30], remove=True)
        remove_objects_in_bad_coverage(J_output_path, coverage_column="J_coverage")

    ## Now do K-forced photometry from J image.
    JKfp_ex = CatalogExtractor.from_dxs_spec(
        field, tile, "J", measurement_band="K", prefix=args.prefix
    )
    if args.extract:
        print_header("J ap fp on K")
        JKfp_ex.extract()
        fix_sextractor_column_names(JKfp_ex.catalog_path, band="K", suffix="_Jfp")
    # stick them together.
    Jfp_output_dir = paths.get_catalog_dir(field, tile, "J")
    Jfp_combined_stem = paths.get_catalog_stem(field, tile, "JK", prefix=args.prefix)
    Jfp_output_path =  Jfp_output_dir / f"{Jfp_combined_stem}.fits"
    Jfp_matcher = CatalogMatcher(
        J_output_path, output_path=Jfp_output_path, ra="J_ra", dec="J_dec"
    )
    if args.match_fp:
        print_header("match J and K forced")
        Jfp_matcher.match_catalog(
            JKfp_ex.catalog_path, ra="K_ra_Jfp", dec="K_dec_Jfp", error=1.0
        )
        fix_column_names(Jfp_output_path, column_lookup={"Separation": "Jfp_separation"})





    ##=========================Extract catalog from K-image.
    Kcat_stem = paths.get_catalog_stem(field, args.tile, "Ki", prefix=args.prefix) 
    K_ex = CatalogExtractor.from_dxs_spec(
        field, tile, "K", catalog_stem=Kcat_stem, prefix=args.prefix
    )
    K_coverage_map_path = paths.get_mosaic_path(field, args.tile, "K").with_suffix(".cov.good_cov.fits")
    K_weight_map_path = paths.get_mosaic_path(field, args.tile, "K").with_suffix(".weight.fits")
    #K_norm_weight_map = K_ex.detection_mosaic_path.with_suffix(".norm_weight.fits")
    if args.extract:
        print_header("K photom")
        K_ex.extract()
        K_ex.add_snr(snr_fluxes, **snr_flux_formats)      
        fix_sextractor_column_names(K_ex.catalog_path, band="K")
        K_ex.add_map_value(K_coverage_map_path, "K_coverage", ra="K_ra", dec="K_dec")
        #make_normalised_weight_map(
        #    K_weight_map_path, K_coverage_map_path, K_norm_weight_map
        #)
        #K_ex.add_map_value(K_norm_weight_map, "K_norm_weight", ra="K_ra", dec="K_dec")
        K_ex.add_column({"K_tile": args.tile})
    ## Match its crosstalks
    K_xproc = CrosstalkProcessor.from_dxs_spec(
        field, tile, "K", star_catalog.copy(), n_cpus=args.n_cpus
    )
    assert "k_m" in K_xproc.star_catalog.colnames
    if args.collate_crosstalks:
        K_xproc.collate_crosstalks(mag_column="k_m", mag_limit=13.0)
    Kout_stem = paths.get_catalog_stem(field, args.tile, "K", prefix=args.prefix)
    K_output_path = K_ex.catalog_path.parent / f"{Kout_stem}.fits"
    if args.match_crosstalks:
        print_header("K crosstalks")
        K_xproc.match_crosstalks_to_catalog(
            K_ex.catalog_path, ra="K_ra", dec="K_dec", output_path=K_output_path, 
        )
        fix_crosstalk_column_names(K_output_path, band="K")
        explode_column(K_output_path, "K_flux_radius", suffixes=[20, 50, 90])
        for col in ["K_flux_aper", "K_fluxerr_aper", "K_mag_aper", "K_magerr_aper"]:
            explode_column(K_output_path, col, suffixes=[10, 18, 20, 30], remove=True)
        remove_objects_in_bad_coverage(K_output_path, coverage_column="K_coverage")
            
    ## Now do K-forced photometry from J image.
    KJfp_ex = CatalogExtractor.from_dxs_spec(
        field, tile, "K", measurement_band="J", prefix=args.prefix
    )
    if args.extract:
        print_header("K ap fp on J")
        KJfp_ex.extract()
        fix_sextractor_column_names(KJfp_ex.catalog_path, band="J", suffix="_Kfp")
    ## stick them together.
    Kfp_output_dir = paths.get_catalog_dir(field, tile, "K")
    Kfp_combined_stem = paths.get_catalog_stem(field, tile, "KJ", prefix=args.prefix)
    Kfp_output_path =  Kfp_output_dir / f"{Kfp_combined_stem}.fits"
    Kfp_matcher = CatalogMatcher(
        K_output_path, output_path=Kfp_output_path, ra="K_ra", dec="K_dec"
    )
    if args.match_fp:
        Kfp_matcher.match_catalog(
            KJfp_ex.catalog_path, ra="J_ra_Kfp", dec="J_dec_Kfp", error=1.0
        )
        fix_column_names(Kfp_output_path, column_lookup={"Separation": "Kfp_separation"})

    ##======================== extract H

    ## Extract catalog from H-image.
    H_ex = CatalogExtractor.from_dxs_spec(field, tile, "H", prefix=args.prefix)
    if H_ex.detection_mosaic_path.exists() and args.extract:
        print_header("H-band")
        H_ex.extract()
        fix_sextractor_column_names(H_ex.catalog_path, band="H")
        H_cov_map_path = K_ex.detection_mosaic_path.with_suffix(".cov.good_cov.fits")
        H_ex.add_map_value(H_cov_map_path, "H_coverage", ra="H_ra", dec="H_dec")
        remove_objects_in_bad_coverage(H_ex.catalog_path, coverage_column="H_coverage")
        H_ex.add_column({"H_tile": args.tile})


    ##===========================Match pair of outputs.

    pair_output_stem = paths.get_catalog_stem(field, tile, "", prefix=args.prefix)
    pair_output_dir = paths.get_catalog_dir(field, tile, "")
    pair_output_path = pair_output_dir / f"{pair_output_stem}.fits"
    pair_matcher = CatalogPairMatcher(
        J_output_path, K_output_path, pair_output_path, 
        output_ra="ra", output_dec="dec",
        ra1="J_ra", dec1="J_dec", 
        ra2="K_ra", dec2="K_dec",
    )
    if args.match_pair:
        print_header("Match pair")
        pair_matcher.best_pair_match(error=2.0)
        pair_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
        pair_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")


    if args.match_extras:
        print_header("match extras")
        if H_ex.catalog_path.exists():
            pair_matcher.match_catalog(H_ex.catalog_path, ra="H_ra", dec="H_dec", error=2.0)
            pair_matcher.fix_column_names(column_lookup={"Separation": "H_separation"})
        
        ps_name = f"{field}_panstarrs"
        ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
        pair_matcher.match_catalog(ps_catalog_path, ra="i_ra", dec="i_dec", error=2.0)
        pair_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})
    
    ##=========================== do the same for fp catalogs.

    pairfp_output_stem = paths.get_catalog_stem(field, tile, "fp", prefix=args.prefix)
    pairfp_output_dir = paths.get_catalog_dir(field, tile, "")
    pairfp_output_path = pair_output_dir / f"{pairfp_output_stem}.fits"
    pairfp_matcher = CatalogPairMatcher(
        Jfp_output_path, Kfp_output_path, pairfp_output_path, 
        output_ra="ra", output_dec="dec",
        ra1="J_ra", dec1="J_dec", 
        ra2="K_ra", dec2="K_dec",
    )
    if args.match_pair:
        print_header("Match fp pairs")
        pairfp_matcher.best_pair_match(error=2.0)
        pairfp_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
        pairfp_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")


    if args.match_extras:
        print_header("match extras")
        if H_ex.catalog_path.exists():
            pairfp_matcher.match_catalog(H_ex.catalog_path, ra="H_ra", dec="H_dec", error=2.0)
            pairfp_matcher.fix_column_names(column_lookup={"Separation": "H_separation"})
        
        ps_name = f"{field}_panstarrs"
        ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
        pairfp_matcher.match_catalog(ps_catalog_path, ra="i_ra", dec="i_dec", error=2.0)
        pairfp_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})

    """
    # Make some quick plots.
    bins = np.arange(14, 22, 0.5)

    JK_cov_images = [
        J_ex.detection_mosaic_path.with_suffix(".cov.good_cov.fits"),
        K_ex.detection_mosaic_path.with_suffix(".cov.good_cov.fits"),
    ]
    ps_cov_image = [paths.input_data_path / f"external/panstarrs/masks/{field}_mask.fits"]


    qp = QuickPlotter.from_fits(pair_output_path)

    ra_limits = (np.min(qp.full_catalog["ra"]), np.max(qp.full_catalog["ra"]))
    dec_limits = (np.min(qp.full_catalog["dec"]), np.max(qp.full_catalog["dec"]))

    JK_survey_area = calc_survey_area(
        JK_cov_images, ra_limits=ra_limits, dec_limits=dec_limits
    )
    print(f"JK survey area = {JK_survey_area:.3f} sq. deg")
    JKps_survey_area = calc_survey_area(
        JK_cov_images + ps_cov_image, ra_limits=ra_limits, dec_limits=dec_limits
    )
    print(f"JK + opt. area = {JKps_survey_area:.3f} sq. deg")
    qp.remove_crosstalks()
    qp.remove_bad_magnitudes(["J", "K"], catalog=qp.catalog)
    print(f"Catalog length: {len(qp.catalog)}")
    qp.create_selection("gals", ("J_mag_auto - K_mag_auto > 1.0"), catalog=qp.catalog)
    qp.create_selection("JKfp_gals", ("J_mag_auto - K_mag_auto_Jfp > 1.0"), catalog=qp.catalog)
    qp.create_selection("JfpK_gals", ("J_mag_auto_Kfp - K_mag_auto > 1.0"), catalog=qp.catalog)
    qp.create_selection("drgs", ("J_mag_auto - K_mag_auto > 2.3"), catalog=qp.gals)
    qp.create_selection("eros", ("i_mag_kron - K_mag_auto > 4.5"), catalog=qp.gals)

    qp.create_plot("n_plot")
    qp.n_plot.plot_number_density(
        "K_mag_auto", selection=qp.catalog, bins=bins, label="catalog", survey_area=JK_survey_area
    )
    qp.n_plot.plot_number_density(
        "K_mag_auto", selection=qp.gals, bins=bins, label="gals", survey_area=JK_survey_area
    )
    qp.n_plot.plot_number_density(
        "K_mag_auto", selection=qp.JfpK_gals, bins=bins, label="gals J K-ap", survey_area=JK_survey_area
    )
    qp.n_plot.plot_number_density(
        "K_mag_auto_Jfp", selection=qp.JKfp_gals, bins=bins, label="gals K J-ap", survey_area=JK_survey_area
    )
    qp.n_plot.plot_number_density(
        "K_mag_auto", selection=qp.drgs, bins=bins, label="DRGs", survey_area=JK_survey_area
    )
    qp.n_plot.plot_number_density(
        "K_mag_auto", selection=qp.eros, bins=bins, label="EROs", survey_area=JKps_survey_area
    )
    qp.n_plot.axes.semilogy()
    qp.n_plot.axes.legend()

    #print(qp.catalog.colnames)
    
    qp.create_plot("jk_plot")
    qp.jk_plot.color_magnitude(
        c1="J_mag_auto", c2="K_mag_auto", mag="K_mag_auto", selection=qp.catalog,
        color="k", s=1
    )
    qp.jk_plot.fig.suptitle("stilts matched")

    qp.create_plot("jkfp_plot")
    qp.jkfp_plot.color_magnitude(
        c1="J_mag_auto", c2="K_mag_auto_Jfp", mag="K_mag_auto_Jfp", selection=qp.catalog,
        color="k", s=1
    )
    qp.jk_plot.fig.suptitle("forced J apertures")
    
    qp.create_plot("jfpk_plot")
    qp.jfpk_plot.color_magnitude(
        c1="J_mag_auto_Kfp", c2="K_mag_auto", mag="K_mag_auto", selection=qp.catalog,
        color="k", s=1
    )
    qp.jk_plot.fig.suptitle("forced K apertures")

    if "H_mag_auto" in qp.catalog.colnames:
        qp.create_plot("jh_hk")
        qp.jh_hk.color_color(
            c_x1="J_mag_auto", c_x2="H_mag_auto", c_y1="H_mag_auto", c_y2="K_mag_auto",
            selection=qp.catalog,
            s=1, color="k",
        )
    plt.show()"""
    
    #qp.save_all_plots()

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






