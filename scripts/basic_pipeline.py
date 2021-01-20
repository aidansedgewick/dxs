import json
import yaml
from argparse import ArgumentParser

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
    CrosstalkProcessor
)
from dxs.utils.misc import check_modules
from dxs.utils.table import fix_column_names, fix_sextractor_column_names
from dxs.quick_plotter import QuickPlotter, Plot
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tile", type=int)
    parser.add_argument("--extract", action="store_true", default=False)
    parser.add_argument("--collate_crosstalks", action="store_true", default=False)
    parser.add_argument("--match_crosstalks", action="store_true", default=False)
    parser.add_argument("--match_fp", action="store_true", default=False)
    parser.add_argument("--match_pair", action="store_true", default=False)
    parser.add_argument("--match_extras", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)

    args = parser.parse_args()

    if args.full:
        args.extract = True
        args.collate_crosstalks = True
        args.match_crosstalks = True
        args.match_fp = True
        args.match_pair = True
        args.match_extras = True

    check_modules(["swarp", "sex", "stilts"])

    field_name = survey_config["code_to_field"][args.field]
    star_catalog_path = (
        paths.input_data_path / f"external/tmass/tmass_{field_name}_stars.csv"
    )
    star_catalog = Table.read(star_catalog_path, format="ascii")

    crosstalk_columns = (
        [f"crosstalk_{x}" for x in ["ra", "dec", "direction", "order", "flag"]]
        + [f"parent_{x}" for x in ["id", "ra", "dec", "mag"]]
    )

    print(f"Match crosstalk stars in {star_catalog_path.name}")


    ##============================Extract catalog from J-image.
    J_ex = CatalogExtractor.from_dxs_spec(args.field, args.tile, "J")
    if args.extract:
        J_ex.extract()
        fix_sextractor_column_names(J_ex.catalog_path, band="J")

    ## Match its crosstalks
    J_xproc = CrosstalkProcessor.from_dxs_spec(
        args.field, args.tile, "J", star_catalog=star_catalog.copy()
    ) # star_cat is actually an arg, but for spelled out for clarity.
    assert "j_m" in J_xproc.star_catalog.colnames
    if args.collate_crosstalks:
        J_xproc.collate_crosstalks(mag_column="j_m", mag_limit=13.0)
    J_with_xtalks_path = J_ex.catalog_path.parent / f"{J_ex.catalog_path.stem}_x.cat"
    if args.match_crosstalks:
        J_xproc.match_crosstalks_to_catalog(
            J_ex.catalog_path, ra="J_ra", dec="J_dec", 
            output_path=J_with_xtalks_path, band="J"
        )
        fix_column_names(J_with_xtalks_path, input_columns=crosstalk_columns, band="J")
    
    ## Now do K-forced photometry from J image.
    JKfp_ex = CatalogExtractor.from_dxs_spec(args.field, args.tile, "J", measurement_band="K")
    if args.extract:
        JKfp_ex.extract()
        fix_sextractor_column_names(JKfp_ex.catalog_path, band="K", suffix="_Jfp")

    # stick them together.
    J_output_path = paths.get_catalog_dir(args.field, args.tile, "K") / "EN04JKfp.cat"
    J_matcher = CatalogMatcher(
        J_with_xtalks_path, output_path=J_output_path, ra="J_ra", dec="J_dec"
    )
    if args.match_fp:
        J_matcher.match_catalog(
            JKfp_ex.catalog_path, ra="K_ra_Jfp", dec="K_dec_Jfp", error=1.0
        )
        fix_column_names(J_output_path, column_lookup={"Separation": "Jfp_separation"})
    #J_output = Table.read(J_output_path)
    #print(J_output.colnames)




    ##=========================Extract catalog from K-image.
    K_ex = CatalogExtractor.from_dxs_spec(args.field, args.tile, "K")
    if args.extract:
        K_ex.extract()
        fix_sextractor_column_names(K_ex.catalog_path, band="K")

    ## Match its crosstalks
    K_xproc = CrosstalkProcessor.from_dxs_spec(
        args.field, args.tile, "K", star_catalog=star_catalog.copy()
    ) # star_cat is actually an arg, but for spelled out for clarity.
    assert "k_m" in K_xproc.star_catalog.colnames
    if args.collate_crosstalks:
        K_xproc.collate_crosstalks(mag_column="k_m", mag_limit=13.0)
    K_with_xtalks_path = K_ex.catalog_path.parent / f"{K_ex.catalog_path.stem}_x.cat"
    if args.match_crosstalks:
        K_xproc.match_crosstalks_to_catalog(
            K_ex.catalog_path, ra="K_ra", dec="K_dec", output_path=K_with_xtalks_path, band="K"
        )
        fix_column_names(K_with_xtalks_path, input_columns=crosstalk_columns, band="K")

    ## Now do K-forced photometry from J image.
    KJfp_ex = CatalogExtractor.from_dxs_spec(args.field, args.tile, "K", measurement_band="J")
    if args.extract:
        KJfp_ex.extract()
        fix_sextractor_column_names(KJfp_ex.catalog_path, band="J", suffix="_Kfp")

    ## stick them together.
    K_output_path = paths.get_catalog_dir(args.field, args.tile, "K") / "EN04KJfp.cat"
    K_matcher = CatalogMatcher(
        K_with_xtalks_path, output_path=K_output_path, ra="K_ra", dec="K_dec"
    )
    if args.match_fp:
        K_matcher.match_catalog(
            KJfp_ex.catalog_path, ra="J_ra_Kfp", dec="J_dec_Kfp", error=1.0
        )
        fix_column_names(K_output_path, column_lookup={"Separation": "Kfp_separation"})
    #K_output = Table.read(K_output_path)
    #print(K_output.colnames)




    ##===========================Match pair of outputs.
    pair_output_stem = paths.get_catalog_stem(args.field, args.tile, "")
    pair_output_dir = paths.get_catalog_dir(args.field, args.tile, "")
    pair_output_path = pair_output_dir / f"{pair_output_stem}.fits"
    pair_matcher = CatalogPairMatcher(
        J_output_path, K_output_path, pair_output_path, best_ra="ra", best_dec="dec",
        ra1="J_ra", dec1="J_dec", ra2="K_ra", dec2="K_dec",
    )
    if args.match_pair:
        pair_matcher.best_pair_match(error=2.0)
        pair_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
        pair_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")

    ## Extract catalog from H-image.
    H_ex = CatalogExtractor.from_dxs_spec(args.field, args.tile, "H")
    if H_ex.detection_mosaic_path.exists() and args.extract:
        H_ex.extract()
        fix_sextractor_column_names(H_ex.catalog_path, band="H")
    if args.match_extras:
        if H_ex.catalog_path.exists():
            pair_matcher.match_catalog(H_ex.catalog_path, ra="H_ra", dec="H_dec", error=2.0)
            pair_matcher.fix_column_names(column_lookup={"Separation": "H_separation"})
        
        ps_name = f"{args.field}_panstarrs"
        ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
        pair_matcher.match_catalog(ps_catalog_path, ra="i_ra", dec="i_dec", error=2.0)
        pair_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})

    # Make some quick plots.
    bins = np.arange(14, 22, 0.5)

    qp = QuickPlotter.from_fits(pair_output_path)
    qp.remove_crosstalks()
    qp.create_selection("drgs", ("J_mag_auto - K_mag_auto > 2.3"))
    qp.create_selection("eros", ("i_mag_kron - K_mag_auto > 4.5"))
    qp.create_plot("n_plot")
    qp.n_plot.plot_number_density("K_mag_auto", selection=qp.drgs, bins=bins)
    qp.n_plot.plot_number_density("K_mag_auto", selection=qp.catalog, bins=bins)
    qp.n_plot.plot_number_density("K_mag_auto", selection=qp.eros, bins=bins)
    qp.n_plot.axes.semilogy()

    #print(qp.catalog.colnames)

    qp.create_plot("jk_plot")
    qp.jk_plot.color_magnitude(
        c1="J_mag_auto", c2="K_mag_auto", mag="K_mag_auto", selection=qp.catalog,
        color="k", s=1
    )
    if "H_mag_auto" in qp.catalog.colnames:
        qp.create_plot("jh_hk")
        qp.jh_hk.color_color(
            c_x1="J_mag_auto", c_x2="H_mag_auto", c_y1="H_mag_auto", c_y2="K_mag_auto",
            selection=qp.catalog,
            s=1, color="k",
        )

    #plt.show()








