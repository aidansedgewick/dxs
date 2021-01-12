import json

import pandas

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

if __name__ == "__main__":

    check_modules(["swarp", "sex", "stilts"])

    star_catalog_path = (
        paths.input_data_path / "external_catalogs/tmass/tmass_ElaisN1_stars.csv"
    )
    star_catalog = Table.read(star_catalog_path, format="ascii")
    
    ## Extract catalog from J-image.
    J_extractor = CatalogExtractor.from_dxs_spec("EN", 4, "J")
    #J_extractor.extract()
    #fix_sextractor_column_names(J_extractor.catalog_path, band="J")
    ## Match its crosstalks
    J_xproc = CrosstalkProcessor.from_dxs_spec("EN", 4, "J", star_catalog=star_catalog.copy()) # star_cat is actually an arg, but for spelled out for clarity.
    #assert "j_m" in J_xproc.star_catalog.colnames
    #J_xproc.collate_crosstalks(mag_column="j_m", mag_limit=13.0)
    #J_xproc.match_crosstalks_to_catalog(J_extractor.catalog_path, ra="J_ra", dec="J_dec")
    ## Now do K-forced photometry from J image.
    JKfp_extractor = CatalogExtractor.from_dxs_spec("EN", 4, "J", measurement_band="K")
    #JKfp_extractor.extract()
    #fix_sextractor_column_names(JKfp_extractor.catalog_path, band="K", suffix="_Jfp")
    ## stick them together.
    J_matcher = CatalogMatcher.from_dxs_spec("EN", 4, "J", ra="J_ra", dec="J_dec")
    J_output = paths.get_catalog_dir("EN", 4, "K") / "EN04JKfp.cat"
    #J_matcher.match_catalog(
    #    JKfp_extractor.catalog_path, 
    #    ra="K_ra_Jfp", dec="K_dec_Jfp", error=1.0,
    #    output_path=J_output
    #)
    
    ## Extract catalog from K-image
    K_extractor = CatalogExtractor.from_dxs_spec("EN", 4, "K")
    #K_extractor.extract()
    #fix_sextractor_column_names(K_extractor.catalog_path, band="K")
    ## Match crosstalks
    K_xproc = CrosstalkProcessor.from_dxs_spec("EN", 4, "K", star_catalog=star_catalog.copy())
    assert "k_m" in K_xproc.star_catalog.colnames
    #K_xproc.collate_crosstalks(mag_column="k_m", mag_limit=13.0)
    #K_xproc.match_crosstalks_to_catalog(K_extractor.catalog_path, ra="K_ra", dec="K_dec")
    ## K-forced photometry
    KJfp_extractor = CatalogExtractor.from_dxs_spec("EN", 4, "K", measurement_band="J")
    #KJfp_extractor.extract()
    #fix_sextractor_column_names(KJfp_extractor.catalog_path, band="J", suffix="_Kfp")
    # stick them together
    K_matcher = CatalogMatcher.from_dxs_spec("EN", 4, "K", ra="K_ra", dec="K_dec")
    K_output = paths.get_catalog_dir("EN", 4, "K") / "EN04KJfp.cat"
    #K_matcher.match_catalog(
    #    KJfp_extractor.catalog_path, 
    #    ra="J_ra_Kfp", dec="J_dec_Kfp", error=1.0,
    #    output_path=K_output
    #)
    
    # pair match both.
    pair_output_path = paths.get_catalog_dir("EN", 4, "") / "EN04.cat"
    #pair_matcher = CatalogPairMatcher(
    #    J_output, K_output, output_path=pair_output_path, 
    #    ra1="J_ra", dec1="J_dec",
    #    ra2="K_ra", dec2="K_dec", 
    #)
    #pair_matcher.best_pair_match(error=2.0)
    #fix_column_names(pair_output, column_lookup={"Separation": "JK_separation"})
        
    qp = QuickPlotter.from_fits(pair_output_path)
    print(qp.catalog.colnames)
    qp.create_selection("eros", "None_j_mag_auto - None_k_mag_auto > 4.5")
    qp.create_plot("n_plot")
    qp.n_plot.plot_number_density("None_k_mag_auto", selection=qp.eros)
    qp.n_plot.plot_number_density("None_k_mag_auto", selection=qp.catalog)
    qp.n_plot.axes.semilogy()
    plt.show()








