import logging
import pickle
import yaml
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

from dustmaps import sfd
from easyquery import Query
from treecorr import NNCorrelation, Catalog

from dxs.utils.image import uniform_sphere, objects_in_coverage, calc_survey_area
from dxs import QuickPlotter
from dxs.utils.misc import calc_range, calc_mids, print_header
from dxs.utils.phot import ab_to_vega, vega_to_ab, apply_extinction
from dxs.utils.region import in_only_one_tile
from dxs import paths

logger = logging.getLogger("correlator")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

J_offset = survey_config["ab_vega_offset"]["J"]
K_offset = survey_config["ab_vega_offset"]["K"]

pldat_dir = paths.input_data_path / "plotting"

kim11_number_counts_path = pldat_dir / "kim11_number_counts.csv"
kim14_number_counts_path = pldat_dir / "kim14_number_counts.csv"
arcilaosejo19_number_counts_path = pldat_dir / "arcilaosejo19_number_counts.csv"
mccracken10_number_counts_path = pldat_dir / "mccracken10_number_counts.csv"
kajisawa06_number_counts_path = pldat_dir / "kajisawa06_number_counts.csv"

mag_min, mag_max = 17.0, 23.0
dm = 0.5
mag_bins = np.arange(mag_min, mag_max + 1. * dm, dm)
mag_mids = calc_mids(mag_bins)

ic_guess = 0.01

default_treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"

field_choices = ["SA", "LH", "EN", "XM", "Euclid"]
default_fields = ["SA", "LH", "EN", "XM"]
object_choices = ["gals", "eros_245", "ero_295"]

if __name__ == "__main__":

    #import matplotlib
    #matplotlib.use('Agg')


    parser = ArgumentParser()
    parser.add_argument("--fields", choices=field_choices, nargs="+", default=default_fields)
    parser.add_argument("--K-cut", action="store", default=20.7, type=float)
    parser.add_argument("--objects", action="store", choices=object_choices, default="eros_245")
    parser.add_argument(
        "--treecorr-config", action="store", default=default_treecorr_config_path
    )
    parser.add_argument("--skip-correlation", action="store_true", default=False)
    args = parser.parse_args()

    fields = args.fields
    K_cut = args.K_cut
    skip_correlation = args.skip_correlation

    print("look at fields")

    with open(args.treecorr_config, "r") as f:
        treecorr_config = yaml.load(f, Loader=yaml.FullLoader)
    randoms_density = treecorr_config.pop("randoms_density", 10_000) # sq. deg.
    print(f"RANDOMS DENSITY IS: {randoms_density}")

    """
    jwk_counts = pd.read_csv(jwk_number_counts_path, delim_whitespace=True, na_values=["-"])
    lao_counts = pd.read_csv(lao_number_counts_path)
    mccracken_counts = pd.read_csv(mccracken_number_counts_path)
    kajisawa_counts = pd.read_csv(kajisawa_number_counts_path)"""

    kim11_dat = pd.read_csv(kim11_number_counts_path)
    kim14_dat = pd.read_csv(kim14_number_counts_path)
    arcilaosejo19_dat = pd.read_csv(arcilaosejo19_number_counts_path)
    mccracken10_dat = pd.read_csv(mccracken10_number_counts_path)
    kajisawa06_dat = pd.read_csv(kajisawa06_number_counts_path)

    Nero_fig, Nero_ax = plt.subplots()
    Nero_fig.suptitle("Number counts EROs")
    Nero_ax.scatter(
        kim14_dat["Kmag"].values, kim14_dat["galaxies"].values, 
        s=20, marker="o", color="k", label="Kim+14, galaxies"
    )
    Nero_ax.scatter(
        kim14_dat["Kmag"].values, kim14_dat["ero245_hsc"].values, 
        s=20, marker="x", color="k", label=r"Kim+2014, EROs $i-K>2.45$"
    )
    Nero_ax.scatter(
        kim14_dat["Kmag"].values, kim14_dat["ero295_hsc"].values, 
        s=20, marker="^", color="k", label=r"Kim+2014, EROs $i-K>2.95$"
    )

    NsfgzK_fig, NsfgzK_ax = plt.subplots()
    NsfgzK_fig.suptitle("Number counts SF galaxies")
    NsfgzK_ax.scatter(
        arcilaosejo19_dat["Kmag"].values, arcilaosejo19_dat["sf_gzK"].values, 
        s=20, marker="x", color="k", label="Arcila-Osejo+2019, SF gzK",
    )
    NsfgzK_ax.scatter(
        mccracken10_dat["Kmag"].values, mccracken10_dat["sf_BzK"].values, 
        s=20, marker="^", color="k", label="McCracken+2010, SF BzK",
    )
    NsfgzK_ax.scatter(
        mccracken10_dat["Kmag"].values, mccracken10_dat["galaxies"].values, 
        s=20, marker="o", color="k", label="McCracken+2010, galaxies"
    )

    NpegzK_fig, NpegzK_ax = plt.subplots()
    NpegzK_fig.suptitle("Number counts PE galaxies")
    NpegzK_ax.scatter(
        arcilaosejo19_dat["Kmag"].values, arcilaosejo19_dat["pe_gzK"].values, 
        s=20, marker="x", color="k", label="Arcila-Osejo+2019"
    )
    NpegzK_ax.scatter(
        mccracken10_dat["Kmag"].values, mccracken10_dat["pe_BzK"].values, 
        s=20, marker="^", color="k", label="McCracken+2010"
    )
    NpegzK_ax.scatter(
        mccracken10_dat["Kmag"].values, mccracken10_dat["galaxies"].values, 
        s=20, marker="o", color="k", label="McCracken+2010, galaxies"
    )

    Ndrg_fig, Ndrg_ax = plt.subplots()
    Ndrg_fig.suptitle("Number counts DRGs")
    Ndrg_ax.scatter(
        kajisawa06_dat["Kmag"].values, kajisawa06_dat["drg"].values, 
        s=20, marker="^", color="k", label="Kajisawa+2006, J-K[Vega] > 2.3"
    )
    Ndrg_ax.scatter(
        kim11_dat["Kmag"].values, kim11_dat["drg"].values, 
        s=20, marker="x", color="k", label="Kim+2011, J-K[Vega] > 2.3"
    )
    sfdq = sfd.SFDQuery()

    """
    corr_fig, corr_ax = plt.subplots()
    if args.objects == "eros_245":
        jwk_path = paths.input_data_path / "plotting/jwk_2pcf_ps_eros_245.csv"
    else:
        jwk_path = None
    if jwk_path is not None:
        jwk = pd.read_csv(jwk_path, names=["x", "w"])
        corr_ax.scatter(jwk["x"], jwk["w"], color="k", label="Kim+ 2014")"""
        
    correlator_results = {}
    for ii, field in enumerate(fields):
        print_header(f"look at field {field}")

        if field in ["SA", "EN", "LH", "XM"]:
            suffix = "_panstarrs"
            catalog_path = paths.get_catalog_path(field, 0, "", suffix=suffix)
            if not catalog_path.exists():
                print(f"skip {catalog_path}: missing")
                continue
            full_catalog = Table.read(catalog_path)

            nir_mag_type = "aper_30"
            nir_tot_mag_type = "auto"
            opt_mag_type = "aper_30" # ps use aper_18 or aper_30, cfhtls use aper_2 or aper_3.

            gmag = f"g_mag_{opt_mag_type}"
            imag = f"i_mag_{opt_mag_type}"
            zmag = f"z_mag_{opt_mag_type}"
            Jmag = f"J_mag_{nir_mag_type}"
            Kmag = f"K_mag_{nir_mag_type}"
            Ktot_mag = f"K_mag_{nir_tot_mag_type}"
            
            logger.info("apply reddening")
            coords = SkyCoord(ra=full_catalog["ra"], dec=full_catalog["dec"], unit="deg")
            ebv = sfdq(coords)

            full_catalog[gmag] = apply_extinction(full_catalog[gmag], ebv, band="g")
            full_catalog[imag] = apply_extinction(full_catalog[imag], ebv, band="i")
            full_catalog[zmag] = apply_extinction(full_catalog[zmag], ebv, band="z")

            logger.info("do ab transform")
            #full_catalog[gmag] = vega_to_ab(full_catalog[gmag], band="g")
            #full_catalog[imag] = vega_to_ab(full_catalog[imag], band="i")
            #full_catalog[zmag] = vega_to_ab(full_catalog[zmag], band="z")
            
            full_catalog[Jmag] = vega_to_ab(full_catalog[Jmag], band="J")
            full_catalog[Kmag] = vega_to_ab(full_catalog[Kmag], band="K")
            full_catalog[Ktot_mag] = vega_to_ab(full_catalog[Ktot_mag], band="K")

            catalog = Query(
                "J_crosstalk_flag < 1", "K_crosstalk_flag < 1", 
                "J_coverage > 0", "K_coverage > 0"
            ).filter(full_catalog)

            if "panstarrs" in suffix and "hsc" not in suffix:
                opt = "panstarrs"
            elif "hsc" in suffix and "panstarrs" not in suffix:
                opt = "hsc"
            elif "panstarrs" in suffix and "hsc" in suffix:
                raise ValueError(f"suffix {suffix} contains hsc AND panstarrs?!")
            else:
                raise ValueError(f"suffix {suffix} has no optical data?")
            optical_mask_path = paths.input_data_path / f"external/{opt}/masks/{field}_mask.fits"
            print_path = optical_mask_path.relative_to(paths.base_path)
            print(f"optical mask at\n    {print_path}")
            
            J_mask_path = paths.masks_path / f"{field}_J_good_cov_mask.fits"
            K_mask_path = paths.masks_path / f"{field}_K_good_cov_mask.fits"
            opt_mask_list = [J_mask_path, K_mask_path, optical_mask_path]
            nir_mask_list = [J_mask_path, K_mask_path]

        if field == "Euclid":
            euclid_cat_path = paths.input_data_path / "external/euclidsim/euclid_k.cat.fits"
            catalog = Table.read(euclid_cat_path)
            opt_mask_list = [paths.input_data_path / "external/euclidsim/euclid_mask.fits"]
            nir_mask_list = opt_mask_list

            nir_mag_type = "kron"
            nir_tot_mag_type = "kron"
            opt_mag_type = "kron" # ps use aper_18 or aper_30, cfhtls use aper_2 or aper_3.

            gmag = f"g_mag_{opt_mag_type}"
            imag = f"i_mag_{opt_mag_type}"
            zmag = f"z_mag_{opt_mag_type}"
            Jmag = f"J_mag_{nir_mag_type}"
            Kmag = f"K_mag_{nir_mag_type}"
            Ktot_mag = f"K_mag_{nir_tot_mag_type}"

            #h_factor = 0.0np.log10(0.7)
            #print(h_factor)
            #for mag_col in [gmag, imag, zmag, Jmag, Kmag, Ktot_mag]:                
            #    catalog[mag_col] = catalog[mag_col] + h_factor


        ra_limits = calc_range(catalog["ra"])
        dec_limits = calc_range(catalog["dec"])
        opt_area = calc_survey_area(
            opt_mask_list, ra_limits=ra_limits, dec_limits=dec_limits, density=randoms_density
        )
        nir_area = calc_survey_area(
            nir_mask_list, ra_limits=ra_limits, dec_limits=dec_limits, density=randoms_density
        )
        
        print(f"{field}: NIR area: {nir_area:.2f}, opt_area: {opt_area:.2f}")

        ##=========== select galaxies ===========]##

        opt_catalog_mask = objects_in_coverage(
            opt_mask_list, catalog["ra"], catalog["dec"]
        )
        opt_catalog = catalog[ opt_catalog_mask ]

        gals = Query(
            f"({Jmag}-{J_offset}) - ({Kmag}-{K_offset}) > 1.0",
        ).filter(catalog)
        gal_hist, _ = np.histogram(gals[Ktot_mag], bins=mag_bins)
        gal_norm = gal_hist / (nir_area)
        gal_err = np.sqrt(gal_hist) / nir_area

        ##=========== select eros  ===========##
    
        eros_245 = Query(f"{imag} - {Kmag} > 2.45", f"{imag} < 25.0").filter(opt_catalog) #gals)
        ero245_hist, _ = np.histogram(eros_245[Ktot_mag], bins=mag_bins)
        ero245_norm = ero245_hist / (opt_area)
        ero245_err = np.sqrt(ero245_hist) / opt_area

        ero_fig, ero_ax = plt.subplots()
        ero_ax.scatter(eros_245["ra"], eros_245["dec"], s=1, color="k")
        
        eros_295 = Query(f"{imag} - {Kmag} > 2.95", f"{imag} < 25.0").filter(opt_catalog) #gals)
        ero295_hist, _ = np.histogram(eros_295[Ktot_mag], bins=mag_bins)
        ero295_norm = ero295_hist / (opt_area)
        ero295_err = np.sqrt(ero295_hist) / opt_area

        Nero_ax.plot(mag_mids, ero245_norm, color=f"C{ii}", label=field)
        Nero_ax.plot(mag_mids, ero295_norm, color=f"C{ii}", ls=":")
        Nero_ax.plot(mag_mids, gal_norm, color=f"C{ii}", ls="--")

        ##=========== select SF gzKs ===========##

        sf_gzK = Query(
            f"({zmag}-{Kmag}) - 1.27 * ({gmag}-{zmag}) >= -0.022",
            f"{zmag} < 50.", 
            f"{gmag} < 50.",
        ).filter(opt_catalog)
        sf_gzK_hist, _ = np.histogram(sf_gzK[Ktot_mag], bins=mag_bins)
        sf_gzK_norm = sf_gzK_hist / (opt_area)
        sf_gzK_err = np.sqrt(sf_gzK_hist) / (opt_area)
        NsfgzK_ax.plot(mag_mids, sf_gzK_norm, color=f"C{ii}", label=field)
        NsfgzK_ax.plot(mag_mids, gal_norm, color=f"C{ii}", ls="--")

        ##========== select PE gzKs ===========##
    
        pe_gzK = Query(
            f"({zmag}-{Kmag}) - 1.27 * ({gmag}-{zmag}) < -0.022", 
            f"{zmag}-{Kmag} > 2.55",
            f"{zmag} < 50.",
            f"{gmag} < 50.",
        ).filter(opt_catalog)
        pe_gzK_hist, _ = np.histogram(pe_gzK[Ktot_mag], bins=mag_bins)
        pe_gzK_norm = pe_gzK_hist / (opt_area)
        pe_gzK_err = np.sqrt(pe_gzK_hist) / opt_area
        NpegzK_ax.plot(mag_mids, pe_gzK_norm, color=f"C{ii}", label=field)
        NpegzK_ax.plot(mag_mids, gal_norm, color=f"C{ii}", ls="--")

        ##===========select drgs ===========##

        JK_cut = 2.3 + J_offset - K_offset

        drg = Query(
            f"({Jmag}-{Kmag} > {JK_cut})", #f"({Jmag} < 22.0 + {J_offset})"
        ).filter(gals)
        drg_hist, _ = np.histogram(drg[Ktot_mag], bins=mag_bins)
        drg_norm = drg_hist / nir_area
        Ndrg_ax.plot(mag_mids, drg_norm, color=f"C{ii}", label=field)
        Ndrg_ax.plot(mag_mids, gal_norm, color=f"C{ii}", ls="--")

        Nero_ax.legend()
        Nero_ax.semilogy()

        NsfgzK_ax.semilogy()
        NsfgzK_ax.legend()

        NpegzK_ax.semilogy()
        NpegzK_ax.legend()

        Ndrg_ax.semilogy()
        Ndrg_ax.legend()
        

        """
        iK_fig, iK_ax = plt.subplots()
        iK_ax.scatter(catalog[Kmag], catalog[imag]-catalog[Kmag], s=1, color="k")
        iK_fig.suptitle(field)

        gzK_fig, gzK_ax = plt.subplots()
        gzK_ax.scatter(catalog[gmag] - catalog[zmag], catalog[zmag] - catalog[Kmag], s=1, color="k")
        gzK_ax.scatter(sf_gzK[gmag] - sf_gzK[zmag], sf_gzK[zmag] - sf_gzK[Kmag], s=1, color="r")
        gzK_ax.scatter(pe_gzK[gmag] - pe_gzK[zmag], pe_gzK[zmag] - pe_gzK[Kmag], s=1, color="r")        
        """
    plt.show()
        

