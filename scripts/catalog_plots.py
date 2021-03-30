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

from easyquery import Query
from treecorr import NNCorrelation, Catalog

from dxs.utils.image import uniform_sphere, objects_in_coverage, calc_survey_area
from dxs import QuickPlotter
from dxs.utils.misc import calc_range, calc_mids, print_header
from dxs.utils.phot import ab_to_vega, vega_to_ab
from dxs.utils.region import in_only_one_tile
from dxs import paths

logger = logging.getLogger("correlator")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

jwk_number_counts_path = paths.input_data_path / "plotting/jwk_AB_number_counts.csv"
lao_number_counts_path = paths.input_data_path / "plotting/lao_AB_number_counts.csv"

nir_mag_type = "aper_20"
nir_tot_mag_type = "auto"
opt_mag_type = "aper_30" # ps use aper_18 or aper_30, cfhtls use aper_2 or aper_3.

gmag = f"g_mag_{opt_mag_type}"
imag = f"i_mag_{opt_mag_type}"
zmag = f"z_mag_{opt_mag_type}"
Jmag = f"J_mag_{nir_mag_type}"
Kmag = f"K_mag_{nir_mag_type}"
Ktot_mag = f"K_mag_{nir_tot_mag_type}"


mag_min, mag_max = 17.0, 23.0
dm = 0.5
mag_bins = np.arange(mag_min, mag_max + 1. * dm, dm)
mag_mids = calc_mids(mag_bins)

ic_guess = 0.01

default_treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"

field_choices = ["SA", "LH", "EN", "XM"]
object_choices = ["gals", "eros_245", "ero_295"]

if __name__ == "__main__":

    #import matplotlib
    #matplotlib.use('Agg')


    parser = ArgumentParser()
    parser.add_argument("--fields", choices=field_choices, nargs="+", default=field_choices)
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

    with open(args.treecorr_config, "r") as f:
        treecorr_config = yaml.load(f, Loader=yaml.FullLoader)
    randoms_density = treecorr_config.pop("randoms_denity", 10_000) # sq. deg.

    jwk_counts = pd.read_csv(jwk_number_counts_path, delim_whitespace=True, na_values=["-"])
    lao_counts = pd.read_csv(lao_number_counts_path)

    Nero_fig, Nero_ax = plt.subplots()
    Nero_ax.scatter(
        jwk_counts["Kmag"].values, jwk_counts["galaxies"].values, 
        s=20, marker="o", color="k", label="Kim+14, galaxies"
    )
    Nero_ax.scatter(
        jwk_counts["Kmag"].values, jwk_counts["ero245_ps"].values, 
        s=20, marker="x", color="k", label=r"Kim+14, EROs $i-K>2.45$"
    )
    Nero_ax.scatter(
        jwk_counts["Kmag"].values, jwk_counts["ero295_ps"].values, 
        s=20, marker="^", color="k", label=r"Kim+14, EROs $i-K>2.95$"
    )   

    NgzK_fig, NgzK_ax = plt.subplots()
    NgzK_ax.scatter(
        lao_counts["Kmag"].values, lao_counts["sf_gzK"].values, 
        s=20, marker="x", color="k", label="AO+2019, starforming",
    )
    NgzK_ax.scatter(
        lao_counts["Kmag"].values, lao_counts["pe_gzK"].values, 
        s=20, marker="^", color="k", label="AO+2019, passive"
    )


    corr_fig, corr_ax = plt.subplots()
    if args.objects == "eros_245":
        jwk_path = paths.input_data_path / "plotting/jwk_2pcf_ps_eros_245.csv"
    else:
        jwk_path = None
    if jwk_path is not None:
        jwk = pd.read_csv(jwk_path, names=["x", "w"])
        corr_ax.scatter(jwk["x"], jwk["w"], color="k", label="Kim+ 2014")
        
    correlator_results = {}
    for ii, field in enumerate(fields):
        print_header(f"look at field {field}")
        catalog_path = paths.get_catalog_path(field, 0, "", suffix="_panstarrs")
        if not catalog_path.exists():
            continue
        full_catalog = Table.read(catalog_path)

        full_catalog[gmag] = vega_to_ab(full_catalog[gmag], band="g")
        full_catalog[imag] = vega_to_ab(full_catalog[imag], band="i")
        full_catalog[zmag] = vega_to_ab(full_catalog[zmag], band="z")
        
        full_catalog[Jmag] = vega_to_ab(full_catalog[Jmag], band="J")
        full_catalog[Kmag] = vega_to_ab(full_catalog[Kmag], band="K")
        full_catalog[Ktot_mag] = vega_to_ab(full_catalog[Ktot_mag], band="K")

        catalog = Query(
            "J_crosstalk_flag < 1", "K_crosstalk_flag < 1", 
            "J_coverage > 0", "K_coverage > 0"
        ).filter(full_catalog)
                
        optical_mask_path = paths.input_data_path / f"external/panstarrs/masks/{field}_mask.fits"
        J_mask_path = paths.masks_path / f"{field}_J_good_cov_mask.fits"
        K_mask_path = paths.masks_path / f"{field}_K_good_cov_mask.fits"
        opt_mask_list = [J_mask_path, K_mask_path, optical_mask_path]
        nir_mask_list = [J_mask_path, K_mask_path]

        ra_limits = calc_range(catalog["ra"])
        dec_limits = calc_range(catalog["dec"])
        opt_area = calc_survey_area(
            opt_mask_list, ra_limits=ra_limits, dec_limits=dec_limits, density=1e5
        )
        nir_area = calc_survey_area(
            nir_mask_list, ra_limits=ra_limits, dec_limits=dec_limits, density=1e5
        )
        
        print(f"{field}: NIR area: {nir_area:.2f}, opt_area: {opt_area:.2f}")

        gals = Query(f"({Jmag}-0.9) - ({Kmag}-1.9) > 1.0").filter(catalog)
        gal_hist, _ = np.histogram(gals[Ktot_mag], bins=mag_bins)
        gal_hist = gal_hist / nir_area

        eros_245 = Query(f"{imag} - {Kmag} > 2.45", f"{imag} < 25.0").filter(catalog)
        ero245_hist, _ = np.histogram(eros_245[Ktot_mag], bins=mag_bins)
        ero245_hist = ero245_hist / (opt_area)
        
        eros_295 = Query(f"{imag} - {Kmag} > 2.95", f"{imag} < 25.0").filter(catalog)
        ero295_hist, _ = np.histogram(eros_295[Ktot_mag], bins=mag_bins)
        ero295_hist = ero295_hist / (opt_area)

        Nero_ax.plot(mag_mids, ero245_hist, color=f"C{ii}", label=field)
        Nero_ax.plot(mag_mids, ero295_hist, color=f"C{ii}", ls=":")
        Nero_ax.plot(mag_mids, gal_hist, color=f"C{ii}", ls="--")

        sf_gzK = Query(
            f"({zmag}-{Kmag}) - 1.27 * ({gmag}-{zmag}) >= -0.022", 
            f"{zmag} < 50.", 
            f"{gmag} < 50.",
        ).filter(catalog)
        sf_gzK_hist, _ = np.histogram(sf_gzK[Ktot_mag], bins=mag_bins)
        sf_gzK_hist = sf_gzK_hist / (opt_area)

        pe_gzK = Query(
            f"({zmag}-{Kmag}) -1.27 * ({gmag}-{zmag}) < -0.022", 
            f"{zmag}-{Kmag} > 2.55",
            f"{zmag} < 50.",
            f"{gmag} < 50.",
        ).filter(catalog)
        pe_gzK_hist, _ = np.histogram(pe_gzK[Ktot_mag], bins=mag_bins)
        pe_gzK_hist = pe_gzK_hist / (opt_area)

        NgzK_ax.plot(mag_mids, pe_gzK_hist, color=f"C{ii}", label=field)
        NgzK_ax.plot(mag_mids, sf_gzK_hist, color=f"C{ii}", ls="--")

        Nero_ax.legend()
        Nero_ax.semilogy()

        NgzK_ax.semilogy()
        NgzK_ax.legend()

        pkl_path = paths.data_path / f"{field}_{args.objects}_K{int(K_cut*10)}_Ncorr_data.pkl"
        if not skip_correlation:
            if args.objects == "eros_245":
                cat = eros_245
                object_area = opt_area
            if args.objects == "gals":
                cat = gals
                object_area = nir_area

            obj_cat = Query(f"{Ktot_mag} < {K_cut}").filter(cat)

            obj_fig, obj_ax = plt.subplots()
            obj_ax.scatter(obj_cat["ra"], obj_cat["dec"], s=1, color="k")           
            
            data_catalog = Catalog(
                ra=obj_cat["ra"], 
                dec=obj_cat["dec"], 
                ra_units="deg", dec_units="deg",
                npatch=treecorr_config.get("npatch", 1),
            )
            # Now sort some randoms.
            full_randoms = SkyCoord(
                uniform_sphere(ra_limits, dec_limits, density=randoms_density), 
                unit="degree"
            )
            random_mask = objects_in_coverage(
                opt_mask_list, full_randoms.ra, full_randoms.dec
            )
            randoms = full_randoms[ random_mask ]
            random_catalog = Catalog(
                ra=randoms.ra, 
                dec=randoms.dec, 
                ra_units="deg", dec_units="deg",
                patch_centers=data_catalog.patch_centers,
            )
            if "num_threads" not in treecorr_config:
                treecorr_config["num_threads"] = 3
            dd = NNCorrelation(config=treecorr_config)
            rr = NNCorrelation(config=treecorr_config)
            dr = NNCorrelation(config=treecorr_config)

            logger.info("starting dd process")
            dd.process(data_catalog)
            logger.info("starting rr process")
            rr.process(random_catalog)
            logger.info("starting dr process")
            dr.process(data_catalog, random_catalog)

            w_ls, w_ls_var = dd.calculateXi(rr=rr, dr=dr)

            corr_data = {
                "x": dd.rnom/3600.,
                "dd": dd.npairs,
                "dr": dr.npairs,
                "rr": rr.npairs,
                "w_ls": w_ls,
                "w_ls_var": w_ls_var,
                "n_data": len(obj_cat),
                "n_random": len(randoms.ra),
                "area": object_area,
                "random_density": randoms_density,
                "K_cut": K_cut
            }

            with open(pkl_path, "wb+") as f:
                pickle.dump(corr_data, f)

        else:
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    corr_data = pickle.load(f)
            else:
                corr_data = None

        if corr_data is not None:
            corr_ax.plot(corr_data["x"], corr_data["w_ls"], color=f"C{ii}", label=field)
            corr_ax.plot(corr_data["x"], corr_data["w_ls"] + ic_guess, color=f"C{ii}", ls="--", label=field)
            correlator_results[field] = corr_data
    DD_total = np.vstack([v["dd"] for k, v in correlator_results.items()]).sum(axis=0)
    DR_total = np.vstack([v["dr"] for k, v in correlator_results.items()]).sum(axis=0)
    RR_total = np.vstack([v["rr"] for k, v in correlator_results.items()]).sum(axis=0)

    print(DD_total, DR_total, RR_total)
    
   
    nD_total = np.sum([v["n_data"] for k, v in correlator_results.items()])
    nR_total = np.sum([v["n_random"] for k, v in correlator_results.items()])
    print(nD_total, nR_total)


    dd_total = DD_total / (0.5 * nD_total * (nD_total - 1))
    dr_total = DR_total / (nD_total * nR_total)
    rr_total = RR_total / (0.5 * nR_total * (nR_total - 1))

    ls_total = (dd_total - 2 * dr_total + rr_total) / rr_total

    corr_ax.plot(corr_data["x"], ls_total, color="k")
    corr_ax.plot(corr_data["x"], ls_total + ic_guess, ls="--", color="k")
    

corr_ax.loglog()
corr_ax.set_xlim(3e-4, 3e0)
corr_ax.set_ylim(5e-3, 1e1)
corr_ax.legend()
plt.show()
plt.close()


        
        





