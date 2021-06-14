import logging
import yaml
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy import cosmology
from astropy.coordinates import SkyCoord
from astropy.table import Table

from dustmaps import sfd
from easyquery import Query
from treecorr import NNCorrelation, Catalog

from dxs.utils.phot import ab_to_vega, vega_to_ab, apply_extinction
from dxs.utils.image import calc_survey_area
from dxs.utils.misc import print_header, calc_mids

from dxs import analysis_tools as tools

from dxs import paths

logger = logging.getLogger("analysis")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

analysis_config_path = paths.config_path / "analysis_config.yaml"
with open(analysis_config_path, "r") as f:
    analysis_config = yaml.load(f, Loader=yaml.FullLoader)

counts_config = analysis_config["counts"]
lf_config = analysis_config["luminosity_function"]
uvj_config = analysis_config["uvj"]
corr_config = analysis_config["correlation_function"]

default_treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"

field_choices = ["EN", "SA", "LH", "XM"]
object_choices = ["eros_245", "eros_295", "drgs", "sf_gzKs", "pe_gzKs"]

opt_mag_type = "aper_30"
nir_mag_type = "aper_30"
mir_mag_type = "aper_30"

nir_tot_mag_type = "auto"

gmag = f"g_mag_{opt_mag_type}"
rmag = f"r_mag_{opt_mag_type}"
imag = f"i_mag_{opt_mag_type}"
zmag = f"z_mag_{opt_mag_type}"
ymag = f"z_mag_{opt_mag_type}"
opt_mags = {"g": gmag, "r": rmag, "i": imag, "z": zmag, "y": ymag}

Jmag = f"J_mag_{nir_mag_type}"
Hmag = f"H_mag_{nir_mag_type}"
Kmag = f"K_mag_{nir_mag_type}"
nir_mags = {"J": Jmag, "H": Hmag, "K": Kmag}

I1mag = f"I1_mag_{mir_mag_type}"
I2mag = f"I2_mag_{mir_mag_type}"

Ktot_mag = f"K_mag_{nir_tot_mag_type}"

plotting_data_dir = paths.input_data_path / "plotting"
kim11_number_counts_path = plotting_data_dir / "kim11_number_counts.csv"
kim14_number_counts_path = plotting_data_dir / "kim14_number_counts.csv"
arcilaosejo19_number_counts_path = plotting_data_dir / "arcilaosejo19_number_counts.csv"
mccracken10_number_counts_path = plotting_data_dir / "mccracken10_number_counts.csv"
kajisawa06_number_counts_path = plotting_data_dir / "kajisawa06_number_counts.csv"
hartley08_lf_path = plotting_data_dir / "hartley2008_BzK_lf.csv"

selection_queries = {
    #"galaxies": (f"({Jmag} - 0.938) - ({Kmag} - 1.900) > 1.0", ),
    "eros_245": (f"{imag} - {Kmag} > 2.45", f"{imag} < 25.0", ),
    "eros_295": (f"{imag} - {Kmag} > 2.95", f"{imag} < 25.0", ),
    "drgs": (f"({Jmag} - 0.938) - ({Kmag} - 1.900) > 2.3", ),
    "sf_gzKs": (
        f"({zmag} - {Kmag}) - 1.27 * ({gmag} - {zmag}) >= -0.022", 
        f"{gmag} < 50.0", f"{zmag} < 50.0", 
    ),
    "pe_gzKs": (
        f"({zmag} - {Kmag}) - 1.27 * ({gmag} - {zmag}) < -0.022", 
        f"{zmag} - {Kmag} > 2.55", 
        f"{gmag} < 50.0", f"{zmag} < 50.0", 
    ),
}

cosmol = cosmology.FlatLambdaCDM(H0=70.0, Om0=0.3, ) #Planck15

if __name__ == "__main__":

    ###====================== sort the inputs ====================###
    parser = ArgumentParser()
    parser.add_argument("--fields", default=field_choices, choices=field_choices, nargs="+")
    parser.add_argument("--optical", default="panstarrs")
    parser.add_argument("--mir", default=None)
    parser.add_argument("--zphot", nargs="?", default=None)
    parser.add_argument("--objects", default=object_choices, choices=object_choices, nargs="+")
    parser.add_argument("--lf", default=False, action="store_true")
    parser.add_argument("--corr", default=False, action="store_true")
    parser.add_argument("--treecorr-config", default=default_treecorr_config_path)
    parser.add_argument("--uvj", default=False, action="store_true")
    args = parser.parse_args()

    suffix = f"_{args.optical}"
    if args.mir is not None:
        suffix = suffix + f"_{args.mir}"
    if args.zphot is not None:
        suffix = suffix + f"_zphot_{args.zphot}"

    print(args.objects)

    with open(args.treecorr_config, "r") as f:
        treecorr_config = yaml.load(f, Loader=yaml.FullLoader)
    randoms_density = treecorr_config.pop("randoms_density", 1e4)

    print(f"randoms density is {int(randoms_density)} / sq. deg.")
        

    ###==================== make some useful arrays ================###
    K_bright = counts_config["K_bright"]
    K_faint = counts_config["K_faint"]
    d_K = counts_config["Kbin_width"]
    K_bins = np.arange(K_bright, K_faint + d_K, d_K)
    K_mids = calc_mids(K_bins)

    d_absM = lf_config["absM_bin_width"]
    bins_M_bright = lf_config["M_bright"]
    bins_M_faint = lf_config["M_faint"]
    absM_bins = np.arange(bins_M_bright, bins_M_faint + d_absM, d_absM)
    absM_mids = calc_mids(absM_bins)

    ###==================== load some other stuff ===================###
    sfdq = sfd.SFDQuery()

    ###=================== load some plotting data =================###
    """
    kim11_dat = pd.read_csv(kim11_number_counts_path)
    kim14_dat = pd.read_csv(kim14_number_counts_path)
    arcilaosejo19_dat = pd.read_csv(arcilaosejo19_number_counts_path)
    mccracken10_dat = pd.read_csv(mccracken10_number_counts_path)
    kajisawa06_dat = pd.read_csv(kajisawa06_number_counts_path)
    """
    hartley08_dat = pd.read_csv(hartley08_lf_path)

    ###================== initalize plots ==================###

    counts_plot_lookup = {}
    lf_plot_lookup = {}
    corr_plot_lookup = {}

    if "eros_245" in args.objects or "eros_295" in args.objects:
        Nero_fig, Nero_ax = plt.subplots()
        Nero_fig.suptitle("ERO number counts")
        counts_plot_lookup["eros_245"] = (Nero_fig, Nero_ax)
        counts_plot_lookup["eros_295"] = (Nero_fig, Nero_ax)
        """
        Nero_ax.scatter(
            kim14_dat["Kmag"].values, kim14_dat["galaxies"].values, 
            s=20, marker="o", color="k", label="Kim+14, galaxies"
        )
        if "eros_245" in args.objects:                
            if args.optical == "panstarrs":
                Nero_ax.scatter(kim14_dat["Kmag"].values, kim14_dat["ero245_ps"].values)
            elif args.optical == "hsc":
                Nero_ax.scatter(kim14_dat["Kmag"].values, kim14_dat["ero245_hsc"].values)
        if "eros_295" in args.objects:                
            if args.optical == "panstarrs":
                Nero_ax.scatter(kim14_dat["Kmag"].values, kim14_dat["ero295_ps"].values)
            elif args.optical == "hsc":
                Nero_ax.scatter(kim14_dat["Kmag"].values, kim14_dat["ero295_hsc"].values)
        """
        Nero_ax.semilogy()
       
    if "sf_gzKs" in args.objects:
        NsfgzK_fig, NsfgzK_ax = plt.subplots()
        NsfgzK_fig.suptitle("Number counts SF galaxies")
        counts_plot_lookup["sf_gzKs"] = (NsfgzK_fig, NsfgzK_ax)
        """
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
        """
        NsfgzK_ax.semilogy()

    if "pe_gzKs" in args.objects:
        NpegzK_fig, NpegzK_ax = plt.subplots()
        NpegzK_fig.suptitle("Number counts PE galaxies")
        counts_plot_lookup["pe_gzKs"] = (NpegzK_fig, NpegzK_ax)
        """
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
        """
        NpegzK_ax.semilogy()


    if "drgs" in args.objects:
        Ndrg_fig, Ndrg_ax = plt.subplots()
        Ndrg_fig.suptitle("Number counts DRGs")
        counts_plot_lookup["drgs"] = (Ndrg_fig, Ndrg_ax)
        """
        Ndrg_ax.scatter(
            kajisawa06_dat["Kmag"].values, kajisawa06_dat["drg"].values, 
            s=20, marker="^", color="k", label="Kajisawa+2006, J-K[Vega] > 2.3"
        )
        Ndrg_ax.scatter(
            kim11_dat["Kmag"].values, kim11_dat["drg"].values, 
            s=20, marker="x", color="k", label="Kim+2011, J-K[Vega] > 2.3"
        )
        """
        Ndrg_ax.semilogy()

    if args.lf:
        lf_fig, lf_axes = plt.subplots(2, 4, figsize=(10, 5))
        lf_fig.subplots_adjust(wspace=0., hspace=0.)
        lf_axes = lf_axes.flatten()
        for kk, lf_ax in enumerate(lf_axes):
            lf_ax.semilogy()
            lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
            lf_ax.set_ylim(1e-7, 1e-2)
            if kk % 4 != 0:
                lf_ax.set_yticks([])
            if kk < 4:
                lf_ax.set_xticks([])

        if "eros_245" in args.objects:
            eros_245_lf_fig, eros_245_lf_ax = plt.subplots()
            eros_245_lf_ax.semilogy()
            eros_245_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
            lf_plot_lookup["eros_245"] = (eros_245_lf_fig, eros_245_lf_ax)

        if "eros_295" in args.objects:
            eros_295_lf_fig, eros_295_lf_ax = plt.subplots()
            eros_295_lf_ax.semilogy()
            eros_295_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
            lf_plot_lookup["eros_295"] = (eros_295_lf_fig, eros_295_lf_ax)

        if "drgs" in args.objects:
            drgs_lf_fig, drgs_lf_ax = plt.subplots()
            drgs_lf_ax.semilogy()
            drgs_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
            lf_plot_lookup["drgs"] = (drgs_lf_fig, drgs_lf_ax)
        
        if "sf_gzKs" in args.objects or "pe_gzKs" in args.objects:
            gzK_lf_fig, gzK_lf_ax = plt.subplots()
            gzK_lf_ax.semilogy()
            gzK_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
            if "pe_gzKs" in args.objects:
                gzK_lf_ax.plot(
                    hartley08_dat["K"].values, hartley08_dat["phi_pe_gzK"].values,
                    ls="--", color="r", label="Hartley+2008, PE"
                )
            if "sf_gzKs" in args.objects:
                gzK_lf_ax.plot(
                    hartley08_dat["K"].values, hartley08_dat["phi_sf_gzK"].values,
                    ls="--", color="b", label="Hartley+2008, SF"
                )
            lf_plot_lookup["sf_gzKs"] = (gzK_lf_fig, gzK_lf_ax)
            lf_plot_lookup["pe_gzKs"] = (gzK_lf_fig, gzK_lf_ax)

    if args.corr:
        if "eros_245" in args.objects:
            eros_245_corr_fig, eros_245_corr_ax = plt.subplots()
            eros_245_corr_ax.loglog()
            eros_245_corr_ax.set_xlim(5e-4, 5e0)
            eros_245_corr_ax.set_ylim(1e-3, 1e1)
            corr_plot_lookup["eros_245"] = (eros_245_corr_fig, eros_245_corr_ax)

        if "eros_295" in args.objects:
            eros_295_corr_fig, eros_295_corr_ax = plt.subplots()
            eros_295_corr_ax.loglog()
            eros_295_corr_ax.set_xlim(5e-4, 5e0)
            eros_295_corr_ax.set_ylim(1e-3, 1e1)
            corr_plot_lookup["eros_295"] = (eros_295_corr_fig, eros_295_corr_ax)

        if "drgs" in args.objects:
            drgs_corr_fig, drgs_lf_ax = plt.subplots()
            drgs_corr_ax.semilogy()
            drgs_corr_ax.set_xlim(5e-4, 5e0)
            drgs_corr_ax.set_ylim(1e-3, 1e1)
            corr_plot_lookup["drgs"] = (drgs_lf_fig, drgs_lf_ax)

        if "sf_gzK" in args.objects:
            sf_gzK_corr_fig, sf_gzK_corr_ax = plt.subplots()
            sf_gzK_corr_ax.loglog()
            sf_gzK_corr_ax.set_xlim(5e-4, 5e0)
            sf_gzK_corr_ax.set_ylim(1e-3, 1e1)
            corr_plot_lookup["sf_gzK"] = (sf_gzK_corr_fig, sf_gzK_corr_ax)

        if "pe_gzK" in args.objects:
            pe_gzK_corr_fig, pe_gzK_corr_ax = plt.subplots()
            pe_gzK_corr_ax.loglog()
            pe_gzK_corr_ax.set_xlim(5e-4, 5e0)
            pe_gzK_corr_ax.set_ylim(1e-3, 1e1)
            corr_plot_lookup["pe_gzK"] = (pe_gzK_corr_fig, pe_gzK_corr_ax)

    total_number_counts = {obj: np.zeros(len(K_mids)) for obj in args.objects}
    if "galaxies" not in total_number_counts:
        total_number_counts["galaxies"] = np.zeros(len(K_mids))
    total_area = {obj: 0. for obj in args.objects}
    if "galaxies" not in total_area:
        total_area["galaxies"] = 0.
    comb_corr_data = {obj: {} for obj in args.objects}

    for ii, field in enumerate(args.fields):
        print_header(f"look at {field}") 

        ###================= does the catalog exist? ===============###
        catalog_path = paths.get_catalog_path(field, 00, "", suffix=suffix)
        print_path = catalog_path.relative_to(paths.base_path)
        if not catalog_path.exists():
            print(f"no catalog {print_path}")
            continue
        print(f"load catalog {print_path}")

        ###====================== masks and area =====================###
        mask_lookup = {}
        for band in "J H K".split():
            mask_lookup[band] = paths.masks_path / f"{field}_{band}_good_cov_mask.fits"
        for band in "g r i z y".split():
            mask_lookup[band] = (
                paths.input_data_path / f"external/{args.optical}/masks/{field}_{band}_mask.fits"
            )

        JK_mask_list  = [mask_lookup[b] for b in "J K".split()]
        iJK_mask_list = [mask_lookup[b] for b in "i J K".split()]
        gzK_mask_list = [mask_lookup[b] for b in "g z J K".split()]
        
        for mask_list in [iJK_mask_list, gzK_mask_list, JK_mask_list]:
            for p in mask_list:
                if not p.exists():
                    raise IOError(f"no mask {p.relative_to(paths.base_path)}")

        JK_area = calc_survey_area(JK_mask_list, density=1e3)
        iJK_area = calc_survey_area(iJK_mask_list, density=1e3)
        gzK_area = calc_survey_area(gzK_mask_list, density=1e3)

        print(f"JK_area: {JK_area:.2f}, iJK_area: {iJK_area:.2f}, gzK_area: {gzK_area:.2f}")

        area_lookup = {
            "galaxies": JK_area, "drgs": JK_area, "eros_245": iJK_area, "eros_295": iJK_area, 
            "sf_gzKs": gzK_area, "pe_gzKs": gzK_area,            
        }

        mask_list_lookup = {
            "galaxies": JK_mask_list, "drgs": JK_mask_list, 
            "eros_245": iJK_mask_list, "eros_295": iJK_mask_list, 
            "sf_gzK": gzK_mask_list, "pe_gzK": gzK_mask_list,
        }

        ###================= load and prepare catalog =================###
        full_catalog = Table.read(catalog_path)

        # remove CROSSTALK artefacts
        catalog = Query(
            "J_crosstalk_flag < 1", "K_crosstalk_flag < 1", 
            "J_coverage > 0", "K_coverage > 0"
        ).filter(full_catalog)
        del full_catalog

        logger.info("apply reddening")
        coords = SkyCoord(ra=catalog["ra"], dec=catalog["dec"], unit="deg")
        ebv = sfdq(coords)
        for band, col in opt_mags.items():
            catalog[col] = apply_extinction(catalog[col], ebv, band=band)

        for band, col in nir_mags.items():
            if col not in catalog.columns:
                continue               
            catalog[col] = vega_to_ab(catalog[col], band=band)
        catalog[Ktot_mag] = vega_to_ab(catalog[Ktot_mag], band="K")

        catalog = Query(
            f"({Jmag} - 0.938) - ({Kmag} - 1.900) > 1.0",
        ).filter(catalog)

        ## Number counts of galaxies.
        gal_hist, _ = np.histogram(catalog[Ktot_mag], bins=K_bins)
        gal_hist_norm = gal_hist / JK_area
        gal_hist_err = np.sqrt(gal_hist) / JK_area
        
        total_number_counts["galaxies"] = total_number_counts["galaxies"] + gal_hist
        total_area["galaxies"] = total_area["galaxies"] + JK_area

        if args.lf:
            catalog = Query(
                f"0.1 < z_phot", f"z_phot < 2.5",
                f"0.1 < z_phot_chi2 / nusefilt", f"z_phot_chi2 / nusefilt < 100.0",
            ).filter(catalog)

            absM = tools.calc_absM(catalog[Ktot_mag], distmod=catalog["DISTMOD"], cosmol=cosmol)
            catalog.add_column(absM.data, name="absM")

            z_min = tools.r_at_ref_mag(lf_config["K_bright"], catalog["absM"], cosmol=cosmol)
            z_max = tools.r_at_ref_mag(lf_config["K_faint"], catalog["absM"], cosmol=cosmol)

            catalog.add_columns([z_min, z_max], names=["z_min", "z_max"])

            if "galaxies" in args.objects:
                gal_lf_config = lf_config["galaxies"]

                z_low_arr = gal_lf_config["z_bins_low"]
                z_high_arr = gal_lf_config["z_bins_high"]
                M_bright_arr = gal_lf_config["absM_bright"]
                M_faint_arr = gal_lf_config["absM_faint"]


                Mz_fig, Mz_ax = plt.subplots()
                Mz_ax.scatter(catalog["z_phot"], catalog["absM"], color="k", s=1, alpha=0.5)

                for kk, (z_low, z_high) in enumerate(zip(z_low_arr, z_high_arr)):
                    print(f"select galaxies {z_low} < z < {z_high}")
                    M_bright = M_bright_arr[kk]-1.0
                    M_faint = M_faint_arr[kk]-1.0
                    z_cat = Query(
                        f"{z_low} < z_phot", f"z_phot < {z_high}",
                        f"{M_faint} < absM", f"absM < {M_bright}"
                    ).filter(catalog)

                    Mz_ax.fill_between(
                        [z_low, z_high], y1=[M_bright, M_bright], y2=[M_faint, M_faint], 
                        color=f"C{kk}", alpha=0.5
                    )
                    Mz_ax.set_ylim(Mz_ax.get_ylim()[-1], Mz_ax.get_ylim()[0])

                    z_max = np.minimum(z_cat["z_max"], z_high)
                    z_min = np.maximum(z_cat["z_min"], z_low)
                    vmax = tools.calc_vmax(z_min, z_max, JK_area * (np.pi / 180.) ** 2, cosmol)
                    phi, phi_err = tools.calc_luminosity_function(z_cat["absM"], vmax, absM_bins)
                    phi = phi / d_absM

                    lf_ax = lf_axes[kk]
                    lf_ax.errorbar(absM_mids, phi, yerr=phi_err)
            



        ###===============================================================###
        ###========================== DO OBJECTS =========================###
        ###===============================================================###

        for jj, obj in enumerate(args.objects):

            ### ================= OBJECT NUMBER COUNTS ==================== ###
            Nfig, Nax = counts_plot_lookup[obj]   
            color_query = selection_queries[obj]
            selection_area = area_lookup[obj]

            print(f"\nselect {obj} with:\n {color_query}")
            selection = Query(*color_query).filter(catalog)
            Nhist, _ = np.histogram(selection[Ktot_mag], bins=K_bins)
            total_number_counts[obj] = total_number_counts[obj] + Nhist
            total_area[obj] = total_area[obj] + selection_area

            Nhist_norm = Nhist / selection_area
            Nhist_err = np.sqrt(Nhist) / selection_area

            Nax.errorbar(K_mids, Nhist_norm, yerr=Nhist_err, color=f"C{ii}", label=field, alpha=0.8)
            Nax.errorbar(K_mids, gal_hist_norm, yerr=gal_hist_err, color=f"C{ii}", ls="--")
            
            ### =============== OBJECT LUMINOSITY FUNCTIONS =============== ###
            if args.lf:
                obj_lf_fig, obj_lf_axes = lf_plot_lookup[obj]

                obj_lf_config = lf_config[obj]

                z_low_arr = obj_lf_config["z_bins_low"]
                z_high_arr = obj_lf_config["z_bins_high"]
                M_bright_arr = obj_lf_config["absM_bright"]
                M_faint_arr = obj_lf_config["absM_faint"]

                Mz_fig, Mz_ax = plt.subplots()
                Mz_ax.scatter(catalog["z_phot"], catalog["absM"], color="k", s=1, alpha=0.1)
                Mz_ax.scatter(selection["z_phot"], selection["absM"], color="r", s=1, alpha=0.6)

                for kk, (z_low, z_high) in enumerate(zip(z_low_arr, z_high_arr)):
                    print(f"select {obj} {z_low} < z < {z_high}")
                    M_bright = M_bright_arr[kk]
                    M_faint = M_faint_arr[kk]
                    obj_z_cat = Query(
                        f"{z_low} < z_phot", f"z_phot < {z_high}",
                        f"{M_faint} < absM", f"absM < {M_bright}",
                    ).filter(selection)

                    Mz_ax.fill_between(
                        [z_low, z_high], y1=[M_bright, M_bright], y2=[M_faint, M_faint], 
                        color=f"C{kk}", alpha=0.5
                    )
                    Mz_ax.set_ylim(Mz_ax.get_ylim()[-1], Mz_ax.get_ylim()[0])

                    z_max = np.minimum(obj_z_cat["z_max"], z_high)
                    z_min = np.maximum(obj_z_cat["z_min"], z_low)
                    vmax = tools.calc_vmax(z_min, z_max, selection_area * (np.pi / 180.) ** 2, cosmol)
                    phi, phi_err = tools.calc_luminosity_function(obj_z_cat["absM"], vmax, absM_bins)
                    phi = phi / d_absM

                    lf_ax = obj_lf_axes#[kk]
                    lf_ax.errorbar(absM_mids, phi, yerr=phi_err, color=f"C{ii}")
            
            ### ============== OBJECT CORRELATION FUNCTIONS =============== ###
            if args.corr:
                obj_corr_fig, obj_corr_ax = corr_plot_lookup[obj]
                obj_corr_config = corr_config[obj]
                for kk, K_lim in enumerate(obj_corr_config["K_limits"]):
                    print(f"corr for {obj} with K_tot < {K_lim}")
                    corr_selection = Query(f"{Ktot_mag} < {K_lim}").filter(selection)
                    ra_lim = (np.min(corr_selection["ra"]), np.max(corr_selection["ra"]))
                    dec_lim = (np.min(corr_selection["dec"]), np.max(corr_selection["dec"]))
                    obj_mask_list = mask_list_lookup[obj]
                    randoms = tools.get_randoms(
                        ra_lim, dec_lim, obj_mask_list, randoms_density=randoms_density
                    )

                    obj_pos_fig, obj_pos_ax = plt.subplots()
                    obj_pos_ax.scatter(randoms.ra, randoms.dec, s=1, color="k")
                    obj_pos_ax.scatter(corr_selection["ra"], corr_selection["dec"], s=1, color="r")
                
                    dd, dr, rd, rr = tools.prepare_components(
                        corr_selection, randoms, treecorr_config
                    )
                    x = dd.rnom / 3600.
                    w, w_err = dd.calculateXi(dr=dr, rd=rd, rr=rr)
                    obj_corr_ax.errorbar(x, w, yerr=w_err, color=f"C{ii}")
                    Nd = len(corr_selection)
                    Nr = len(randoms)
                    dd_pairs = dd.npairs
                    rr_pairs = rr.npairs
                    dr_pairs = dr.npairs if dr is not None else None
                    rd_pairs = rd.npairs if rd is not None else None
                    corr_data = {
                        "x": x, "dd": dd_pairs, "dr": dr_pairs, "rd": rd_pairs, "rr": rr_pairs, 
                        "w": w, "w_err": w_err, "Nd": Nd, "Nr": Nr
                    }
                    if kk not in comb_corr_data[obj]:
                        comb_corr_data[obj][kk] = {}
                    comb_corr_data[obj][kk][field] = corr_data
                    
        
        if args.uvj:
            uvj_fig, uvj_axes = plt.subplots(2, 3, figsize=(8, 6))
            uvj_axes = uvj_axes.flatten()
            z_low_arr = uvj_config["z_bins_low"]
            z_high_arr = uvj_config["z_bins_high"]

            V_J = 2.5 * np.log10(catalog["restJ"] / catalog["restV"])
            U_V = 2.5 * np.log10(catalog["restV"] / catalog["restU"])
            catalog.add_columns([U_V, V_J], names=["U_V", "V_J"])

            for kk, (z_low, z_high) in enumerate(zip(z_low_arr, z_high_arr)):
                uvj_ax = uvj_axes[kk]
                z_cat = Query(
                    f"{z_low} < z_phot", f"z_phot < {z_high}",
                    f"0.1 < z_phot / nusefilt", f"z_phot / nusefilt < 2.0",
                ).filter(catalog)

                uvj_ax.plot([-1.0, 0.75, 1.5, 1.5], [1.3, 1.3, 1.9, 3.0], color="k", lw=2)
                uvj_ax.set_xlim(-1.0, 3.0)
                uvj_ax.set_ylim(-1.0, 3.0)

                uvj_gals = Query(
                    "U_V > (0.8 * V_J + 0.7)", "U_V > 1.3", "V_J < 1.5"
                ).filter(z_cat)
                ssfr = np.log10(z_cat["sfr"] / z_cat["mass"])
                scatter = uvj_ax.scatter(
                    z_cat["V_J"], z_cat["U_V"], c=ssfr, 
                    s=1, alpha=0.5, vmin=-13., vmax=-9.
                )
                
            obj_lf_config = lf_config[obj]
            plt.colorbar(scatter, cax=uvj_axes[-1])


    ### combined info plotting.
    
    total_gal_hist = total_number_counts["galaxies"]
    total_gal_area = total_area["galaxies"]

    total_gal_hist_norm = total_gal_hist / total_gal_area
    total_gal_hist_err = np.sqrt(total_gal_hist) / total_gal_area

    for obj in args.objects:
        Nfig, Nax = counts_plot_lookup[obj]
        selection_area = total_area[obj]

        Nhist = total_number_counts[obj]
        
        Nhist_norm = Nhist / selection_area
        Nhist_err = np.sqrt(Nhist) / selection_area

        Nax.errorbar(K_mids, Nhist_norm, yerr=Nhist_err, label="comb.", color="k")
        Nax.errorbar(K_mids, gal_hist_norm, yerr=gal_hist_err, color="k") #, label="comb.", color="k")

        if args.corr:
            obj_corr_fig, obj_corr_ax = corr_plot_lookup[obj]
            obj_corr_data = comb_corr_data[obj]
            obj_corr_config = corr_config[obj]
            for kk, obj_corr_kk in obj_corr_data.items():
                x_list = [d["x"] for field, d in obj_corr_kk.items()]
                for x in x_list:
                    assert np.allclose(x, x_list[0])
                x = x_list[0]
            
                K_lim = obj_corr_config["K_limits"][kk]
                total_dd = sum([ d["dd"] for field, d in obj_corr_kk.items() ])
                total_dr = sum([ d["dr"] for field, d in obj_corr_kk.items() ])
                total_rr = sum([ d["rr"] for field, d in obj_corr_kk.items() ])

                Nd = sum([ d["Nd"] for field, d in obj_corr_kk.items() ])
                Nr = sum([ d["Nr"] for field, d in obj_corr_kk.items() ])

                DD_tot = total_dd / (0.5 * Nd * (Nd - 1))
                DR_tot = total_dr / (Nd * Nr)
                RR_tot = total_rr / (0.5 * Nr * (Nr - 1))

                w_tot = (DD_tot - 2 * DR_tot + RR_tot) / RR_tot
                w_err = (1. + w_tot) / np.sqrt(total_dd)
            
                obj_corr_ax.errorbar(x, w_tot, yerr=w_err, color="k")

    plt.show()







