import logging
import os
import pickle
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from astropy import cosmology
from astropy.coordinates import SkyCoord
from astropy.table import Table

from dustmaps import sfd
from easyquery import Query
from treecorr import NNCorrelation, Catalog, estimate_multi_cov

from dxs.utils.phot import ab_to_vega, vega_to_ab, apply_extinction
from dxs.utils.image import calc_survey_area, objects_in_coverage
from dxs.utils.misc import print_header, calc_mids

from dxs import analysis_tools as tools

from dxs import paths

logger = logging.getLogger("analysis")

if not os.isatty(0):
    logger.info("use mpl 'Agg' backend")
    matplotlib.use("Agg")

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

default_treecorr_config_path = paths.config_path / "treecorr/treecorr_default_d02.yaml"

field_choices = ["EN", "SA", "LH", "XM", "Euclid"]

object_types = ["galaxies", "eros_245", "eros_295", "drgs", "sf_gzKs", "pe_gzKs"]
object_names = {
    "galaxies": "galaxies", 
    "eros_245": r"EROs, $(i-K) > 2.45$", 
    "eros_295": r"EROs, $(i-K)_{AB} > 2.95$",
    "drgs": r"DRGs, $(J-K)_{AB}$ > 1.34",
    "sf_gzKs": "SF-gzKs",
    "pe_gzKs": "PE-gzKs",
}
select_from_galaxies = ["galaxies", "sf_gzKs", "pe_gzKs", "eros_245", "eros_295", ]
select_from_catalog = ["drgs"]

if not set(select_from_galaxies).isdisjoint(select_from_catalog):
    msg = "object types in 'select_from_galaxies' should not also be in 'select_from_catalog'"
    raise ValueError(msg)

object_choices = object_types + ["all"]

###====================== sort the inputs ====================###
parser = ArgumentParser()
parser.add_argument("--fields", default=field_choices, choices=field_choices, nargs="+")
parser.add_argument("--optical", default="panstarrs")
parser.add_argument("--mir", default=None)
parser.add_argument("--zphot", nargs="?", default=None)
parser.add_argument("--objects", default=[], choices=object_choices, nargs="+")
parser.add_argument("--color-plots", default=False, action="store_true")
parser.add_argument("--venn-plots", default=False, action="store_true")
parser.add_argument("--cp-density", default=0.2, type=float)
parser.add_argument("--lf", default=False, action="store_true")
parser.add_argument("--corr", default=False, action="store_true")
parser.add_argument("--treecorr-config", default=default_treecorr_config_path)
parser.add_argument("--uvj", default=False, action="store_true")
parser.add_argument("--save-figs", default=False, action="store_true")
parser.add_argument("--force-write", default=False, action="store_true")
parser.add_argument("--read-corr", default=False, action="store_true")
args = parser.parse_args()

#number_counts_plot_kwargs = {
#    "ref_galaxies": {"ls": ".",}
#    "galaxies": {"marker": "s", "ms": 10},
#    "eros_245": 
#    "sf_gzKs": 
    
    
cosmol = cosmology.FlatLambdaCDM(H0=70.0, Om0=0.3, ) #Planck15

#if __name__ == "__main__":




### Which magnitudes do we want to use?

opt_mag_type = "aper_30"
nir_mag_type = "aper_30"
mir_mag_type = "aper_30"

opt_tot_mag_type = "kron"
nir_tot_mag_type = "auto"

gmag = f"g_mag_{opt_mag_type}"
rmag = f"r_mag_{opt_mag_type}"
imag = f"i_mag_{opt_mag_type}"
zmag = f"z_mag_{opt_mag_type}"
ymag = f"y_mag_{opt_mag_type}"
opt_mags = {"g": gmag, "r": rmag, "i": imag, "z": zmag, "y": ymag}

Jmag = f"J_mag_{nir_mag_type}"
Hmag = f"H_mag_{nir_mag_type}"
Kmag = f"K_mag_{nir_mag_type}"
nir_mags = {"J": Jmag, "H": Hmag, "K": Kmag}

I1mag = f"I1_mag_{mir_mag_type}"
I2mag = f"I2_mag_{mir_mag_type}"

W1mag = f"W1_mag_auto"

itot_mag = f"i_mag_{opt_tot_mag_type}"
Ktot_mag = f"K_mag_{nir_tot_mag_type}"


### What selections will we make?

# MUST leave trailing comma, as queries are tuples.
selection_queries = {
    #"galaxies": (f"({Jmag} - 0.938) - ({Kmag} - 1.900) > 1.0", ),
    "eros_245": (f"{imag} - {Kmag} > 2.55", f"{imag} < 25.0", ),
    "eros_295": (f"{imag} - {Kmag} > 2.95", f"{imag} < 25.0", ),
    #"drgs": (f"({Jmag} - 0.938) - ({Kmag} - 1.900) > 2.3", ),
    "drgs": (f"{Jmag} - {Kmag} > 1.34", ), # AB equiv of >2.3
    "sf_gzKs": (
        f"({zmag} - {Kmag}) - 1.27 * ({gmag} - {zmag}) >= -0.022", 
        f"{gmag} < 50.0", f"{zmag} < 50.0",f"{Kmag} < 22.2"
    ),
    "pe_gzKs": (
        f"({zmag} - {Kmag}) - 1.27 * ({gmag} - {zmag}) < -0.022", 
        f"{zmag} - {Kmag} > 2.55", 
        f"{gmag} < 50.0", f"{zmag} < 50.0", f"{Kmag} < 22.5"
    ),
}


suffix = f"_{args.optical}"
if args.mir is not None:
    suffix = suffix + f"_{args.mir}"
if args.zphot is not None:
    suffix = suffix + f"_photoz_{args.zphot}"

if "all" in args.objects:
    args.objects = object_types

logger.info(f"objects: {args.objects}")       

if args.corr:
    print(f"treecorr_config: {Path(args.treecorr_config).name}")
with open(args.treecorr_config, "r") as f:
    corr_processing_config = yaml.load(f, Loader=yaml.FullLoader)
    treecorr_config = corr_processing_config["treecorr"]
randoms_density = corr_processing_config.get("randoms_density", 1e4)
patch_size = corr_processing_config.get("patch_size", None)

print(f"randoms density is {int(randoms_density)} / sq. deg.")
print(f"corr_processing_config: patch size {patch_size} sq. deg.")    

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
absM_grid = np.linspace(bins_M_bright, bins_M_faint, 100)

d_zdist = 0.1
zdist_bins = np.arange(0., 6. + d_zdist, d_zdist)


###==================== load some other stuff ===================###
sfdq = sfd.SFDQuery()


###====================== create "storage" ======================###

total_number_counts = {obj: np.zeros(len(K_mids)) for obj in args.objects}
if "galaxies" not in total_number_counts:
    total_number_counts["galaxies"] = np.zeros(len(K_mids))
total_area = {obj: 0. for obj in args.objects}
if "galaxies" not in total_area:
    total_area["galaxies"] = 0.

corr_data = {obj: {} for obj in args.objects}
treecorr_data = {obj: {} for obj in args.objects}

if args.read_corr:
    args.corr = True
    for obj in args.objects:
        corr_data_path = paths.data_path / "analysis/corr_{obj}_{args.optical}.pkl"
        if corr_data_path.exists():
            corr_data_path = paths.data_path / "analysis/corr_{args.optical}.pkl"
            with open(corr_data_path, "rb") as f:
                dat = pickle.load(f)
                corr_data[obj] = dat    

###====================== create DataFrames =====================###

counts_df_lookup = {"galaxies": pd.DataFrame({"K": K_mids})}
for obj in args.objects:
    counts_df_lookup[obj] =  pd.DataFrame({"K": K_mids})

###=================== load some plotting data ==================###

plotting_data_dir = paths.input_data_path / "plotting"

kim11_number_counts_path = plotting_data_dir / "kim11_number_counts.csv"
kim14_number_counts_path = plotting_data_dir / "kim14_number_counts.csv"
arcilaosejo19_number_counts_path = plotting_data_dir / "arcilaosejo19_number_counts.csv"
mccracken10_number_counts_path = plotting_data_dir / "mccracken10_number_counts.csv"
kajisawa06_number_counts_path = plotting_data_dir / "kajisawa06_number_counts.csv"
davies21_number_counts_path = plotting_data_dir / "davies21_number_counts.csv"

kim14_zdist_path = plotting_data_dir / "kim14_zdist.csv"
grazian07_zdist_path = plotting_data_dir / "grazian07_zdist.csv"

hartley08_lf_path = plotting_data_dir / "hartley2008_BzK_lf.csv"

kim11_drgs_corr_path = plotting_data_dir / "kim11_drg_eros_K_lt_188.csv"
kim14_eros_corr_path = plotting_data_dir / "jwk_2pcf_ps_eros_245.csv"


kim11_dat = pd.read_csv(kim11_number_counts_path)
kim14_dat = pd.read_csv(kim14_number_counts_path)
arcilaosejo19_dat = pd.read_csv(arcilaosejo19_number_counts_path)
mccracken10_dat = pd.read_csv(mccracken10_number_counts_path)
kajisawa06_dat = pd.read_csv(kajisawa06_number_counts_path)
davies21_dat = pd.read_csv(davies21_number_counts_path)
hartley08_dat = pd.read_csv(hartley08_lf_path)

###================== initalize plots ==================###

counts_plot_lookup = {}
counts_fields_lines_lookup = {obj: [] for obj in args.objects}
counts_main_lines_lookup = {obj: [] for obj in args.objects}
color_plots_lookup = {}
lf_plot_lookup = {}
corr_plot_lookup = {}
zdist_plot_lookup = {}

obj_ids_lookup = {obj: [] for obj in args.objects}

literature_galaxy_counts = [
    (kim14_dat["Kmag"].values, kim14_dat["galaxies"].values, 
        {"s": 20, "marker": "o", "color": "k", "label": "Kim+2014, galaxies"}
    ),
    (mccracken10_dat["Kmag"].values, mccracken10_dat["galaxies"].values, 
        {"s":20, "marker": "P", "color": "k", "label": "McCracken+2010, galaxies"}
    ),
    (davies21_dat["Kmag"].values, davies21_dat["galaxies"].values,
        {"s":20, "marker": "s", "color": "k", "label": "Davies+2021, galaxies"}
    ),
]
literature_counts = {"galaxies": []}
literature_counts["eros_245"] = [
    (kim14_dat["Kmag"].values, kim14_dat["ero245_ps"].values,
        {"s": 20, "marker": "^", "color": "k", "label": "Kim+2014, EROs (PS1)"}
    ),
]
literature_counts["eros_295"] = [
    (kim14_dat["Kmag"].values, kim14_dat["ero295_ps"].values,
        {"s": 20, "marker": "^", "color": "k", "label": "Kim+2014, EROs (PS1)"}
    ),
]
literature_counts["drgs"] = [
    (kajisawa06_dat["Kmag"].values, kajisawa06_dat["drg"].values, 
        {"s": 20, "marker": "^", "color": "k", "label": "Kajisawa+2006, DRGs"}
    ),
    (kim11_dat["Kmag"].values, kim11_dat["drg"].values, 
        {"s": 20, "marker": "x", "color" :"k", "label": "Kim+2011, DRGs"}
    ),
]
literature_counts["sf_gzKs"] = [
    (arcilaosejo19_dat["Kmag"].values, arcilaosejo19_dat["sf_gzK"].values, 
        {"s": 20, "marker": "^", "color": "k", "label": "Arcila-Osejo+2019, SF gzK"}
    ),
    (mccracken10_dat["Kmag"].values, mccracken10_dat["sf_BzK"].values, 
        {"s": 20, "marker": "x", "color": "k", "label": "McCracken+2010, SF BzK"}
    ),
]
literature_counts["pe_gzKs"] = [
    (arcilaosejo19_dat["Kmag"].values, arcilaosejo19_dat["pe_gzK"].values, 
        {"s": 20, "marker": "^", "color": "k", "label": "Arcila-Osejo+2019, PE gzK"}
    ),
    (mccracken10_dat["Kmag"].values, mccracken10_dat["pe_BzK"].values, 
        {"s": 20, "marker": "x", "color": "k", "label": "McCracken+2010, PE BzK"}
    ),
]

for obj in args.objects:
    obj_lit_counts = literature_counts[obj]
    Nfig, Nax = plt.subplots()        
    pl_dat = literature_counts[obj] + literature_galaxy_counts
    for x, y, kwargs in pl_dat:
        l = Nax.scatter(x, y, **kwargs)
        counts_main_lines_lookup[obj].append(l)
    Nax.set_xlim(16, 24)
    Nax.semilogy()
    counts_plot_lookup[obj] = (Nfig, Nax)

if args.color_plots:
    JK_fig, JK_ax = plt.subplots()
    JK_ax.set_xlim(16.0, 23.0)
    JK_ax.set_ylim(-1.5, 2.5)
    JK_ax.set_xlabel(r"$K$", fontsize=12)
    JK_ax.set_ylabel(r"$J-K$", fontsize=12)

    iK_fig, iK_ax = plt.subplots()
    iK_ax.set_xlim(16.0, 23.0)
    iK_ax.set_ylim(-1.0, 5.0)
    iK_ax.set_xlabel(r"$K$", fontsize=12)
    iK_ax.set_ylabel(r"$i-K$", fontsize=12)

    gzK_fig, gzK_ax = plt.subplots()
    gzK_ax.set_xlim(-1.0, 5.5)
    gzK_ax.set_ylim(-1.5, 3.0)
    gzK_ax.set_xlabel(r"$g-z$", fontsize=12)
    gzK_ax.set_ylabel(r"$z-K$", fontsize=12)

if args.lf:
    lf_fig, lf_axes = plt.subplots(2, 4, figsize=(10, 5))
    lf_fig.subplots_adjust(wspace=0., hspace=0.)
    lf_axes = lf_axes.flatten()
    gal_lf_config = lf_config["galaxies"]
    z_mids = 0.5 * (np.array(gal_lf_config["z_bins_low"]) + np.array(gal_lf_config["z_bins_high"]))
    for kk, lf_ax in enumerate(lf_axes):
        phi_Cirasulo = tools.phi_Cirasulo(z_mids[kk], absM_grid)
        lf_ax.semilogy()
        lf_ax.plot(absM_grid, phi_Cirasulo)
        lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
        lf_ax.set_ylim(1e-7, 1e-2)
        if kk % 4 != 0:
            lf_ax.set_yticks([])
        if kk < 4:
            lf_ax.set_xticks([])

    if "eros_245" in args.objects:
        eros_245_lf_fig, eros_245_lf_ax = plt.subplots()
        eros_245_lf_fig.suptitle(f"eros_245, zphot {args.zphot}")
        eros_245_lf_ax.semilogy()
        eros_245_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
        eros_245_lf_ax.set_ylim(5e-8, 5e-3)
        lf_plot_lookup["eros_245"] = (eros_245_lf_fig, eros_245_lf_ax)

    if "eros_295" in args.objects:
        eros_295_lf_fig, eros_295_lf_ax = plt.subplots()
        eros_295_lf_fig.suptitle(f"eros_295, zphot {args.zphot}")
        eros_295_lf_ax.semilogy()
        eros_295_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
        eros_295_lf_ax.set_ylim(5e-8, 5e-3)
        lf_plot_lookup["eros_295"] = (eros_295_lf_fig, eros_295_lf_ax)

    if "drgs" in args.objects:
        drgs_lf_fig, drgs_lf_ax = plt.subplots()
        drgs_lf_fig.suptitle(f"drgs, zphot {args.zphot}")
        drgs_lf_ax.semilogy()
        drgs_lf_ax.set_xlim(absM_bins[-1], absM_bins[0])
        lf_plot_lookup["drgs"] = (drgs_lf_fig, drgs_lf_ax)
    
    if "sf_gzKs" in args.objects or "pe_gzKs" in args.objects:
        gzK_lf_fig, gzK_lf_ax = plt.subplots()
        gzK_lf_fig.suptitle(f"gzKs, zphot {args.zphot}")
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


### load some zdist plotting data.
kim14_zdist_dat = pd.read_csv(kim14_zdist_path)
grazian07_zdist_dat = pd.read_csv(grazian07_zdist_path)

literature_zdist = {"galaxies": []}
literature_zdist["eros_245"] = [
    (kim14_zdist_dat["z"].values, kim14_zdist_dat["eros_245"].values, 
        {"where":"mid", "color":"k", "lw":"5", "label":"Kim+2014"}
    ),
]
literature_zdist["eros_295"] = [
    (kim14_zdist_dat["z"].values, kim14_zdist_dat["eros_295"].values, 
        {"where":"mid", "color":"k", "lw":"5", "label":"Kim+2014"}
    ),
]
literature_zdist["drgs"] = [
    (grazian07_zdist_dat["z"].values, grazian07_zdist_dat["DRG"].values, 
        {"where":"mid", "color":"k", "lw":"5", "label":"Kim+2014"}
    ),
]
literature_zdist["sf_gzKs"] = [
    (grazian07_zdist_dat["z"].values, grazian07_zdist_dat["sBzK"].values, 
        {"where":"mid", "color":"k", "lw":"5", "label":"Kim+2014"}
    ),
]
literature_zdist["pe_gzKs"] = [
    (grazian07_zdist_dat["z"].values, grazian07_zdist_dat["pBzK"].values, 
        {"where":"mid", "color":"k", "lw":"5", "label":"Kim+2014"}
    ),
]

zdist_plot_config = {
    "eros_245": {"xlim": (0.5, 2.0)},
    "eros_295": {"xlim": (0.5, 2.0)},
    "drgs": {"xlim": (0.5, 4.0)},
    "sf_gzKs": {"xlim": (0.5, 4.0)},
    "pe_gzKs": {"xlim": (0.5, 4.0)},
}

if args.zphot is not None:
    for obj in args.objects:
        zdist_fig, zdist_ax = plt.subplots()
        zdist_fig.suptitle(f"{object_names[obj]} photo-z dist")
        pl_data = literature_zdist[obj]
        xlim = zdist_plot_config.get(obj, {}).get("xlim", (0.5, 3.0))
        zdist_ax.set_xlim(xlim)
        for x, y, kwargs in pl_data:
            zdist_ax.step(x, y, **kwargs)
        zdist_plot_lookup[obj] = (zdist_fig, zdist_ax)


### load some corr plotting data.
kim14_corr_dat = pd.read_csv(kim14_eros_corr_path, names=["x", "w"])

literature_corr = {"galaxies": []}

literature_corr["eros_245"] = [
    (kim14_corr_dat["x"].values, kim14_corr_dat["w"].values,
        {"s":10, "color":"k", "label":"Kim+2014"}
    ),
]
literature_corr["eros_295"] = []
literature_corr["drgs"] = []
literature_corr["sf_gzKs"] = []
literature_corr["pe_gzKs"] = []

corr_plot_config = {}

if args.corr:
    for obj in args.objects:
        corr_fig, corr_ax = plt.subplots()
        pl_data = literature_corr[obj]
        for x, y, kwargs in pl_data:
            corr_ax.scatter(x, y, **kwargs)
        xlim = corr_plot_config.get(obj, {}).get("xlim", (5e-4, 5e0))
        ylim = corr_plot_config.get(obj, {}).get("ylim", (1e-3, 1e1))

        corr_ax.set_xlim(xlim)
        corr_ax.set_ylim(ylim)
        corr_ax.loglog()
        corr_plot_lookup[obj] = (corr_fig, corr_ax)


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
    if args.mir == "swire":
        for band in "I1 I2".split():
            mask_lookup[band] = (
                paths.input_data_path / f"external/swire/masks/{field}_{band}_mask.fits"
            )

    JK_mask_list  = [mask_lookup[b] for b in "J K".split()]
    iJK_mask_list = [mask_lookup[b] for b in "i J K".split()]
    gzK_mask_list = [mask_lookup[b] for b in "g z J K".split()]
    if args.mir == "swire":
        iJKI1_mask_list = [mask_lookup[b] for b in "i J K I1".split()]
        JKI1_mask_list = [mask_lookup[b] for b in "J K I1".split()]
    
    for mask_list in [iJK_mask_list, gzK_mask_list, JK_mask_list]:
        for p in mask_list:
            if not p.exists():
                raise IOError(f"no mask {p.relative_to(paths.base_path)}")

    area_density = counts_config["area_density"]
    JK_area = calc_survey_area(JK_mask_list)#, density=area_density)
    iJK_area = calc_survey_area(iJK_mask_list)#, density=area_density)
    gzK_area = calc_survey_area(gzK_mask_list)#, density=area_density)
    print(f"JK_area: {JK_area:.2f}, iJK_area: {iJK_area:.2f}, gzK_area: {gzK_area:.2f}")
    if args.mir == "swire":        
        JKI1_area = calc_survey_area(giJKI1_mask_list)#, density=area_density)
        iJKI1_area = calc_survey_area(iJKI1_mask_list)#, density=area_density)
        print("JK_I1 area: {JKI1_area:.2f}, iJK_I1 area: {iJK_I1}")

    area_lookup = {
        "galaxies": JK_area, "drgs": JK_area, 
        "eros_245": iJK_area, "eros_295": iJK_area, 
        "sf_gzKs": gzK_area, "pe_gzKs": gzK_area,            
    }

    mask_list_lookup = {
        "galaxies": JK_mask_list, "drgs": JK_mask_list, 
        "eros_245": iJK_mask_list, "eros_295": iJK_mask_list, 
        "sf_gzKs": gzK_mask_list, "pe_gzKs": gzK_mask_list,
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

    #for band, col in nir_mags.items():
    #    if col not in catalog.columns:
    #        continue               
    #    catalog[col] = vega_to_ab(catalog[col], band=band)
    #catalog[Ktot_mag] = vega_to_ab(catalog[Ktot_mag], band="K")


    if W1mag in catalog.columns:
        logger.info("reject stars with WISE criteria.")
        star_q1 = Query(
            f"({imag}-{Kmag}) < 0.76 * ({gmag} - {imag}) - 0.85", 
            f"{gmag} < 30.0", f"{imag} < 30.0", f"{Kmag} < 30.0"
        )

        star_q2 = Query(
            f"{rmag} - {W1mag} < 2.29 * ({rmag} - {imag}) - 0.66",
            f"{rmag} < 30.0", f"{imag} < 30.0", f"{W1mag} < 30.0"
        )

        star_q3 = Query(f"{Jmag} - {Kmag} < 0.04", f"{Ktot_mag} < 19.")

        galaxy_query = ~((star_q1 & star_q2) | star_q3)
        galaxies = galaxy_query.filter(catalog)
        
    else:
        logger.info("reject stars with J-K criteria.")
        galaxies = Query(
            f"({Jmag} - 0.938) - ({Kmag} - 1.900) > 1.0",
        ).filter(catalog)

    
    if args.color_plots:
        lc = len(catalog)
        lcd = int(lc * args.cp_density)
        inds = np.linspace(0, lc, lcd)[:-1].astype(int)
        ccat = catalog[inds]
        K_dat = ccat[Ktot_mag]
        iK_dat = ccat[imag]-ccat[Kmag]
        JK_dat = ccat[Jmag]-ccat[Kmag]
        gz_dat = ccat[gmag]-ccat[zmag]
        zK_dat = ccat[zmag]-ccat[Kmag]
        iK_ax.scatter(K_dat, iK_dat, marker=".", s=1, color=f"C{ii}")#, alpha=0.2)
        JK_ax.scatter(K_dat, JK_dat, marker=".", s=1, color=f"C{ii}")#, alpha=0.2)
        gzK_ax.scatter(gz_dat, zK_dat, marker=".", s=1, color=f"C{ii}")#, alpha=0.2)


    ## Number counts of galaxies.
    gal_hist, _ = np.histogram(galaxies[Ktot_mag], bins=K_bins)
    gal_hist_norm = gal_hist / JK_area
    gal_hist_err = np.sqrt(gal_hist) / JK_area
    counts_df_lookup["galaxies"][field] = gal_hist_norm
    
    total_number_counts["galaxies"] = total_number_counts["galaxies"] + gal_hist
    total_area["galaxies"] = total_area["galaxies"] + JK_area

    if args.lf:
        #catalog = Query(
            #f"0.1 < z_phot", f"z_phot < 2.5",
            #f"0.1 < z_phot_chi2 / nusefilt", 
            #f"z_phot_chi2 / nusefilt < 20.0",
        #).filter(catalog)

        if "DISTMOD" in catalog.columns:
            distmod = catalog["DISTMOD"]
            z_phot = None
        else:
            distmod = None
            z_phot = catalog["z_phot"]

        absM = tools.calc_absM(
            catalog[Ktot_mag], distmod=distmod, z_phot=z_phot, cosmol=cosmol
        )
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
                M_bright = M_bright_arr[kk] - 1.0
                M_faint = M_faint_arr[kk] - 1.0
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
        if obj == "galaxies":
            continue

        ### ================= OBJECT NUMBER COUNTS ==================== ###
        Nfig, Nax = counts_plot_lookup[obj]   
        color_query = selection_queries[obj]
        selection_area = area_lookup[obj]
        selection_mask_list = mask_list_lookup[obj]

        
        if obj in select_from_galaxies:
            print(f"\nselect {obj} from GALAXIES with:\n {color_query}")
            selection = Query(*color_query).filter(galaxies)
        elif obj in select_from_catalog:
            print(f"\nselect {obj} from CATALOG with:\n {color_query}")
            selection = Query(*color_query).filter(catalog)
        else:
            raise ValueError(
                "don't know whether to select from GALAXIES or CATALOG (add to line 60/61)"
            )

        # Keep only the objects which are in the "good" parts of the data.
        selection_mask = objects_in_coverage(
            selection_mask_list, selection["ra"], selection["dec"]
        )
        selection = selection[ selection_mask ]
           
        Nhist, _ = np.histogram(selection[Ktot_mag], bins=K_bins)
        total_number_counts[obj] = total_number_counts[obj] + Nhist
        total_area[obj] = total_area[obj] + selection_area

        #ax.scatter(selection["ra"], selection["dec"], s=1)
        #ax2.scatter(
        #    selection[Ktot_mag], 
        #    selection[imag] - selection[Kmag],# - selection["K_mag_aper_30"],
        #    s=1
        #)

        Nhist_norm = Nhist / selection_area

        Nhist_err = np.sqrt(Nhist) / selection_area

        field_name = survey_config["code_to_field"][field]
        line = Nax.errorbar(K_mids, Nhist_norm, yerr=Nhist_err, color=f"C{ii}", label=field_name, alpha=0.8)
        counts_fields_lines_lookup[obj].append(line)
        Nax.errorbar(K_mids, gal_hist_norm, yerr=gal_hist_err, color=f"C{ii}", ls="--")

        counts_df_lookup[obj][field] = Nhist_norm

        if args.zphot is not None:
            obj_zdist_fig, obj_zdist_ax = zdist_plot_lookup[obj]
            zhist,_ = np.histogram(selection["z_phot"], bins=zdist_bins)
            zdist = zhist / (selection_area * d_zdist)
            obj_zdist_ax.fill_between(
                zdist_bins[:-1], zdist, 
                step="pre", edgecolor=f"C{ii}", facecolor="none", #hatch=hatch_styles[ii]
            )
        
        ### =============== OBJECT LUMINOSITY FUNCTIONS =============== ###
        if args.lf:
            obj_lf_fig, obj_lf_axes = lf_plot_lookup[obj]

            obj_lf_config = lf_config[obj]

            z_low_arr = obj_lf_config["z_bins_low"]
            z_high_arr = obj_lf_config["z_bins_high"]
            M_bright_arr = obj_lf_config["absM_bright"]
            M_faint_arr = obj_lf_config["absM_faint"]

            Mz_fig, Mz_ax = plt.subplots()
            Mz_fig.suptitle(f"{obj}, zphot {args.zphot}")
            Mz_ax.scatter(catalog["z_phot"], catalog["absM"], color="k", s=1, alpha=0.1)
            Mz_ax.scatter(selection["z_phot"], selection["absM"], color="r", s=1, alpha=0.6)
            Mz_ax.set_ylim(-16., -30.)
            Mz_ax.set_xlim(-1.1, 6.1)


            for kk, (z_low, z_high) in enumerate(zip(z_low_arr, z_high_arr)):
                z_mid = 0.5 * (z_low + z_high)
                
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
                z_max = np.minimum(obj_z_cat["z_max"], z_high)
                z_min = np.maximum(obj_z_cat["z_min"], z_low)
                vmax = tools.calc_vmax(z_min, z_max, selection_area * (np.pi / 180.) ** 2, cosmol)
                phi, phi_err = tools.calc_luminosity_function(obj_z_cat["absM"], vmax, absM_bins)
                phi = phi / d_absM

                label = f"{z_low} < z < {z_high}" if ii == 0 else None

                lf_ax = obj_lf_axes#[kk]
                lf_ax.errorbar(absM_mids, phi, yerr=phi_err, color=f"C{kk}", label=label)
            lf_ax.legend()
            #Mz_fig.savefig(f"~/{obj}_{args.zphot}_B.png")
        
        ### ============== OBJECT CORRELATION FUNCTIONS =============== ###
        if args.corr:
            obj_corr_fig, obj_corr_ax = corr_plot_lookup[obj]
            obj_corr_config = corr_config[obj]
            
            # where should we put the randoms?
            obj_mask_list = mask_list_lookup[obj]
            ra_lim = (np.min(selection["ra"]), np.max(selection["ra"]))
            dec_lim = (np.min(selection["dec"]), np.max(selection["dec"]))

            # generate some randoms
            randoms = tools.get_randoms(
                ra_lim, dec_lim, obj_mask_list, randoms_density=randoms_density
            )

            if patch_size is not None:
                patch_ratio = selection_area / patch_size
                N_patches = int(round(patch_ratio, 0))
                print(f"N_patches={N_patches} (rounded from {patch_ratio:.2f})")
            else:
                N_patches = 1

            # only really need to process the randoms once...
            print("##======= process RR")
            rr_cat, rr = tools.prepare_auto_component(
                treecorr_config, ra=randoms.ra, dec=randoms.dec, npatch=N_patches#, cat_only=True
            )
            print(rr_cat.patch_centers)

            #fig, ax = plt.subplots()
            #ax.scatter(rr_cat.ra, rr_cat.dec, c=rr_cat.patch % 10, s=1, cmap="tab10")
            if "rr_cat" not in corr_data[obj]:
                corr_data[obj]["rr_cat"] = {}
            corr_data[obj]["rr_cat"][field] = rr_cat
            
            for kk, K_lim in enumerate(obj_corr_config["K_limits"]):
                corr_selection = Query(f"{Ktot_mag} < {K_lim}").filter(selection)
                print(f"\ncorr for {len(corr_selection)} {obj} with K_tot < {K_lim}")

                obj_pos_fig, obj_pos_ax = plt.subplots()
                obj_pos_fig.suptitle(f"{field} {obj} {K_lim}")
                obj_pos_ax.scatter(randoms.ra, randoms.dec, s=1, color="k")
                obj_pos_ax.scatter(corr_selection["ra"], corr_selection["dec"], s=1, color="r")

                ### prepare components
                print("##======= process DD")
                dd_cat, dd = tools.prepare_auto_component(
                    treecorr_config, ra=corr_selection["ra"], dec=corr_selection["dec"],
                    patch_centers=rr_cat.patch_centers
                )
                print("##======= process DR")
                _, _, dr = tools.prepare_cross_component(
                    treecorr_config,
                    cat1=dd_cat, cat2=rr_cat, patch_centers=rr_cat.patch_centers
                )
                rd = None

                ### calculate the correlation function!
                x = dd.rnom / 3600.
                w, w_var = dd.calculateXi(dr=dr, rd=rd, rr=rr)
                w_err = np.sqrt(w_var)

                obj_corr_ax.errorbar(x, w, yerr=w_err, color=f"C{ii}") # PLOT!

                ### collect some data to save.
                corr_data_kk = {
                    "x": x, "w": w, "w_var": w_var, "w_err": w_err, # key numbers
                    "Nd": len(corr_selection),  #number of obj
                    "Nr": len(randoms),
                    "dd": dd.npairs, 
                    "rr": rr.npairs,
                    "dr": dr.npairs if dr is not None else None, 
                    "rd": rd.npairs if rd is not None else None,
                    "NN_dd": dd, "NN_dr": dr, "NN_rd": rd, "NN_rr": rr, # actual objects.
                }
                if kk not in corr_data[obj]:
                    corr_data[obj][kk] = {}
                corr_data[obj][kk][field] = corr_data_kk
    
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
            
        #obj_lf_config = lf_config["uvj"]
        plt.colorbar(scatter, cax=uvj_axes[-1])

### combined info plotting.

total_gal_hist = total_number_counts["galaxies"]
total_gal_area = total_area["galaxies"]

total_gal_hist_norm = total_gal_hist / total_gal_area
total_gal_hist_err = np.sqrt(total_gal_hist) / total_gal_area

counts_df_lookup["galaxies"].insert(
    loc=1, column="combined", value=total_gal_hist_norm
)

for obj in args.objects:
    Nfig, Nax = counts_plot_lookup[obj]
    print(obj)
    selection_area = total_area[obj]

    Nhist = total_number_counts[obj]
    
    Nhist_norm = Nhist / selection_area
    Nhist_err = np.sqrt(Nhist) / selection_area

    counts_df_lookup[obj].insert(
        loc=1, column="combined", value=Nhist_norm
    )

    obj_label = object_names[obj]
    obj_l = Nax.errorbar(
        K_mids, Nhist_norm, yerr=Nhist_err, label=obj_label, 
        color="k"
    )
    gal_l = Nax.errorbar(
        K_mids, total_gal_hist_norm, yerr=total_gal_hist_err, label="galaxies", 
        color="k", ls="--"
    ) #, label="comb.", color="k")
    counts_main_lines_lookup[obj].extend([obj_l, gal_l])
    l = Nax.errorbar([17,17], [0,0], yerr=[0,0], label="combined", color="k")
    counts_fields_lines_lookup[obj].append(l)
    
    fields_legend = Nax.legend(handles=counts_fields_lines_lookup[obj], loc=2)
    Nax.add_artist(fields_legend)

    main_lines_legend = Nax.legend(handles=counts_main_lines_lookup[obj], loc=4)
    Nax.add_artist(main_lines_legend)

    #Nax.legend()
    Nax.set_ylabel(r"N/deg$^{2}$/0.5 mag", fontsize=14)
    Nax.set_xlabel(r"$K_{AB}$", fontsize=14)
    
    if args.corr:
        obj_corr_fig, obj_corr_ax = corr_plot_lookup[obj]
        obj_corr_data = corr_data[obj]
        obj_corr_config = corr_config[obj]
        for kk, obj_corr_kk in obj_corr_data.items():
            print(kk)
            if isinstance(kk, str):
                print("skipping")
                continue
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
            w_var = w_err ** 2

            #corrs = [d["NN_dd"] for field, d in obj_corr_kk.items()]
            #w_cov = estimate_multi_cov(corrs, method="jackknife")

            #print(w_cov)

            #fig, ax = plt.subplots()
            #ax.imshow(w_cov)
            #plt.show()
            #w_var = w_cov.diagonal()
            #w_err = np.sqrt(w_var)
        
            obj_corr_ax.errorbar(x, w_tot, yerr=w_err, color="k")

            combined_data = {
                "x": x, "w": w_tot, "w_var": w_var, "w_err": w_err, 
                "Nd": Nd, "Nr": Nr, 
                "dd": total_dd, "dr": total_dr, "rd": None, "rr": total_rr, 
                
            }
            corr_data[obj][kk]["combined"] = combined_data
        
            

###==============write out some stuff================###

analysis_dir = paths.data_path / "analysis"
analysis_dir.mkdir(exist_ok=True, parents=True)
for obj, df in counts_df_lookup.items():
    output_path = analysis_dir / f"counts_{obj}_{args.optical}.csv"
    if args.force_write or (not output_path.exists()):
        df.to_csv(output_path, index=False, float_format="%.2f")


for obj, dat in corr_data.items():
    try:
        dth = "_" + Path(args.treecorr_config).stem.split("_")[-1]
    except:
        dth = ""
    corr_data_path = analysis_dir / f"corr_{obj}_{args.optical}{dth}.pkl"
    if corr_data_path.exists() is False and args.corr:
        print(f"writing {corr_data_path}")
        with open(corr_data_path, "wb+") as f:
            pickle.dump(dat, f)

"""
for obj, dat in treecorr_data.items():
    print("dat")
    treecorr_data_path = analysis_dir / f"treecorr_{obj}_{args.optical}.pkl"
    if treecorr_data_path.exists() is False and args.corr:
        print(f"write to {treecorr_data_path}")
        with open(treecorr_data_path, "wb+") as f:
            pickle.dump(dat, f)"""

if args.save_figs:
    for obj, (fig, ax) in counts_plot_lookup.items():
        outpath = analysis_dir / f"counts_{obj}_{args.optical}.pdf"
        fig.tight_layout()
        fig.savefig(outpath, dpi=400)

"""
if args.save_figs:
    for obj, (fig, ax) in zdist_plot_lookup.items():
        fig.savefig(f"/cosma/home/durham/dc-sedg2/{obj}_A_{args.zphot}.png")    

    for obj, (fig, ax) in lf_plot_lookup.items():
        fig.savefig(f"/cosma/home/durham/dc-sedg2/{obj}_C_{args.zphot}.png")
"""
if os.isatty(0):
    plt.show()





