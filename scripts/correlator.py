import logging
import pickle
import yaml

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
from dxs.utils.misc import calc_range, print_header
from dxs.utils.phot import ab_to_vega, vega_to_ab
from dxs.utils.region import in_only_one_tile
from dxs import paths

logger = logging.getLogger("correlator")

field = "EN"

nir_mag_type = "aper_20"
nir_tot_mag_type = "auto"
opt_mag_type = "aper_30" # ps use aper_18 or aper_30, cfhtls use aper_2 or aper_3.

gmag = f"g_mag_{opt_mag_type}"
imag = f"i_mag_{opt_mag_type}"
zmag = f"z_mag_{opt_mag_type}"
Jmag = f"J_mag_{nir_mag_type}"
Kmag = f"K_mag_{nir_mag_type}"
Ktot_mag = f"K_mag_{nir_tot_mag_type}"


survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

randoms_density = 10_000

ic = 0.008

###================= start =================###

catalog_path = paths.catalogs_path / f"{field}00/{field}00_panstarrs.fits"
#catalog = Table.read(catalog_path)

N_tiles = survey_config.get("tiles_per_field").get(field, 12)
field_mosaic_list = [paths.get_mosaic_path(field, x, "K") for x in range(1, N_tiles+1)]

jwk_number_counts_path = paths.input_data_path / "plotting/jwk_AB_number_counts.csv"
lao_number_counts_path = paths.input_data_path / "plotting/lao_AB_number_counts.csv"

qp = QuickPlotter.from_fits(
    catalog_path, 
    #"-0.5 < dec", "dec < 1.0", "333.0 < ra", "ra < 335.0", 
    f"{Jmag} < 30.", f"{Kmag} < 30."#, f"{imag} < 30."#, f"{gmag} < 30.", f"{zmag} < 30."
)
qp.full_catalog[imag] = qp.full_catalog[imag].filled(99)

"""
logger.info("starting mask")
med_ra = np.nanmedian(qp.full_catalog["i_ra"])
med_dec = np.nanmedian(qp.full_catalog["i_dec"])
dra = qp.full_catalog["i_ra"] - med_ra
ddec = (qp.full_catalog["i_dec"] - med_dec)
mask = dra*dra + ddec*ddec < 1.0**2

logger.info("made mask")
qp.full_catalog = qp.full_catalog[mask]"""

coords = SkyCoord(
    ra=qp.full_catalog["ra"], dec=qp.full_catalog["dec"], unit="degree"
)
#in_one_tile_mask = in_only_one_tile(field_mosaic_list, coords)
#qp.full_catalog = qp.full_catalog[ in_one_tile_mask ]

ra_limits = calc_range(qp.full_catalog["ra"])
dec_limits = calc_range(qp.full_catalog["dec"])

optical_mask_path = paths.input_data_path / f"external/panstarrs/masks/{field}_mask.fits"
J_mask_path = paths.masks_path / f"{field}_J_good_cov_mask.fits"
K_mask_path = paths.masks_path / f"{field}_K_good_cov_mask.fits"
mask_list = [J_mask_path, K_mask_path, optical_mask_path]

#catalog_mask = objects_in_coverage(mask_list, qp.full_catalog["ra"], qp.full_catalog["dec"])
#qp.full_catalog = qp.full_catalog[ catalog_mask ]

nir_area = calc_survey_area(
    [J_mask_path, K_mask_path], ra_limits=ra_limits, dec_limits=dec_limits
)
opt_area = calc_survey_area(
    [J_mask_path, K_mask_path, optical_mask_path],
    ra_limits=ra_limits, dec_limits=dec_limits
)
qp.remove_crosstalks(catalog=qp.full_catalog)
#qp.catalog[imag] -= 0.39 # fix bc. incorrectly converted to Vega from AB.[

#qp.catalog[gmag] = ab_to_vega(qp.catalog[gmag], band="g")
#qp.catalog[imag] = ab_to_vega(qp.catalog[imag], band="i")
#qp.catalog[zmag] = ab_to_vega(qp.catalog[zmag], band="z")

qp.catalog[Jmag] = vega_to_ab(qp.catalog[Jmag], band="J")
qp.catalog[Kmag] = vega_to_ab(qp.catalog[Kmag], band="K")
qp.catalog[Ktot_mag] = vega_to_ab(qp.catalog[Ktot_mag], band="K")

qp.create_selection("gals", f"{Jmag}-0.91-({Kmag}-1.85) > 1.0")
qp.create_selection(
    "eros",
    f"{imag} - {Kmag} > 2.45",
    f"{Kmag} > 18", 
    f"{imag} < 25.0",
    #f"i_flux_{opt_mag_type} < 10**10",
    catalog="catalog",
)
qp.create_selection(
    "sf_gzK",
    f"({zmag}-{Kmag}) -1.27*({gmag}-{zmag}) >= -0.022",
    f"{zmag} < 50.",
    f"{gmag} < 50.",
    catalog="gals"
)
qp.create_selection(
    "pe_gzK",
    f"({zmag}-{Kmag}) -1.27*({gmag}-{zmag}) < -0.022", 
    f"{zmag}-{Kmag}>2.55",
    f"{zmag} < 50.",
    f"{gmag} < 50.",
    catalog="gals"
)
qp.create_selection(
    "drgs", f"{Jmag}-{Kmag} > 1.3", catalog="gals"
)


full_randoms = SkyCoord(
    uniform_sphere(ra_limits, dec_limits, density=randoms_density), unit="degree"
)
random_mask = objects_in_coverage(mask_list, full_randoms.ra, full_randoms.dec)
#single_tile_mask = in_only_one_tile(field_mosaic_list, full_randoms)
"""
dra = full_randoms.ra.degree - med_ra
ddec = full_randoms.dec.degree - med_dec
circle_mask = dra*dra + ddec*ddec < 1.0**2
"""
randoms = full_randoms[ random_mask ]#& circle_mask ]#& single_tile_mask]

#rra_mask = (randoms.ra > 332.2 * u.degree) & (randoms.ra < 336. * u.degree)
#rdec_mask = (randoms.dec > -1.0 * u.degree) & (randoms.dec < 1.6 * u.degree)
#randoms = randoms[ rra_mask & rdec_mask ]

ero_mask = objects_in_coverage(mask_list, qp.eros["ra"], qp.eros["dec"])
qp.eros = qp.eros[ ero_mask ]

qp.create_plot("iK_plot")
qp.iK_plot.color_magnitude(imag, Kmag, Kmag, selection=qp.catalog, s=1, alpha=0.1)
qp.iK_plot.color_magnitude(imag, Kmag, Kmag, selection=qp.eros, s=1, alpha=0.1)

qp.create_plot("JK_plot")
qp.JK_plot.color_magnitude(Jmag, Kmag, Kmag, selection=qp.catalog, s=1, alpha=0.1)
qp.JK_plot.color_magnitude(Jmag, Kmag, Kmag, selection=qp.drgs, s=1, alpha=0.1)

qp.create_plot("gzK_plot")
qp.gzK_plot.color_color(zmag, Kmag, gmag, zmag, selection=qp.catalog, s=1, color="C0", alpha=0.1)
qp.gzK_plot.color_color(zmag, Kmag, gmag, zmag, selection=qp.sf_gzK, s=1, color="C1", alpha=0.5)
qp.gzK_plot.color_color(zmag, Kmag, gmag, zmag, selection=qp.pe_gzK, s=1, color="C2", alpha=0.5)

dm = 0.5
min_mag, max_mag = 16.25, 23.0
bins = np.arange(min_mag - 0.5*dm, max_mag + 1.5*dm, dm)



qp.create_plot("num_density")
qp.num_density.plot_number_density(
    Ktot_mag, selection=qp.gals, bins=bins, survey_area=nir_area, label="galaxies, J-K>1.0",
)
qp.num_density.plot_number_density(
    Ktot_mag, selection=qp.eros, bins=bins, survey_area=opt_area, label="eros, i-K > 4.5",
)
#qp.num_density.plot_number_density(
#    Ktot_mag, selection=qp.drgs, bins=bins, survey_area=nir_area, label="drgs, J-K > 2.3",
#)
#qp.num_density.axes.scatter(jwk["K"], jwk["drgs"], marker="^", color="k", label="jwk drg")

jwk = pd.read_csv(jwk_number_counts_path, delim_whitespace=True, na_values=["-"])
qp.num_density.axes.scatter(
    jwk["Kmag"].values, jwk["galaxies"].values, marker="o", color="k", label="jwk gal"
)
qp.num_density.axes.scatter(
    jwk["Kmag"].values, jwk["ero245_ps"].values, marker="x", color="k", label="jwk ero295"
)
qp.num_density.axes.semilogy()
qp.num_density.axes.legend()

qp.create_plot("gzK_density")
lao = pd.read_csv(lao_number_counts_path)

qp.gzK_density.plot_number_density(
    Ktot_mag, selection=qp.pe_gzK, bins=bins, survey_area=opt_area, label="pe gzK",
)
qp.gzK_density.plot_number_density(
    Ktot_mag, selection=qp.sf_gzK, bins=bins, survey_area=opt_area, label="sf gzK",
)
qp.gzK_density.axes.scatter(
    lao["Kmag"].values, lao["sf_gzK"].values, marker="^", color="k", label="lao sf gzK"
)
qp.gzK_density.axes.scatter(
    lao["Kmag"].values, lao["pe_gzK"].values, marker="s", color="k", label="lao pe gzK"
)

qp.gzK_density.axes.semilogy()
qp.gzK_density.axes.legend()


qp.create_selection("to_correlate", f"{Ktot_mag} < 20.7", catalog="eros")

qp.create_plot("coords_plot")
qp.coords_plot.plot_positions(randoms.ra, randoms.dec, s=1)
qp.coords_plot.plot_coordinates("ra", "dec", selection=qp.to_correlate, s=1)

#plt.show()
#plt.close()


treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"
with open(treecorr_config_path, "r") as f:
    treecorr_config = yaml.load(f, Loader=yaml.FullLoader)

data_catalog = Catalog(
    ra=qp.to_correlate["ra"], 
    dec=qp.to_correlate["dec"], 
    ra_units="deg", dec_units="deg",
    npatch=treecorr_config.get("npatch", 25),
)

random_catalog = Catalog(
    ra=randoms.ra, 
    dec=randoms.dec, 
    ra_units="deg", dec_units="deg",
    patch_centers=data_catalog.patch_centers,
)

treecorr_config["num_threads"] = 3

dd = NNCorrelation(config=treecorr_config)
rr = NNCorrelation(config=treecorr_config)
dr = NNCorrelation(config=treecorr_config)


print(np.log10(dd.rnom[1:]/dd.rnom[:-1]))
dd.process(data_catalog)
rr.process(random_catalog)
dr.process(data_catalog, random_catalog)


jwk_path = paths.input_data_path / "plotting/jwk_2pcf_ps_eros_245.csv"
jwk = pd.read_csv(jwk_path, names=["x", "w"])
print(jwk)

#xi_naive, varxi_naive = dd.calculateXi(rr=rr) #, dr=dr)
xi_ls, varxi_ls = dd.calculateXi(rr=rr, dr=dr)

dd.write("./test_out.csv", rr=rr, dr=dr)

print("corr")
print(xi_ls)

print(np.log10(dd.rnom[1:]/dd.rnom[:-1]))

fig, ax = plt.subplots()
ax.plot(dd.rnom/3600., dd.npairs, label="dd counts")
ax.plot(dr.rnom/3600., dr.npairs, label="dr counts")
ax.plot(rr.rnom/3600., rr.npairs, label="rr counts")
ax.legend()
ax.loglog()

lc1 = len(qp.to_correlate["ra"])
lc2 = len(randoms.ra)

ndd = dd.npairs/(0.5*lc1*(lc1-1))
ndr = dr.npairs/(lc1*lc2)
nrr = rr.npairs/(0.5*lc2*(lc2-1))

fig, ax = plt.subplots()
ax.plot(dd.rnom/3600., ndd, label="<dd>")
ax.plot(dr.rnom/3600., ndr, label="<dr>")
ax.plot(rr.rnom/3600., nrr, label="<rr>")
ax.legend()
ax.loglog()

corr_data = {
    "x": dd.rnom/3600.,
    "dd": dd.npairs,
    "dr": dr.npairs,
    "rr": rr.npairs,
    "xi_ls": xi_ls,
    "var_xi": varxi_ls,
    "n_data": lc1,
    "n_random": lc2,
}

pkl_path = f"./{field}_corr_data.pkl"
with open(pkl_path, "wb+") as f:
    pickle.dump(corr_data, f)

corr_df = pd.DataFrame()



print("X")
print(dd.rnom/3600.)

fig, ax = plt.subplots()
#ax.plot(dd.rnom/3600., xi_naive, color="C0", marker="x", label="naive")
#ax.plot(dd.rnom/3600., xi_naive + ic, color="C0", marker="x", ls="--", label="naive + ic")
ax.errorbar(dd.rnom/3600., xi_ls, yerr=varxi_ls, color="C1", marker="^", label="LS est.")
ax.plot(dd.rnom/3600., xi_ls + ic, color="C1", marker="^", ls="--", label="LS est. + ic")
ax.scatter(jwk["x"], jwk["w"], s=10, color="k", label="Kim+ 2011")
ax.legend()
ax.loglog()
#plt.show()


























