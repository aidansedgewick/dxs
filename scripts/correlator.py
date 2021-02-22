import logging
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
from dxs.utils.misc import calc_range
from dxs.utils.phot import ab_to_vega, vega_to_ab
from dxs.utils.region import in_only_one_tile
from dxs import paths



logger = logging.getLogger("main")

field = "SA"

nir_mag_type = "aper_30"
nir_tot_mag_type = "auto"
opt_mag_type = "aper_3"

gmag = f"G_mag_{opt_mag_type}"
imag = f"I_mag_{opt_mag_type}"
zmag = f"Z_mag_{opt_mag_type}"
Jmag = f"J_mag_{nir_mag_type}"
Kmag = f"K_mag_{nir_mag_type}"
Ktot_mag = f"K_mag_{nir_tot_mag_type}"


survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

randoms_density = 10_000

ic = 0.02



###================= start =================###

catalog_path = paths.catalogs_path / f"{field}00/m{field}00_cfhtls.fits"
#catalog = Table.read(catalog_path)

N_tiles = survey_config.get("tiles_per_field").get(field, 12)
field_mosaic_list = [paths.get_mosaic_path(field, x, "K") for x in range(1, N_tiles+1)]

jwk_number_counts_path = paths.input_data_path / "plotting/jwk_AB_number_counts.csv"
lao_number_counts_path = paths.input_data_path / "plotting/lao_AB_number_counts.csv"

qp = QuickPlotter.from_fits(
    catalog_path, 
    #"-1.0 < dec", "dec < 1.6", "332.2 < ra", "ra < 333.9", 
    f"{Jmag} < 30.", f"{Kmag} < 30.", f"{imag} < 30.", f"{gmag} < 30.", f"{zmag} < 30."
)
qp.full_catalog[imag] = qp.full_catalog[imag].filled(99)

coords = SkyCoord(
    ra=qp.full_catalog["ra"], dec=qp.full_catalog["dec"], unit="degree"
)
#in_one_tile_mask = in_only_one_tile(field_mosaic_list, coords)
#qp.full_catalog = qp.full_catalog[ in_one_tile_mask ]

ra_limits = calc_range(qp.full_catalog["ra"])
dec_limits = calc_range(qp.full_catalog["dec"])

optical_mask_path = paths.input_data_path / f"external/panstarrs/masks/{field}_mask.fits"
J_mask_path = paths.masks_path / f"{field}_J_good_cov_stars_mask.fits"
K_mask_path = paths.masks_path / f"{field}_K_good_cov_stars_mask.fits"
mask_list = [J_mask_path, K_mask_path]#, optical_mask_path]

#catalog_mask = objects_in_coverage(mask_list, qp.full_catalog["ra"], qp.full_catalog["dec"])
#qp.full_catalog = qp.full_catalog[ catalog_mask ]

nir_area = calc_survey_area(
    [J_mask_path, K_mask_path], ra_limits=ra_limits, dec_limits=dec_limits
)
opt_area = calc_survey_area(
    [J_mask_path, K_mask_path],#, optical_mask_path],
    ra_limits=ra_limits, dec_limits=dec_limits
)
print("loaded")
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
    f"{imag} - {Kmag} > 3.05",
    f"{Kmag} > 14", 
    f"{imag} < 25.0",
    #f"i_flux_{opt_mag_type} < 10**10",
    catalog="catalog",
)
qp.create_selection(
    "sf_gzK",
    f"({zmag}-{Kmag}) -1.27*({gmag}-{zmag}) >= -0.022",
    catalog="gals"
)
qp.create_selection(
    "pe_gzK",
    f"({zmag}-{Kmag}) -1.27*({gmag}-{zmag}) < -0.022", f"{zmag}-{Kmag}>2.55",
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
randoms = full_randoms[ random_mask ]#& single_tile_mask]

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
    jwk["Kmag"].values, jwk["ero295_hsc"].values, marker="x", color="k", label="jwk ero295"
)
qp.num_density.axes.semilogy()
qp.num_density.axes.legend()

qp.create_plot("gzK_density")
lao = pd.read_csv(lao_number_counts_path)

qp.gzK_density.plot_number_density(
    Ktot_mag, selection=qp.pe_gzK, bins=bins, survey_area=opt_area, label="drgs, J-K > 2.3",
)
qp.gzK_density.plot_number_density(
    Ktot_mag, selection=qp.sf_gzK, bins=bins, survey_area=opt_area, label="drgs, J-K > 2.3",
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



plt.show()


cat1 = Catalog(
    ra=qp.to_correlate["ra"], dec=qp.to_correlate["dec"], ra_units="deg", dec_units="deg"
)
cat2 = Catalog(ra=randoms.ra, dec=randoms.dec, ra_units="deg", dec_units="deg")

treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"
with open(treecorr_config_path, "r") as f:
    treecorr_config = yaml.load(f, Loader=yaml.FullLoader)

treecorr_config["num_threads"] = 3

dd = NNCorrelation(config=treecorr_config)
rr = NNCorrelation(config=treecorr_config)
dr = NNCorrelation(config=treecorr_config)

dd.process(cat1)
rr.process(cat2)
dr.process(cat1, cat2)


jwk_path = paths.input_data_path / "plotting/jwkdat.csv"
jwk = pd.read_csv(jwk_path, names=["x", "w"])

xi, varxi = dd.calculateXi(rr=rr) #, dr=dr)
xi_ls, varxi_ls = dd.calculateXi(rr=rr, dr=dr)

print(xi)

print(np.log10(dd.rnom[1:]/dd.rnom[:-1]))

fig, ax = plt.subplots()
ax.plot(dd.rnom/3600., dd.npairs, label="dd counts")
ax.plot(dr.rnom/3600., dr.npairs, label="dr counts")
ax.plot(rr.rnom/3600., rr.npairs, label="rr counts")
ax.legend()
ax.loglog()

fig, ax = plt.subplots()
#ax.plot(dd.rnom/3600., xi)
ax.plot(dd.rnom/3600., xi, color="C0", marker="x", label="naive")
ax.plot(dd.rnom/3600., xi + ic, color="C0", marker="x", ls="--", label="naive + ic")
ax.plot(dd.rnom/3600., xi_ls, color="C1", marker="^", label="LS est.")
ax.scatter(jwk["x"], jwk["w"], s=10, color="k", label="Kim+ 2011")
ax.legend()
ax.loglog()
plt.show()


























