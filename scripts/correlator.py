import logging
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table

from easyquery import Query
from treecorr import NNCorrelation, Catalog

from dxs.utils.image import uniform_sphere, objects_in_coverage, calc_survey_area
from dxs import QuickPlotter
from dxs.utils.misc import calc_range
from dxs import paths



logger = logging.getLogger("main")

field = "SA"

nir_mag_type = "aper_30"
opt_mag_type = "aper_3"

gmag = f"G_mag_{opt_mag_type}"
imag = f"I_mag_{opt_mag_type}"
zmag = f"Z_mag_{opt_mag_type}"
Jmag = f"J_mag_{nir_mag_type}"
Kmag = f"K_mag_{nir_mag_type}"



ic = 0.02

catalog_path = paths.catalogs_path / f"{field}00/{field}00_cfhtls.fits"
#catalog = Table.read(catalog_path)

jwk_number_counts_path = paths.input_data_path / "plotting/jwk_number_counts.csv"

qp = QuickPlotter.from_fits(
    catalog_path, 
    "dec< 1.6", "dec>-1.0", "ra<333.9", "ra>333.2", 
    f"{Jmag} < 30.0", f"{Kmag} < 30.0",
)
print("loaded")
qp.remove_crosstalks(catalog=qp.full_catalog)
qp.catalog[imag] -= 0.39 # fix bc. incorrectly converted to Vega from AB.

qp.create_selection("gals", f"{Jmag}-{Kmag} > 1.0")
qp.create_selection(
    "eros",
    f"{imag} - {Kmag} > 4.5",
    f"{Kmag} > 14", 
    #f"{Kmag} < 18.8", 
    f"{imag} < 24.6",
    #f"i_flux_{opt_mag_type} < 10**10",
    catalog="catalog",
)
qp.create_selection(
    "drgs", f"{Jmag}-{Kmag} > 2.3", catalog="gals"
)

ra_limits = calc_range(qp.catalog["ra"])
dec_limits = calc_range(qp.catalog["dec"])

optical_mask_path = paths.input_data_path / f"external/panstarrs/masks/{field}_mask.fits"
J_mask_path = paths.masks_path / f"{field}_J_good_cov_stars_mask.fits"
K_mask_path = paths.masks_path / f"{field}_K_good_cov_stars_mask.fits"
mask_list = [J_mask_path, K_mask_path]

nir_area = calc_survey_area([J_mask_path, K_mask_path])
opt_area = calc_survey_area([J_mask_path, K_mask_path]) #, optical_mask_path])

full_randoms = uniform_sphere(ra_limits, dec_limits, density=10000)
random_mask = objects_in_coverage(mask_list, full_randoms[:, 0], full_randoms[:, 1])
randoms = full_randoms[ random_mask ]

rra_mask = (randoms[:,1] < 1.6) & (randoms[:,1] > -1.0) 
rdec_mask = (randoms[:,0] < 333.9) & (randoms[:,0] > 332.2) 
randoms = randoms[ rra_mask & rdec_mask ]

ero_mask = objects_in_coverage(mask_list, qp.eros["ra"], qp.eros["dec"])
qp.eros = qp.eros[ ero_mask ]

qp.create_plot("iK_plot")
qp.iK_plot.color_magnitude(imag, Kmag, Kmag, selection=qp.catalog, s=1, alpha=0.1)
qp.iK_plot.color_magnitude(imag, Kmag, Kmag, selection=qp.eros, s=1, alpha=0.1)

qp.create_plot("JK_plot")
qp.JK_plot.color_magnitude(Jmag, Kmag, Kmag, selection=qp.catalog, s=1, alpha=0.1)
qp.JK_plot.color_magnitude(Jmag, Kmag, Kmag, selection=qp.drgs, s=1, alpha=0.1)

qp.create_plot("gzK_plot")
qp.gzK_plot.color_color(gmag, zmag, zmag, Kmag, selection=qp.catalog, s=1, alpha=0.1)


dm = 0.5
min_mag, max_mag = 14.0, 22.0
bins = np.arange(min_mag - 0.5*dm, max_mag + 1.5*dm, dm)



qp.create_plot("num_density")
qp.num_density.plot_number_density(
    Kmag, selection=qp.gals, bins=bins, survey_area=nir_area, label="galaxies, J-K>1.0",
)
qp.num_density.plot_number_density(
    Kmag, selection=qp.eros, bins=bins, survey_area=opt_area, label="eros, i-K > 4.5",
)
qp.num_density.plot_number_density(
    Kmag, selection=qp.drgs, bins=bins, survey_area=nir_area, label="drgs, J-K > 2.3",
)

jwk = pd.read_csv(jwk_number_counts_path, delim_whitespace=True)
qp.num_density.axes.scatter(jwk["K"], jwk["galaxies"], marker="o", color="k", label="jwk gal")
qp.num_density.axes.scatter(jwk["K"], jwk["eros45"], marker="x", color="k", label="jwk ero45")
qp.num_density.axes.scatter(jwk["K"], jwk["drgs"], marker="^", color="k", label="jwk drg")
qp.num_density.axes.semilogy()
qp.num_density.axes.legend()



qp.create_selection("to_correlate", f"{Kmag} < 18.8", catalog="eros")

qp.create_plot("coords_plot")
qp.coords_plot.plot_positions(randoms[:,0], randoms[:,1], s=1)
qp.coords_plot.plot_coordinates("ra", "dec", selection=qp.to_correlate, s=1)

#plt.show()
cat1 = Catalog(
    ra=qp.to_correlate["ra"], dec=qp.to_correlate["dec"], ra_units="deg", dec_units="deg"
)
cat2 = Catalog(ra=randoms[:,0], dec=randoms[:,1], ra_units="deg", dec_units="deg")

treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"
with open(treecorr_config_path, "r") as f:
    treecorr_config = yaml.load(f, Loader=yaml.FullLoader)

treecorr_config["num_threads"] = 3

nn = NNCorrelation(config=treecorr_config)
rr = NNCorrelation(config=treecorr_config)
dr = NNCorrelation(config=treecorr_config)

nn.process(cat1)
rr.process(cat2)
dr.process(cat1, cat2)


jwk_path = "./jwkdat.csv"
jwk = pd.read_csv(jwk_path, names=["x", "w"])

xi, varxi = nn.calculateXi(rr, dr=dr)

print(xi)

fig, ax = plt.subplots()
ax.plot(nn.rnom/3600., nn.npairs)
ax.plot(rr.rnom/3600., rr.npairs)
ax.loglog()

fig, ax = plt.subplots()
ax.plot(nn.rnom/3600., xi)
ax.scatter(nn.rnom/3600., xi, color="C0", marker="x")
ax.scatter(nn.rnom/3600., xi + ic, color="C0", marker="x", ls="--")
ax.scatter(jwk["x"], jwk["w"], s=10, color="k")
ax.loglog()
plt.show()


























