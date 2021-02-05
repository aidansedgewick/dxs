import logging
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.table import Table

from easyquery import Query
from treecorr import NNCorrelation, Catalog

from dxs.utils.image import uniform_sphere, objects_in_multi_coverage
from dxs.utils.misc import calc_range
from dxs import paths

logger = logging.getLogger("main")

field = "SA"

catalog_path = paths.catalogs_path / f"{field}00/{field}00.fits"
catalog = Table.read(catalog_path)

logger.info("catalog loaded")

catalog["i_mag_kron"] -= 0.8 # fix bc. incorrectly converted to Vega from AB.
catalog["K_mag_auto"] += 2.5 # fix bc. didn't include the exptime properly.
catalog["J_mag_auto"] += 2.5 # ""              "" 

logger.info("catalog fixed")

eros = Query(
    "i_mag_kron - K_mag_auto > 4.5", "K_mag_auto > 14", "K_mag_auto < 20.3", "i_mag_kron < 24.6"
).filter(catalog)

logger.info("queried")

fig, ax = plt.subplots()
ax.scatter(catalog["K_mag_auto"], catalog["i_mag_kron"] - catalog["K_mag_auto"], s=1)
ax.scatter(eros["K_mag_auto"], eros["i_mag_kron"] - eros["K_mag_auto"], s=1)
ax.set_ylim(-3, 10)
ax.set_xlim(6, 24)
plt.show()

ra_limits = calc_range(catalog["ra"])
dec_limits = calc_range(catalog["dec"])

optical_mask_path = paths.input_data_path / "external/panstarrs/masks/{field}_mask.fits
J_mask_path = paths.masks_path / "{field}_J.fits"
K_mask_path = paths.masks_path / "{field}_K.fits"
mask_list = [optical_mask_path, J_mask_path, K_mask_path]

full_randoms = uniform_sphere(ra_limits, dec_limits, density=1000)

cat1 = Catalog(ra=eros["ra"], dec=eros["dec"], ra_units="deg", dec_units="deg")
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

xi, varxi = nn.calculateXi(rr, dr=dr)

print(xi)
