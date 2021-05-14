import os
import time
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from dustmaps import sfd
from easyquery import Query
from eazy.photoz import PhotoZ
from eazy.param import EazyParam
from eazy.utils import path_to_eazy_data

from dxs.utils.phot import vega_to_ab, ab_to_flux, apply_extinction

from dxs import paths

parser = ArgumentParser()
parser.add_argument("--n_cpus", default=4, type=int, required=False)

args = parser.parse_args()

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)


suffix = f"_panstarrs_swire"
catalog_path = paths.get_catalog_path("EN", 0, "", suffix=suffix)

full_catalog = Table.read(catalog_path)
full_catalog["id"] = np.arange(len(full_catalog))

nir_mag_type = "aper_30"
nir_tot_mag_type = "auto"
opt_mag_type = "aper_30" # ps use aper_18 or aper_30, cfhtls use aper_2 or aper_3.
swire_mag_type = "aper_30"

mag_type_lookup = {}
mag_type_lookup.update({b: nir_mag_type for b in "J H K".split()})
if "panstarrs" in suffix:
    mag_type_lookup.update({b: opt_mag_type for b in "g r i z y".split()})
if "swire" in suffix:
    mag_type_lookup.update({b: swire_mag_type for b in "I1 I2 I3 I4".split()})

sfdq = sfd.SFDQuery()

for band in "J H K".split():
    color_mag = f"{band}_mag_{nir_mag_type}"
    total_mag = f"{band}_mag_{nir_tot_mag_type}"
    if color_mag in full_catalog.columns:
        print(color_mag)
        full_catalog[color_mag] = vega_to_ab(full_catalog[color_mag], band=band)
    if total_mag in full_catalog.columns:
        print(total_mag)
        full_catalog[total_mag] = vega_to_ab(full_catalog[total_mag], band=band)



panstarrs_translate = {"g": 334, "r": 335, "i": 336, "z": 337, "y": 338}
wfcam_translate = {"J": 263, "H": 264, "K": 265}
swire_translate = {"I1": 18, "I2": 19, "I3": 20, "I4": 21}

full_translate = {}
full_translate.update(wfcam_translate)
if "panstarrs" in suffix:
    full_translate.update(panstarrs_translate)
if "swire" in suffix:
    full_translate.update(swire_translate)

coords = SkyCoord(ra=full_catalog["ra"], dec=full_catalog["dec"], unit="deg")
ebv = sfdq(coords)

translate_dict = {}
for b in "g r i z y J H K I1 I2 I3 I4".split():
    mag_type = mag_type_lookup[b]
    mag_col = f"{b}_mag_{mag_type}"
    snr_col = f"{b}_snr_{mag_type}"
    magerr_col = f"{b}_magerr_{mag_type}"
    if not mag_col in full_catalog.columns:
        print(f"skip {mag_col}")
        continue

    ix = full_translate[b]

    mag_mask = full_catalog[mag_col] < 30.

    if survey_config["reddening_coeffs"].get(b, None) is not None:
        full_catalog[mag_col] = apply_extinction(full_catalog[mag_col], ebv, band=b)
    else:
        print(f"no redenning coeff for band {b}")

    flux_col = f"{b}_flux"
    flux = np.full(len(full_catalog), -99.)
    flux[ mag_mask ] = ab_to_flux(full_catalog[mag_col][ mag_mask ], b) * 1e6 # uJansky
    full_catalog[flux_col] = flux
    translate_dict[flux_col] = f"F{ix}"


    fluxerr_col = f"{b}_fluxerr"
    fluxerr = np.full(len(full_catalog), -99.)
    snr = full_catalog[magerr_col] * np.log(10.) / 2.5
    fluxerr[ mag_mask ] = flux[ mag_mask ] * snr[ mag_mask ]

    #fluxerr[ mag_mask ] =  ab_to_flux(full_catalog[magerr_col][ mag_mask ], b) * 1e6 # uJansky    
    full_catalog[fluxerr_col] = fluxerr
    translate_dict[fluxerr_col] = f"E{ix}"
    

catalog = Query(
    "J_crosstalk_flag < 1", "K_crosstalk_flag < 1", 
    "J_coverage > 0", "K_coverage > 0"
).filter(full_catalog)

imag = "i_mag_aper_30"
Kmag = "K_mag_aper_30"
Ktot_mag = "K_mag_auto"
test_cat = Query(
    f"{Ktot_mag} < 22.0"# f"{imag}-{Kmag} > 2.45", f"{imag} < 30.", #
).filter(catalog)

test_cat["z_spec"] = np.zeros(len(test_cat))

#test_cat = test_cat[:250]

test_cat_path = paths.base_path / "en_photoz_test/en_test.cat.fits"
test_cat.write(test_cat_path, overwrite=True)

def modified_templates_file(
    templates_file=None, base_path=None, output_path=None, names=None
):
    eazy_data_path = Path(path_to_eazy_data())
    if templates_file is None:
        templates_file = (
            eazy_data_path / "templates/fsps_full/tweak_fsps_QSF_12_v3.param"
        )
    templates_file = Path(templates_file)
    if names is None:
        names = "idx path x0 x1 x2".split()
    templates_param = pd.read_csv(
        templates_file, names=names, delim_whitespace=True
    )
    templates_param["path"] = str(eazy_data_path) + "/" + templates_param["path"].astype(str)
    if output_path is None:
        output_path = paths.base_path / templates_file.name
    templates_param.to_csv(output_path, header=False, index=False, sep=" ")
    return output_path

def modified_eazy_parameters(
    paths_to_modify=None, **kwargs
):
    eazy_data_path = Path(path_to_eazy_data())
    paths_to_modify = paths_to_modify or []
    param = EazyParam()
    param["FILTERS_RES"] = eazy_data_path / f"filters" / param["FILTERS_RES"]
    print(kwargs)
    for key, value in kwargs.items():
        #if key.upper() not in param:
        #    print(f"WARNING: unknown key {key.upper()}")
        if isinstance(value, Path):
            value = str(value)
        param[key.upper()] = value
    for path in paths_to_modify:
        print(path, param[path.upper()])
        param[path.upper()] = eazy_data_path / param[path.upper()]

    return param

templates_file_path = modified_templates_file()

param_file_path = paths.base_path / "en_photoz_test/en_photoz.param"
param = modified_eazy_parameters(
    paths_to_modify=[
        "prior_file", "templates_file", "wavelength_file", "temp_err_file",
    ],
    #templates_file=templates_file_path,
    catalog_file=test_cat_path, 
    z_max=6.0, 
    prior_filter="K_flux",
    prior_file="templates/prior_K_TAO.dat",
    main_output_file=paths.base_path / "en_photoz_test/en_photozs",
    #cat_has_extcorr="n",
)
param.write(param_file_path)


"""
param_file_path = Path("./test.param")
param = EazyParam()
param["FILTERS_RES"] = eazy_data / f"filters" / param["FILTERS_RES"]
for p in paths_to_modify:
    param[p] = eazy_data / param[p]
param["TEMPLATES_FILE"] = "./tweak_fsps_QSF_12_v3.param"
param["CATALOG_FILE"] = str(test_ero_cat_path)
param["Z_MAX"] = 6.0
param["PRIOR_FILTER"] = "K_flux"
param.write(param_file_path)
"""


lines = []
for k, v in translate_dict.items():
    if k in test_cat.columns:
        lines.append(f"{k} {v}\n")

translate_path = paths.base_path / "en_photoz_test/en_photozs.translate"
with open(translate_path, "w+") as f:
    f.writelines(lines)


t1 = time.time()
phz = PhotoZ(
    param_file=str(param_file_path), translate_file=str(translate_path), n_proc=args.n_cpus
)
phz.fit_parallel()
t2 = time.time()
dt = t2 - t1
rate = dt / len(test_cat)
print(f"fit {len(test_cat)} parallel in {dt:.2f} s (={rate:.2e}  obj / s)")
t1 = time.time()
zout, hdu = phz.standard_output(prior=True, beta_prior=True, save_fits=True)
t2 = time.time()
dt = t2 - t1
print(f"write stdout in {t2-t1}")

"""
for key, value in zout.meta.items():
    if len(key) > 8:

        if isinstance(value, str):
            print("KEY", key, value, len(key), len(value))            
        if isinstance(value, str) and len(value) > 50:
            try:
                ep = path_to_eazy_data()
                new_value = os.path.relpath(value, ep)
            except:
                new_value = value
            if len(new_value) > 50:
                new_value = new_value[-50:]
            zout.meta[key] = new_value
            print("NEW VALUE", key, value, new_value)"""


    

#zout.write("zout.cat.fits", overwrite=True)

#for ii in range(len(eros)):
#    print(f"plot {ii}")
#    fig, ax = phz.show_fit(ii, id_is_idx=True)
#    fig.savefig(f"./eazy_plots/eazy_{ii:03d}_pdf.png")
#    plt.close()





#print(len(eros))

