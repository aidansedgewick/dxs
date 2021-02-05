import yaml
from collections import namedtuple
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table, join

from easyquery import Query

from dxs import MosaicBuilder, calculate_mosaic_geometry
from dxs.utils.phot import ab_to_vega

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

ps_config = survey_config["panstarrs"]

keep_cols = [
    "objID", "stackDetectID", "ra", "dec", "raErr", "decErr",
    "filterID", "zp", "zpErr", "primaryF", 
    "psfFlux", "psfFluxErr", "kronFlux", "kronFluxErr", "kronRad"
]

max_aper = 4
keep_ap_cols = [
    "objID", "stackDetectID", "primaryF", "isophotFlux", "isophotFluxErr",
    "petFlux", "petFluxErr", "petRadius", "petRadiusErr"
]
for ii in range(1, max_aper+1):
    keep_ap_cols.extend([f"flxR{ii}", f"flxR{ii}Err"])


Aperture = namedtuple(
    "Aperture", [
        "name", "flux", "flux_err", "mag", "mag_err", "snr", "new_flux", "new_flux_err"
    ]
)


def process_panstarrs_catalog(
    input_primary_catalog_path, 
    input_aperture_catalog_path, 
    output_catalog_path, 
    band_priority="izryg"
):


    # Primary catalog.
    full_primary_catalog = Table.read(input_primary_catalog_path, format="fits")
    for col in full_primary_catalog.colnames:
        new_name = col[4:]
        full_primary_catalog.rename_column(col, new_name)
    full_primary_catalog = full_primary_catalog[ keep_cols ]
    primary_catalog = Query("primaryF==1").filter(full_primary_catalog)
    primary_catalog.rename_column("zpErr", "zp_err")

    # Aperture flux catalog.
    full_aperture_catalog = Table.read(input_aperture_catalog_path, format="fits")
    full_aperture_catalog = full_aperture_catalog[ keep_ap_cols ]
    full_aperture_catalog.rename_column("objID", "objID_ap")
    aperture_catalog = Query("primaryF==1").filter(full_aperture_catalog)
    aperture_catalog.rename_column("isophotFlux", "isoFlux")
    aperture_catalog.rename_column("isophotFluxErr", "isoFluxErr")
    aperture_catalog.rename_column("petRadius", "pet_radius")
    aperture_catalog.rename_column("petRadiusErr", "pet_radius_err")

    # Join 'em together.
    jcat = join(
        primary_catalog, aperture_catalog, keys="stackDetectID", join_type="left"
    )
    drop_cols = ["primaryF_1", "primaryF_2", "objID_ap", "stackDetectID"]
    jcat.remove_columns(drop_cols)

    print(f"{output_catalog_path.stem} joined: {len(jcat)}")

    apertures = []

    for ap in ["kron", "iso", "pet"]:
        d = {
            "name": ap, "flux": f"{ap}Flux", "flux_err": f"{ap}FluxErr", 
            "mag": f"mag_{ap}", "mag_err": f"mag_err_{ap}", "snr": f"snr_{ap}", 
            "new_flux": f"flux_{ap}", "new_flux_err": f"flux_err_{ap}"
        }
        apertures.append(Aperture(**d))
    for N_aper in range(1, max_aper+1):
        ap_val = int(round(10*ps_config["apertures"][N_aper-1], 0))
        ap = f"aper{ap_val:02d}"
        d = {
            "name": ap, "flux": f"flxR{N_aper}", "flux_err": f"flxR{N_aper}Err", 
            "mag": f"mag_{ap}", "mag_err": f"magerr_{ap}", "snr": f"snr_{ap}", 
            "new_flux": f"flux_{ap}", "new_flux_err": f"flux_err_{ap}"
        }
        apertures.append(Aperture(**d))

    err_factor = 2.5 / np.log(10) # natural log!

    for ap in apertures:
        flux_col = jcat[ap.flux]
        flux_err_col = jcat[ap.flux_err]
        flux_mask = (0 < flux_col) & (flux_col < 10e10)

        snr_col = np.full(len(jcat), 0.0)
        mag_col = np.full(len(jcat), 99.0)
        mag_err_col = np.full(len(jcat), 99.0)

        missing_err = np.sum(flux_err_col[flux_mask] == 0)
        if missing_err > 0:
            print(f"{ap} has {missing_err} missing flux_err values.")

        snr_col[flux_mask] = flux_col[flux_mask] / flux_err_col[flux_mask]
        mag_col[flux_mask] = jcat["zp"][flux_mask] - 2.5*np.log10( flux_col[flux_mask] )
        mag_err_col[flux_mask] = np.sqrt(
            jcat["zp_err"][flux_mask]**2 + ( err_factor * 1. / snr_col[flux_mask] )**2 
        )
        jcat.add_column(snr_col, name=ap.snr)
        jcat.add_column(mag_col, name=ap.mag)
        jcat.add_column(mag_err_col, name=ap.mag_err)
        if ap.new_flux is not None:
            jcat.rename_column(ap.flux, ap.new_flux)
        if ap.new_flux_err is not None:
            jcat.rename_column(ap.flux_err, ap.new_flux_err)

    # Start an output catalog.
    objID = np.unique(jcat["objID"])
    output_catalog = Table([objID], names=["objID"])
    for bandID, band in enumerate(ps_config["bands"], 1):
        f_cat = Query(f"filterID=={bandID}").filter(jcat)
        for col in f_cat.colnames:
            if col == "objID":
                continue
            f_cat.rename_column(col, f"{band}_{col}")
        f_cat.remove_columns([f"{band}_filterID"])
        for col in f_cat.colnames:
            if col.endswith("mag"):
                #dM = ps_config["ab_to_vega"][band]
                f_cat[col] = ab_to_vega(f_cat[col], band=band)
        output_catalog = join(output_catalog, f_cat, keys="objID", join_type="left")

    ra_col = np.full(len(output_catalog), -99.0)
    dec_col = np.full(len(output_catalog), -99.0)

    for band in band_priority:
        if not any(ra_col < -90.0):
            break
        band_ra = f"{band}_ra"
        band_dec = f"{band}_dec"
        mask = output_catalog[band_ra] > -99. & output_catalog[band_dec] > -99.)
        ra_col[mask] = output_catalog[band_ra][ mask ]
        dec_col[mask] = output_catalog[band_ra][ mask ]
    output_catalog.add_column(ra_col, name="panstarrs_ra")
    output_catalog.add_column(dec_col, name="panstarrs_dec")

    output_catalog.write(output_catalog_path, overwrite=True)

def process_panstarrs_mosaic_mask(
    ps_field, output_path, extension=None, base_dir=None, pixel_scale=1.0
):
    if base_dir is None:
        base_dir = paths.input_data_path / f"external/panstarrs/images"
    if extension is None:
        extension = ".unconv.fits"
    glob_str = str(base_dir / f"{ps_field}/i/skycell**")
    dir_list = glob(glob_str)
    stack_list = []
    for directory in dir_list:
        directory = Path(directory)
        stack_list.append( glob(str(directory / f"*{extension}"))[0] )
    
    center, size = calculate_mosaic_geometry(
        stack_list, ccds=[0],
        pixel_scale=pixel_scale,
        border=100
    )
    mask_config = {
        "combine_type": "max",
        "pixel_scale": 1.0,
        "back_default": 0.0,
        "gain_default": 1.0,
        "interpolate": "nearest",
        "fscalastro_type": None,
        "center_type": "manual",
        "center": center,
        "image_size": size,
        "pixelscale_type": "manual",
    }
    mask_builder = MosaicBuilder(
        stack_list, output_path, swarp_config=mask_config
    )
    mask_builder.write_swarp_list(stack_list)
    mask_builder.build(prepare_hdus=False)

    

if __name__ == "__main__":

    catalog_dir = paths.input_data_path / "external/panstarrs/"

    for ii, (field, field_name) in enumerate(survey_config["code_to_field"].items()):
        ps_field = ps_config["from_dxs_field"][field]
        print(ii, ps_field)
        primary_catalog_path = catalog_dir / f"stackdetectionfull_{ps_field}.fit"
        aperture_catalog_path = catalog_dir / f"stackapflx_{ps_field}.fit"
        output_catalog_path = catalog_dir / f"{field}_panstarrs.fits"
        
        process_panstarrs_catalog(
            primary_catalog_path, aperture_catalog_path, output_catalog_path
        )

        mask_dir = paths.input_data_path / f"external/panstarrs/masks"
        mask_dir.mkdir(exist_ok=True, parents=True)
        output_path = mask_dir / f"{field}_mask.fits"
        process_panstarrs_mosaic_mask(
            ps_field, output_path=output_path, extension=".unconv.fits"
        )
    

