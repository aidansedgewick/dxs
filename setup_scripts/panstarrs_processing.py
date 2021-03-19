import logging
import yaml
from argparse import ArgumentParser
from collections import namedtuple
from glob import glob
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, join, Column, MaskedColumn
from astropy.wcs import WCS


from easyquery import Query
from reproject import reproject_interp
from reproject import mosaicking

from dxs import MosaicBuilder, calculate_mosaic_geometry
from dxs.utils.phot import ab_to_vega

from dxs import paths

logger = logging.getLogger("ps1_processor")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

ps_config = survey_config["panstarrs"]

keep_cols = [
    "objID", "stackDetectID", "ra", "dec", #"raErr", "decErr",
    "filterID", "zp", "zpErr", "primaryF", 
    "psfFlux", "psfFluxErr", "kronFlux", "kronFluxErr", "kronRad"
]

max_aper = 3
keep_ap_cols = [
    "objID", "stackDetectID", "primaryF", "isophotFlux", "isophotFluxErr",
    "petFlux", "petFluxErr", #"petRadius", "petRadiusErr"
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
    logger.info("loading primary catalog...")
    t1 = time()
    full_primary_catalog = Table.read(input_primary_catalog_path, format="fits")
    logger.info(f"loaded in {time()-t1:.2f} sec")
    for col in full_primary_catalog.colnames:
        new_name = col[4:] # get rid of prefix "sdf_" -- abbrev. "stackdetectionfull"?
        full_primary_catalog.rename_column(col, new_name)
    full_primary_catalog = full_primary_catalog[ keep_cols ]
    primary_catalog = Query("primaryF==1").filter(full_primary_catalog)
    primary_catalog.rename_column("zpErr", "zp_err")

    # Aperture flux catalog.
    logger.info("loading aperture flux catalog")
    t1 = time()
    full_aperture_catalog = Table.read(input_aperture_catalog_path, format="fits")
    logger.info(f"loaded in {time()-t1:.2f} sec")
    full_aperture_catalog = full_aperture_catalog[ keep_ap_cols ]
    full_aperture_catalog.rename_column("objID", "objID_ap")
    aperture_catalog = Query("primaryF==1").filter(full_aperture_catalog)
    aperture_catalog.rename_column("isophotFlux", "isoFlux")
    aperture_catalog.rename_column("isophotFluxErr", "isoFluxErr")
    #aperture_catalog.rename_column("petRadius", "pet_radius")
    #aperture_catalog.rename_column("petRadiusErr", "pet_radius_err")

    # Join 'em together.
    logger.info("starting primary/aperture join")
    t1 = time()
    jcat = join(
        primary_catalog, aperture_catalog, keys="stackDetectID", join_type="left"
    )
    logger.info(f"joined in {t1-time():.2f} sec")
    drop_cols = ["primaryF_1", "primaryF_2", "objID_ap", "stackDetectID"]
    jcat.remove_columns(drop_cols)

    logger.info(f"{output_catalog_path.stem} joined: {len(jcat)}")

    apertures = []

    for ap in ["kron", "iso", "pet", "psf"]:
        d = {
            "name": ap, "flux": f"{ap}Flux", "flux_err": f"{ap}FluxErr", 
            "mag": f"mag_{ap}", "mag_err": f"magerr_{ap}", "snr": f"snr_{ap}", 
            "new_flux": f"flux_{ap}", "new_flux_err": f"flux_err_{ap}"
        }
        apertures.append(Aperture(**d))
    for N_aper in range(1, max_aper+1):
        ap_val = int(round(10*ps_config["apertures"][N_aper-1], 0))
        ap = f"aper_{ap_val:02d}"
        d = {
            "name": ap, "flux": f"flxR{N_aper}", "flux_err": f"flxR{N_aper}Err", 
            "mag": f"mag_{ap}", "mag_err": f"magerr_{ap}", "snr": f"snr_{ap}", 
            "new_flux": f"flux_{ap}", "new_flux_err": f"fluxerr_{ap}"
        }
        apertures.append(Aperture(**d))

    err_factor = 2.5 / np.log(10) # natural log!

    for ap in apertures:
        if isinstance(jcat[ap.flux], MaskedColumn):
            jcat[ap.flux] = jcat[ap.flux].filled(-999.)
            jcat[ap.flux_err] = jcat[ap.flux_err].filled(-999.)
        flux_col = jcat[ap.flux]
        flux_err_col = jcat[ap.flux_err]
        print(ap.flux, type(jcat[ap.flux]) )
        flux_mask = (0 < flux_col) & (flux_col < 10e10)

        snr_col = np.full(len(jcat), 0.0)
        mag_col = np.full(len(jcat), 99.0)
        mag_err_col = np.full(len(jcat), 99.0)

        missing_err = np.sum(flux_err_col[flux_mask] == 0)
        if missing_err > 0:
            logger.warn(f"{ap} has {missing_err} missing flux_err values.")

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

        logger.info(f"{ap.mag}, {type(jcat[ap.mag])}")

    for ap in apertures:
        if ap.name == "kron":
            continue
        drop_cols = [ap.new_flux, ap.new_flux_err, ap.snr]
        jcat.remove_columns(drop_cols)
    jcat.remove_columns(["zp", "zp_err"])

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
        #for col in f_cat.colnames:
        #    if col.endswith("mag"):
        #        #dM = ps_config["ab_to_vega"][band]
        #        f_cat[col] = ab_to_vega(f_cat[col], band=band)
        output_catalog = join(output_catalog, f_cat, keys="objID", join_type="left")
        for ap in apertures:
            filler = {
                f"{band}_{ap.mag}": 99., f"{band}_{ap.mag_err}": 99.,
                f"{band}_{ap.new_flux}": -999., f"{band}_{ap.new_flux_err}": -999.,
                f"{band}_{ap.snr}": 0.
            }
            for col, fill_value in filler.items():
                if col in output_catalog.columns:
                    output_catalog[col] = output_catalog[col].filled(fill_value)

    ra_col = np.full(len(output_catalog), -99.)
    dec_col = np.full(len(output_catalog), -99.)

    catalog_len = len(output_catalog)
    for band in band_priority:
        if not any(ra_col < -90.):
            break
        band_ra = f"{band}_ra"
        band_dec = f"{band}_dec"
        output_catalog[band_ra] = output_catalog[band_ra].filled(-99.)
        output_catalog[band_dec] = output_catalog[band_dec].filled(-99.)
        mask = (output_catalog[band_ra] > -90.) & (output_catalog[band_dec] > -90.)
        ra_col[mask] = output_catalog[band_ra][ mask ]
        dec_col[mask] = output_catalog[band_dec][ mask ]
        len_selected = len(ra_col[ra_col > -90.])
        logger.info(f"selected {len_selected} of {catalog_len} coords from {band}")

    logger.info("add new ra/dec")
    output_catalog.add_column(ra_col, name="ra_panstarrs")
    output_catalog.add_column(dec_col, name="dec_panstarrs")

    t1 = time()
    logger.info("starting write...")
    output_catalog.write(output_catalog_path, overwrite=True)
    try:
        print_path = output_catalog_path.relative_to(Path.getcwd())
    except:
        print_path = output_catalog_path
    logger.info(f"catalog written to {print_path} in {(time()-t1)/60.:.2f} min")

def process_panstarrs_mosaic_mask(
    ps_field, output_path, extension=None, base_dir=None, pixel_scale=1.0
):
    if base_dir is None:
        base_dir = paths.input_data_path / f"external/panstarrs/images"
    if extension is None:
        extension = ".unconv.fits"
    glob_str = str(base_dir / f"{ps_field}/i/skycell**")
    dir_list = glob(glob_str)
    mosaic_list = []
    for directory in dir_list:
        directory = Path(directory)
        mosaic_list.append( glob(str(directory / f"*{extension}"))[0] )
    

    input_list = []
    for mosaic_path in mosaic_list:
        with fits.open(mosaic_path) as mos:
            data_array = mos[1].data.copy()
            img_wcs = WCS(mos[1].header)
        data_array = (data_array == 0)
        t = (data_array, img_wcs)
        input_list.append(t)
        
    wcs_out, shape_out = mosaicking.find_optimal_celestial_wcs(
        input_list, resolution = args.resolution * u.arcsec
    )
    logger.info("starting reprojection")
    array_out, footprint = mosaicking.reproject_and_coadd(
        input_list, wcs_out, shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="sum"
    )
    logger.info("finished reprojection")
    header = wcs_out.to_header()
    output_hdu = fits.PrimaryHDU(data=array_out, header=header)
    output_hdu.writeto(output_path, overwrite=True)

    """with fits.open(output_path) as f:
        new_data = (f[0].data == 0)
        new_data = new_data.astype(int)
        output_hdu = fits.PrimaryHDU(data=new_data, header=header)        
        output_hdu.writeto(output_path, overwrite=True)"""

    """
    center, size = calculate_mosaic_geometry(
        stack_list, ccds=[1],
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
    mask_builder.build(prepare_hdus=False)"""

    

if __name__ == "__main__":

    catalog_dir = paths.input_data_path / "external/panstarrs/"

    parser = ArgumentParser()
    parser.add_argument("--fields", default="SA,EN,LH,XM", required=False)
    parser.add_argument("--skip-catalog", action="store_true", default=False)
    parser.add_argument("--skip-mask", action="store_true", default=False)
    parser.add_argument("--mask-extension", default=".unconv.mask.fits")
    parser.add_argument("--resolution", default=2.0, type=float)
    # remember: dashes go to underscores after parse, ie, "--skip-mask" -> args.skip_mask 
    args = parser.parse_args()
    fields = args.fields.split(",")

    for ii, field in enumerate(fields):
        ps_field = ps_config["from_dxs_field"][field]
        logger.info(f"{ii}: {ps_field} = {field}")
        primary_catalog_path = catalog_dir / f"stackdetectionfull_{ps_field}.fit"
        aperture_catalog_path = catalog_dir / f"stackapflx_{ps_field}.fit"
        output_catalog_path = catalog_dir / f"{field}_panstarrs.fits"
        
        if not args.skip_catalog:
            process_panstarrs_catalog(
                primary_catalog_path, aperture_catalog_path, output_catalog_path
            )
        if not args.skip_mask:
            mask_dir = paths.input_data_path / f"external/panstarrs/masks"
            mask_dir.mkdir(exist_ok=True, parents=True)
            output_path = mask_dir / f"{field}_mask.fits"
            process_panstarrs_mosaic_mask(
                ps_field, output_path=output_path, extension=args.mask_extension
            )


