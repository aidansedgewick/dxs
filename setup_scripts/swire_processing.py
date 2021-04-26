import logging
import re
import subprocess
import yaml
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from reproject import reproject_interp
from reproject import mosaicking

from dxs import paths

import matplotlib.pyplot as plt

logger = logging.getLogger("swire_proc")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

swire_config = survey_config["swire"]
url_base = swire_config["url_base"]
field_lookup = swire_config["field_lookup"]
swire_columns = ",".join(col for col in swire_config["select_columns"])


def download_catalog(field, output_path):
    swire_field = field_lookup.get(field, None)
    if swire_field is None and field == "test":
        print("WARN: using small test area inside elaisn1")
        swire_field = "elaisn1"

    ra_limits = survey_config["field_limits"].get(field, {}).get("ra", [242.0, 242.5])
    dec_limits = survey_config["field_limits"].get(field, {}).get("dec", [54.8, 55.2])

    constraints = (
        f"({ra_limits[0]} < ra) and (ra < {ra_limits[1]}) and "
        f"({dec_limits[0]} < dec) and (dec < {dec_limits[1]})"
    )
    query = (
        f"{url_base}"
        f"catalog={swire_field}_cat_s05&"
        f"spatial=None&"
        f"outfmt=1&" # 
        f"constraints={constraints}&" # ampersand included if constraints is not empty string.
        f"selcols={swire_columns}&"
    )

    print(f"DOWNLOAD: {output_path.name}")

    query_cmd = ["wget", f"{query}", "-O", output_path]
    status = subprocess.run(query_cmd)

    print(status)


def process_catalog(input_path, output_path):
    
    cat = Table.read(input_path, format="ascii")
    ap_flux_columns = ["flux_ap3_36", "flux_ap3_45", "flux_ap3_58", "flux_ap3_80"]
    kr_flux_columns = ["flux_kr_36", "flux_kr_45", "flux_kr_58", "flux_kr_80"]

    ap_lookup = {"ap3": "aper_30", "kr": "kron"}
    band_lookup = {36: "I1", 45: "I2", 58: "I3", 80: "I4"}
    
    err_factor = 2.5 / np.log(10) # natural log!
    for old_ap, new_ap in ap_lookup.items():
        for old_band, new_band in band_lookup.items():
            print(old_ap, old_band)
            old_flux_col = f"flux_{old_ap}_{old_band}"
            old_fluxerr_col = f"uncf_{old_ap}_{old_band}"
            if not old_flux_col in cat.columns:
                continue
            assert cat[old_flux_col].unit.name == "ujy"
            assert cat[old_fluxerr_col].unit.name == "ujy"
            cat[old_flux_col] = (cat[old_flux_col].data * u.uJy).to(u.Jy) # / 1e6
            cat[old_fluxerr_col] = (cat[old_fluxerr_col].data * u.uJy).to(u.Jy) # / 1e6            
            new_flux_col = f"{new_band}_flux_{new_ap}"
            mag_col = f"{new_band}_mag_{new_ap}"
            magerr_col = f"{new_band}_magerr_{new_ap}"
            ## What is this equation!?! 
            ## https://en.wikipedia.org/wiki/AB_magnitude#Definition
            ## for f_nu in Jy.

            mag_data = np.full(len(cat), 99.0)
            flux_mask = cat[old_flux_col] > 0.
            mag_data[flux_mask] = -2.5 * np.log10(cat[old_flux_col][flux_mask].data) + 8.9
            cat[mag_col] = mag_data

            print(f"{old_band}->{new_band}, {old_ap}->{new_ap} \n", cat[mag_col])
            new_fluxerr_col = f"{new_band}_fluxerr_{new_ap}"
            snr_col = cat[old_flux_col] / cat[old_fluxerr_col]
            cat[magerr_col] = err_factor * 1. / snr_col

            cat.rename_column(old_flux_col, new_flux_col)
            cat.rename_column(old_fluxerr_col, new_fluxerr_col)
    cat.rename_column("ra", "ra_swire")
    cat.rename_column("dec", "dec_swire")

    cat.write(output_path, overwrite=True)

def download_images(field, image_type=None, force_download=False):

    image_type = ["mask"]
    bulk_download_script_path = base_dir / f"{field}_wget_data.bat"     
    if not bulk_download_script_path.exists() or force_download:
        bulk_download_script_url = (
            swire_config["bulk_image_download_url_base"]
            + swire_config["bulk_image_download_urls"][field]
        )
        download_cmd = [
            "wget", bulk_download_script_url, "-O", bulk_download_script_path
        ]
        status = subprocess.run(download_cmd)


    image_dir = base_dir / f"images/{field}"
    image_dir.mkdir(exist_ok=True, parents=True) 
    with open(bulk_download_script_path, "r") as f:
        script_lines = f.readlines()
    
    image_type_pattern = "(" + "|".join(t for t in image_type) + ")"
    pattern = f"wget .*swire.*I[1,2].*_{image_type}.fits"
    for line in script_lines:
        if not re.match(pattern, line):
            continue
        if "I1" in line:
            band = "I1"
        elif "I2" in line:
            band = "I2"                
        output_dir = image_dir / band
        output_dir.mkdir(exist_ok=True, parents=True)
        line = line.replace("\"", "")
        cmd = line.split()
        image_name = cmd[-1].split("/")[-1] #.replace("\"", "")
        output_path = output_dir / image_name
        if not output_path.exists() or force_download:
            cmd.extend(["-O", output_path])
            print(cmd)
            status = subprocess.run(cmd)

def make_mask(field, band, output_path, image_type="mask", resolution=2.0):
   
    base_dir = paths.input_data_path / f"external/swire/images"
    glob_str = str(base_dir / f"{field}/{band}/*{image_type}.fits")
    mosaic_list = glob(glob_str)

    print(f"mask from {len(mosaic_list)} images")

    input_list = []
    for mosaic_path in mosaic_list:
        with fits.open(mosaic_path) as mos:
            data_array = mos[0].data.copy()
            img_wcs = WCS(mos[0].header)
        data_array = (data_array == 0)
        t = (data_array, img_wcs)
        input_list.append(t)
        
    wcs_out, shape_out = mosaicking.find_optimal_celestial_wcs(
        input_list, resolution=resolution * u.arcsec
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


if __name__ == "__main__":
    

    base_dir = paths.input_data_path / "external/swire/"
    base_dir.mkdir(exist_ok=True, parents=True)

    field_choices = ["EN", "LH", "XM"]
    image_type_choices = ["unc", "mosaic", "cov", "mask"]

    parser = ArgumentParser()
    parser.add_argument(
        "--fields", choices=field_choices, nargs="+", default=field_choices, required=False
    )
    parser.add_argument("--force-download", action="store_true", default=False)
    parser.add_argument("--skip-catalog", action="store_true", default=False)
    parser.add_argument("--skip-mask", action="store_true", default=False)
    parser.add_argument("--image-types", nargs="+", default=["mask"], choices=image_type_choices, required=False)
    parser.add_argument("--resolution", default=2.0, type=float)
    # remember: dashes go to underscores after parse, ie, "--skip-mask" -> args.skip_mask 
    args = parser.parse_args()

    for field in args.fields:
        if not args.skip_catalog:
            tbl_path = base_dir / f"{field}_swire.tbl"
            if not tbl_path.exists() or args.force_download:
                download_catalog(field, tbl_path)
            catalog_dir = base_dir / "catalogs"
            catalog_dir.mkdir(exist_ok=True, parents=True)
            output_path = catalog_dir / f"{field}_swire.cat.fits"
            process_catalog(tbl_path, output_path)

        if not args.skip_mask:
            download_images(
                field, image_type=args.image_types, force_download=args.force_download
            )

            mask_dir = base_dir / "masks"
            mask_dir.mkdir(exist_ok=True, parents=True)
            for image_type in args.image_types:
                for band in "I1 I2".split():
                    output_path = mask_dir / f"{field}_{band}_{image_type}.fits"
                    make_mask(
                        field, band, output_path, image_type=image_type, resolution=args.resolution
                    )
            

            
            


