import subprocess
import shutil
import yaml
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np

from astropy.table import Table, vstack

from dxs.utils.misc import print_header
from dxs.utils.phot import vega_to_ab
from dxs.utils.table import explode_column, fix_column_names

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

astrom_atlas_url = "https://github.com/legacysurvey/unwise_psf/raw/master/py/unwise_psf/data/astrom-atlas.fits"
unwise_coadd_catalog_base_url = "https://faun.rc.fas.harvard.edu/unwise/release/band-merged"
unwise_coadd_imaging_base_url = "http://unwise.me/data/neo6/unwise-coadds/fulldepth/"
# see eg. http://unwise.me/data/neo6/unwise-coadds/fulldepth/017/0170m137/

unwise_dir = paths.input_data_path / "external/unwise/"
unwise_dir.mkdir(exist_ok=True, parents=True)
unwise_catalogs_dir = unwise_dir / "catalogs"
unwise_catalogs_dir.mkdir(exist_ok=True, parents=True)
unwise_masks_dir = unwise_dir / "masks"
unwise_masks_dir.mkdir(exist_ok=True, parents=True)


astrom_atlas_path = unwise_dir / "astrom-atlas.fits"
if astrom_atlas_path.exists() is False:
    subprocess.run(["wget", astrom_atlas_url, "-O", astrom_atlas_path])
atlas = Table.read(astrom_atlas_path)

unwise_zp = 22.5 # see "flux scale" in https://catalog.unwise.me/catalogs.html#bandmerged

def download_data(field, force_download=False):
    # Which catalogs do we want to download?
    ra_limits = survey_config["field_limits"][field]["ra"]
    dec_limits = survey_config["field_limits"][field]["dec"]
    coadd_ra = atlas["CRVAL"][:,0]
    coadd_dec = atlas["CRVAL"][:,1]
    ra_mask = (ra_limits[0]-1.0 < coadd_ra) & (coadd_ra < ra_limits[1]+1.0)
    dec_mask = (dec_limits[0]-1.0 < coadd_dec) & (coadd_dec < dec_limits[1]+1.0)

    field_coadds = atlas[ ra_mask & dec_mask ]

    unwise_masks_field_dir = unwise_masks_dir / f"{field}/"
    unwise_masks_field_dir.mkdir(exist_ok=True, parents=True)

    unwise_catalogs_field_dir = unwise_catalogs_dir / f"{field}"
    unwise_catalogs_field_dir.mkdir(exist_ok=True, parents=True)

    for coadd_id in field_coadds["COADD_ID"]:
        # download catalogs.
        coadd_catalog_path = unwise_catalogs_field_dir / f"{coadd_id}.cat.fits"
        coadd_catalog_url = f"{unwise_coadd_catalog_base_url}/{coadd_catalog_path.name}"
        if coadd_catalog_path.exists() is False or force_download:
            subprocess.run(["wget", coadd_catalog_url, "-O", coadd_catalog_path])
        # Now do masks.
        coadd_mask_url = f"{unwise_coadd_imaging_base_url}/{coadd_id[:3]}/{coadd_id}/unwise-{coadd_id}-msk.fits.gz"
        coadd_gz_mask_path = unwise_masks_field_dir / f"{coadd_id}_msk.fits.gz"
        coadd_mask_path = coadd_gz_mask_path.with_suffix("")
        if coadd_mask_path.exists() is False:
            subprocess.run(["wget", coadd_mask_url, "-O", coadd_gz_mask_path])
            subprocess.run(["gunzip", "-f", coadd_gz_mask_path])
            assert coadd_mask_path.exists()
            #coadd_gz_mask_path.unlink() # think this is rm?! 

        
def process_catalog(field):
    unwise_catalogs_field_dir = unwise_catalogs_dir / f"{field}"    
    glob_pattern = str(unwise_catalogs_field_dir / "*.cat.fits")
    print("glob with", glob_pattern)
    catalog_paths = glob(glob_pattern)

    print(f"{field} consists {len(catalog_paths)} unwise catalogs")

    catalog_list = []
    for catalog_path in catalog_paths:
        cat = Table.read(catalog_path)
        cat = cat[ cat["primary"] == 1]
        catalog_list.append(cat)

    output_cat = vstack(catalog_list)
    drop_cols = [
        "primary", "nm", "x", "y", "dx", "dy", "rchi2", "fracflux", "dspread_model", "sky", 
        "primary12", "ra12", "dec12", "qf"
    ]

    output_cat.remove_columns(drop_cols)
    
    columns_to_explode = [
        col for col in output_cat.colnames if len(output_cat[col].shape) == 2
    ]
    print(columns_to_explode)

    for col in columns_to_explode:
        new_columns = [f"W1_{col}", f"W2_{col}"]
        explode_column(output_cat, col, new_names=new_columns, remove=True)

    err_factor = 2.5 / np.log(10) # natural log!

    for band in ["W1", "W2"]:
        flux_col = output_cat[f"{band}_fluxlbs"]
        flux_err_col = output_cat[f"{band}_dfluxlbs"]
        flux_mask = (0 < flux_col)

        snr_col = np.full(len(output_cat), 0.0)
        mag_col = np.full(len(output_cat), 99.0)
        mag_err_col = np.full(len(output_cat), 99.0)

        missing_err = np.sum(flux_err_col[flux_mask] == 0)
        if missing_err > 0:
            print(f"{band} has {missing_err} missing flux_err values.")

        snr_col[flux_mask] = flux_col[flux_mask] / flux_err_col[flux_mask]
        mag_col[flux_mask] = unwise_zp - 2.5*np.log10( flux_col[flux_mask] )
        mag_col[flux_mask] = vega_to_ab(mag_col[flux_mask], band=band)
        mag_err_col[flux_mask] =  err_factor * 1. / snr_col[flux_mask]
        output_cat.add_column(snr_col, name=f"{band}_snr_auto")
        output_cat.add_column(mag_col, name=f"{band}_mag_auto")
        output_cat.add_column(mag_err_col, name=f"{band}_magerr_auto")     

    output_catalog_path = unwise_catalogs_field_dir / f"{field}_unwise.fits"
    output_cat.write(output_catalog_path, overwrite=True)
    fix_column_names(
        output_catalog_path, 
        column_lookup={"ra": "ra_unwise", "dec": "dec_unwise"}
    )


if __name__ == "__main__":

    field_choices = ["EN", "SA", "LH", "XM"]
    parser = ArgumentParser()
    parser.add_argument("--fields", required=False, choices=field_choices, nargs="+", default=field_choices)
    parser.add_argument("-f", "--force-download", default=False, action="store_true")

    args = parser.parse_args()
    fields = args.fields

    for field in fields:
        print_header(f"process unwise for {field}")
        download_data(field, force_download=args.force_download)
        process_catalog(field)























