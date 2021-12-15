import logging
import subprocess
import shutil
import yaml
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from reproject import mosaicking, reproject_interp, reproject_exact, reproject_adaptive

from dxs import CatalogExtractor, MosaicBuilder

from dxs.utils.misc import print_header, format_flags
from dxs.utils.phot import vega_to_ab
from dxs.utils.table import explode_column, fix_column_names

from dxs import paths

logger = logging.getLogger("unwise_proc")

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
unwise_coadds_dir = unwise_dir / "images"
unwise_coadds_dir.mkdir(exist_ok=True, parents=True)


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

    unwise_coadds_field_dir = unwise_coadds_dir / f"{field}/"
    unwise_coadds_field_dir.mkdir(exist_ok=True, parents=True)

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
        # Now do images.

        cid_dir_base = f"{unwise_coadd_imaging_base_url}/{coadd_id[:3]}/{coadd_id}"

        coadd_w1_url = f"{cid_dir_base}/unwise-{coadd_id}-w1-img-m.fits"
        coadd_w1_path = unwise_coadds_field_dir / f"{coadd_id}_w1.fits"
        if not coadd_w1_path.exists():
            subprocess.run(["wget", coadd_w1_url, "-O", coadd_w1_path])

        coadd_w2_url = f"{cid_dir_base}/unwise-{coadd_id}-w2-img-m.fits"
        coadd_w2_path = unwise_coadds_field_dir / f"{coadd_id}_w2.fits"
        if not coadd_w2_path.exists():
            subprocess.run(["wget", coadd_w2_url, "-O", coadd_w2_path])
           
        coadd_mask_url = f"{cid_dir_base}/unwise-{coadd_id}-msk.fits.gz"
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
        explode_column(output_cat, col, remove=True) # append suffixes 0, 1///
        output_cat.rename_column(f"{col}_0", f"W1_{col}")
        output_cat.rename_column(f"{col}_1", f"W2_{col}")

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

def process_coadds(field, force_coadds=False, n_cpus=None):
    print("process coadds")

    for W in ["W1", "W2"]:
        unwise_coadds_field_dir = unwise_coadds_dir / f"{field}"    
        w_glob_pattern = str(unwise_coadds_field_dir / f"*{W.lower()}.fits")
        print("glob with", w_glob_pattern)
        w_coadd_paths = glob(w_glob_pattern)

        
        w_mosaic_path = unwise_coadds_dir / f"{field}_{W}.fits"
        if not w_mosaic_path.exists() or force_coadds:
            config = {}
            config["pixelscale_type"] = "MAX"
            config["combine_type"] = "AVERAGE"
            config["subtract_back"] = "N"
            config = format_flags(config)

            header_keys = {}
            header_keys["MAGZPT"] =  (
                vega_to_ab(unwise_zp, band=W), "AB ZPT"
            )

            builder = MosaicBuilder(
                w_coadd_paths,
                w_mosaic_path,
                swarp_config=config,
                header_keys=header_keys
            )
            builder.build(prepare_hdus=False, n_cpus=n_cpus)

        """
        print(f"{W}: collect coadd data")
        w1_data_list = []
        for coadd_path in w1_coadd_paths:
            with fits.open(coadd_path) as f:
                w_data_list.append((f[0].data, f[0].header))
        print(f"{W}: find optimal wcs")
        w_wcs, w_shape = mosaicking.find_optimal_celestial_wcs(
            w_data_list
        )
        print(f"{W}: reproject")
        w_array, w_footprint = mosaicking.reproject_and_coadd(
            w_data_list, 
            w_wcs, 
            shape_out=w1_shape, 
            reproject_function=reproject_exact,
        )
        print(f"{W}: output")
        w_header = w_wcs.to_header()
        w_header["MAGZPT"] = vega_to_ab(unwise_zp, band=W)
        w_hdu = fits.PrimaryHDU(data=w_array, header=w_header)
        w_hdu.writeto(w_mosaic_path, overwrite=True)
        """


def forced_photometry(field, tile, band, prefix=None, force_resample=False, n_cpus=None):
    logger.info(f"{field} {tile} {band} resamp")

    dxs_mosaic_path = paths.get_mosaic_path(field, tile, band, prefix=prefix)
    with fits.open(dxs_mosaic_path) as f:
        f_data = f[0].data
        output_shape = f[0].data.shape
        output_wcs = WCS(f[0].header)
        out_xlen, out_ylen = output_wcs.pixel_shape # same as output_shape[::-1]
        center = SkyCoord.from_pixel(out_xlen / 2, out_ylen / 2, output_wcs)
        out_xscale, out_yscale = proj_plane_pixel_scales(output_wcs)
  
    w_resamp_dir = paths.input_data_path / f"external/unwise/images/resampled/{field}"
    w_resamp_dir.mkdir(exist_ok=True, parents=True)
    w_cat_dir = paths.input_data_path / f"external/unwise/catalogs/forced/{field}"
    w_cat_dir.mkdir(exist_ok=True, parents=True)

    for W in ["W1", "W2"]:
        w_resamp_stem = paths.get_mosaic_stem(field, tile, W)
        w_resamp_path = w_resamp_dir / f"{w_resamp_stem}.fits"
        if not w_resamp_path.exists() or force_resample:
            logger.info(f"resample to {w_resamp_path}")
            w_mosaic_path = unwise_coadds_dir / f"{field}_{W}.fits"
            with fits.open(w_mosaic_path) as wm:
                w_data = wm[0].data
                w_header = wm[0].header
                w_wcs = WCS(w_header)
                w_xscale, w_yscale = proj_plane_pixel_scales(w_wcs)
            cutout_shape = (
                int( (out_yscale / w_yscale) * 1.1 * out_ylen),
                int( (out_xscale / w_xscale) * 1.1 * out_xlen)
            )
            print(f"{W}: cutout")
            cutout = Cutout2D(w_data, center, cutout_shape, wcs=w_wcs)
            cutout_footprint = cutout.wcs.calc_footprint()
            output_footprint = SkyCoord(output_wcs.calc_footprint(), unit="deg")
            if not cutout.wcs.footprint_contains(output_footprint).all():
                raise ValueError("cutout not big enough for output")

            cutout_header = cutout.wcs.to_header()
            cutout_hdu = fits.PrimaryHDU(data=cutout.data, header=cutout_header)
            cutout_path = paths.scratch_swarp_path / f"{w_resamp_stem}_cutout.fits"
            cutout_hdu.writeto(cutout_path, overwrite=True)

            logger.info(f"{W}: resample")
            swarp_config = {}
            swarp_config["center"] = center #f"{center[0]:.6f},{center[1]:.6f}"
            swarp_config["image_size"] = output_shape[::-1] #f"{size[0]},{size[1]}"
            swarp_config["center_type"] = "MANUAL"
            swarp_config["pixelscale_type"] = "MANUAL"
            swarp_config["combine_type"] = "MAX"
            swarp_config = format_flags(swarp_config)

            builder = MosaicBuilder(
                [cutout_path],
                w_resamp_path,
                swarp_config=swarp_config,
                header_keys={"magzpt": w_header["MAGZPT"]},
            )
            builder.build(prepare_hdus=False, n_cpus=n_cpus)

            """
            output_array = reproject_adaptive(
                (cutout.data, cutout.wcs), 
                output_wcs, 
                shape_out=output_shape, 
                order="bilinear",
                return_footprint=False
            )
            output_header = output_wcs.to_header()
            output_header["MAGZPT"] = w_header["MAGZPT"]
            w_resamp = fits.PrimaryHDU(data=output_array, header=output_header)
            w_resamp.writeto(w_resamp_path, overwrite=True)
            """
        else:
            logger.info(f"{w_resamp_path} exists")
        
        logger.info(f"{W}: fp")
        w_cat_path = w_cat_dir / f"{w_resamp_stem}.cat.fits"
        ce = CatalogExtractor(
            detection_mosaic_path=dxs_mosaic_path,
            measurement_mosaic_path=w_resamp_path,
            use_weight=True, 
            catalog_path=w_cat_path,
            sextractor_config={
                "checkimage_name": w_cat_dir / f"{w_resamp_stem}.seg.fits"
            }
        )
        ce.extract()
        


if __name__ == "__main__":

    field_choices = ["EN", "SA", "LH", "XM"]
    parser = ArgumentParser()
    parser.add_argument("--fields", required=False, choices=field_choices, nargs="+", default=field_choices)
    parser.add_argument("-f", "--force-download", default=False, action="store_true")
    parser.add_argument("--n-cpus", default=None, type=int)

    args = parser.parse_args()
    fields = args.fields

    for field in fields:
        print_header(f"process unwise for {field}")
        download_data(field, force_download=args.force_download)
        #process_catalog(field)
        process_coadds(field, n_cpus=args.n_cpus)
        for tile in range(1, 13):
            try:
                forced_photometry(field, tile, "K", n_cpus=args.n_cpus)
            except Exception as e:
                print(e)



















