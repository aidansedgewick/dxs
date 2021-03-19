import logging
import sys
from argparse import ArgumentParser
from itertools import product

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from easyquery import Query
from regions import PixCoord, CirclePixelRegion
from reproject import reproject_interp
from reproject import mosaicking


from dxs.utils.misc import tile_parser, create_file_backups, check_modules
from dxs.utils.image import build_ds9_command

from dxs import paths

logger = logging.getLogger("main")

mosaic_extensions = {
    "data": ".fits", 
    "cov": ".cov.fits", 
    "good_cov": ".cov.good_cov.fits"
}


def split_int(x, side=0.25):
    """
    split an integer close to in half. 
    eg: 
    >>> split_int(13) 
    (6, 7)
    if kwarg "side"%1< 0.5, return the largest split first.
    """

    side = side % 1
    if side < 0.5: 
        return (int(np.floor(x/2)), int(np.ceil(x/2)))
    else:
        return (int(np.ceil(x/2)), int(np.floor(x/2)))

    
def mask_stars_in_data(data, wcs, ra, dec, radii):
    pix_scale = np.mean(proj_plane_pixel_scales(wcs)) # TODO better way to get pix scale?

    sky_coord = SkyCoord(ra=ra, dec=dec)
    pix_coord = PixCoord.from_sky(sky_coord, wcs=wcs)
    pix_radii = radii / (pix_scale)
    ylen, xlen = data.shape # order bc zero-indexing...
    #print(data.shape)
    for ii, (pix, rad) in enumerate(zip(pix_coord, pix_radii)):

        if rad < pix_scale:
            continue
        xpix, ypix = pix.xy
        region = CirclePixelRegion(center=pix, radius=rad)
        mask = region.to_mask(mode="exact")
        bool_mask = mask.data.astype(bool)

        slicer = mask.bbox.slices
        data[slicer] = data[slicer] * bool_mask
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tiles")
    parser.add_argument("bands")
    parser.add_argument("--skip-reprojection", action="store_true", default=False)
    parser.add_argument("--skip-stars", action="store_true", default=False)
    parser.add_argument("--resolution", default=2.0, type=float, required=False)
    parser.add_argument("--mosaic-type", choices=["data", "cov", "good_cov"], default="good_cov")
    parser.add_argument("--n_cpus", default=None, type=int)
    # remember: dashes go to underscores after parse, ie, "--skip-mask" -> args.skip_mask 
    args = parser.parse_args()

    field = args.field
    tiles = tile_parser(args.tiles) # converts eg. "1,2,7-10" to [1,2,7,8,9,10]
    bands = args.bands.split(",")

    output_mask_paths = []

    for band in bands:
        mosaic_list = []
        
        suffix = mosaic_extensions[args.mosaic_type]
        for tile in tiles:
            p = paths.get_mosaic_path(field, tile, band)
            use_path = p.with_suffix(suffix)
            if not use_path.exists():
                logger.info(f"{use_path} not found")
            else:
                mosaic_list.append(use_path)
        if len(mosaic_list) == 0:
            logger.info("no mosaics - continue")
            continue

        output_path = paths.masks_path / f"{field}_{band}_{args.mosaic_type}_mask.fits"
        if args.skip_reprojection:
            with fits.open(output_path) as f:
                output_array = f[0].data
                header = f[0].header
                wcs_out = WCS(header)
        else:
            input_list = []
            for mosaic_path in mosaic_list:
                with fits.open(mosaic_path) as mos:
                    t = (mos[0].data, WCS(mos[0].header))
                    input_list.append(t)
                
            wcs_out, shape_out = mosaicking.find_optimal_celestial_wcs(
                input_list, resolution = args.resolution * u.arcsec
            )
            logger.info("starting reprojection")
            output_array, footprint = mosaicking.reproject_and_coadd(
                input_list, wcs_out, shape_out=shape_out,
                reproject_function=reproject_interp,
                combine_function="sum"
            )
            logger.info("finished reprojection")
            header = wcs_out.to_header()
            output_hdu = fits.PrimaryHDU(data=output_array, header=header)
            output_hdu.writeto(output_path, overwrite=True)

            output_mask_paths.append(output_path)
            logger.info(f"written {output_path}")

            input_list = []

        if args.skip_stars is True:
            print("Done!")
            continue # as we are in a loop for bands

        catalog_path = paths.catalogs_path / f"{field}00/sm{field}00_panstarrs.fits"
        catalog = Table.read(catalog_path)

        col = f"{band}_mag_auto"
        if col not in catalog.colnames:
            col = f"K_mag_auto"


        bright = f"J_mag_auto < 12" 
        blue = f"J_mag_aper_20 - K_mag_aper_20 < 1.0"
        stars = Query(f"{bright}", f"{blue}").filter(catalog)

        masked_output_path = paths.masks_path / f"{field}_{band}_{args.mosaic_type}_stars_mask.fits"

        radii = 10*stars[f"{band}_fwhm_world"]
        #radii = np.full(len(stars), 61./3600.)
        print(len(radii), radii.max())

        

        masked_array = mask_stars_in_data(
            output_array, 
            wcs_out, 
            stars[f"{band}_ra"], 
            stars[f"{band}_dec"], 
            radii #
        )

        hdu = fits.PrimaryHDU(data=masked_array, header=header)
        hdu.writeto(masked_output_path, overwrite=True)
        output_mask_paths.append(masked_output_path)
        logger.info(f"write {masked_output_path}")

#ds9_cmd = build_ds9_command(output_mask_paths)
#print(f"now do:\n    {ds9_cmd}")


