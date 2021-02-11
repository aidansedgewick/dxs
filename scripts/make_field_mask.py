import sys
from argparse import ArgumentParser
from itertools import product

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from easyquery import Query
from regions import PixCoord, CirclePixelRegion

from dxs import MosaicBuilder
from dxs import paths

def split_int(x, side=0.25):
    """
    split an integer close to in half. 
    eg: 
    >>> split_int(13) 
    (6, 7)
    if side%1< 0.5, return the largest split first.
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
    ylen, xlen = wcs.array_shape # order bc zero-indexing...
    #print(data.shape)
    for pix, rad in zip(pix_coord, pix_radii):
        xpix, ypix = pix.xy
        region = CirclePixelRegion(center=pix, radius=rad)
        mask = np.invert(region.to_mask(mode="exact").data.astype(bool)).T

        my1, my2 = split_int(mask.shape[0], side=ypix)
        mx1, mx2 = split_int(mask.shape[1], side=xpix)

        xgrid = np.arange(int(xpix - mx1), int(xpix + mx2))
        ygrid = np.arange(int(ypix - my1), int(ypix + my2))

        
        new_shape = (len(ygrid), len(xgrid))
        spix = f"{ypix:.3f}, {xpix:.3f}"

        if mask.shape != new_shape:
            print(mask.shape, new_shape, spix, (mx1, mx2), (my1, my2))

        # carefully select incase some parts of mask are outside data.
        xm = (0 < xgrid) & (xgrid < xlen)
        ym = (0 < ygrid) & (ygrid < ylen)
        mask = mask[ ym, : ][ :, xm ]
        #print(mask.shape)
        # Mask is only big enough to include the circle region, so only mask that rectangle.
        xslicer = slice(
            max(0, xgrid[0]), min(xgrid[-1], xlen)+1, 1
        )
        yslicer = slice(
            max(0, ygrid[0]), min(ygrid[-1], ylen)+1, 1
        )
        data[ yslicer, xslicer ] = data[ yslicer, xslicer ] * mask
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("tiles")
    parser.add_argument("band")
    parser.add_argument("--ignore_stars", action="store_true", default=False)
    parser.add_argument("--mosaic_type", choices=["data", "cov", "good_cov"], default="cov")
    parser.add_argument("--n_cpus", default=None, type=int)

    args = parser.parse_args()

    field = args.field
    tiles = [int(x) for x in args.tiles.split(",")]
    band = args.band

    mosaic_list = []
    mosaic_extensions = {"data": ".fits", "cov": ".cov.fits", "good_cov": ".cov.good_cov.fits"}

    suffix = mosaic_extensions[args.mosaic_type]
    for tile in tiles:
        p = paths.get_mosaic_path(field, tile, band)
        use_path = p.with_suffix(suffix)
        if not use_path.exists():
            raise IOError(f"No file {use_path}")
        
        mosaic_list.append(use_path)

    mask_path = paths.masks_path / f"{field}_{band}_{args.mosaic_type}_mask.fits"
    config_path = paths.config_path / "swarp/coverage.swarp"
    config = {"combine_type": "max", "pixel_scale": 2.0}
    builder = MosaicBuilder(
        mosaic_list, mask_path, swarp_config=config, swarp_config_file=config_path
    )
    builder.build(prepare_hdus=False, n_cpus=args.n_cpus)

    if args.ignore_stars is True:
        print("Done!")
        sys.exit()

    catalog_path = paths.catalogs_path / f"{field}00/{field}00{band}.fits"
    catalog = Table.read(catalog_path)

    stars = Query(f"{band}_mag_auto < 11").filter(catalog)

    star_mask_path = paths.masks_path / f"{field}_{band}_{args.mosaic_type}_stars_mask.fits"

    with fits.open(mask_path) as mask:
        data = mask[0].data
        mask_wcs = WCS(mask[0].header)
        data = mask_stars_in_data(
            data, 
            mask_wcs, 
            stars[f"{band}_ra"], 
            stars[f"{band}_dec"], 
            8*stars[f"{band}_fwhm_world"]
        )

        hdu = fits.PrimaryHDU(data=data, header=mask[0].header)
        hdu.writeto(star_mask_path, overwrite=True)









