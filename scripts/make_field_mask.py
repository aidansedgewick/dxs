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


    
def mask_stars_in_data(data, wcs, ra, dec, radii):
    pix_scale = np.mean(proj_plane_pixel_scales(wcs)) # TODO better way to get pix scale?

    sky_coord = SkyCoord(ra=ra, dec=dec)
    pix_coord = PixCoord.from_sky(sky_coord, wcs=wcs)
    pix_radii = radii / (pix_scale)
    ylen, xlen = wcs.array_shape # order bc zero-indexing...
    print(data.shape)
    for pix, rad in zip(pix_coord, pix_radii):
        xpix, ypix = pix.xy
        region = CirclePixelRegion(center=pix, radius=rad)
        mask = np.invert(region.to_mask(mode="exact").data.astype(bool))

        xgrid = np.arange(int(xpix-rad), int(xpix+rad) + 1)
        ygrid = np.arange(int(ypix-rad), int(ypix+rad) + 1)

        print(mask.shape, ygrid.shape, xgrid.shape, xpix, ypix, rad)

        # carefully select incase some parts of mask are outside data.
        xm = (0<=xgrid) & (xgrid<xlen)
        ym = (0<=ygrid) & (ygrid<ylen)
        mask = mask[ ym, : ][ :, xm ]
        # Mask is only big enough to include the circle region, so only mask that rectangle.
        xslicer = slice(
            max(0, int(xpix - rad)), min(int(xpix + rad) + 1, xlen), 1
        )
        yslicer = slice(
            max(0, int(ypix - rad)), min(int(ypix + rad) + 1, ylen), 1
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
        assert use_path.exists()
        
        mosaic_list.append(use_path)

    mask_path = paths.masks_path / f"{field}_{band}.fits"
    config_path = paths.config_path / "swarp/coverage.swarp"
    config = {"combine_type": "max"}
    builder = MosaicBuilder(
        mosaic_list, mask_path, swarp_config=config, swarp_config_file=config_path
    )
    #builder.build(prepare_hdus=False)

    if args.ignore_stars is True:
        print("Done!")
        sys.exit()

    catalog_path = paths.catalogs_path / f"{field}00/{field}00{band}.fits"
    catalog = Table.read(catalog_path)

    stars = Query(f"{band}_mag_auto < 10").filter(catalog)

    with fits.open(mask_path) as mask:
        data = mask[0].data
        mask_wcs = WCS(mask[0].header)
        data = mask_stars_in_data(
            data, 
            mask_wcs, 
            stars[f"{band}_ra"], 
            stars[f"{band}_dec"], 
            2*stars[f"{band}_fwhm_world"]
        )

        hdu = fits.PrimaryHDU(data=data, header = mask[0].header())
        hdu.writeto(mask_path, overwrite=True)









