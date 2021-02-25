import logging

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from regions import PixCoord, PolygonSkyRegion, CompoundSkyRegion

logger = logging.getLogger("region_utils")

def in_only_one_tile(coord, mosaic_list, ):
    region_list = []
    wcs_list = []
    for mosaic_path in mosaic_list:
        with fits.open(mosaic_path) as f:
            fwcs = WCS(f[0].header)
            wcs_list.append(fwcs)
        footprint = SkyCoord(fwcs.calc_footprint(), unit="degree")        
        reg = PolygonSkyRegion(footprint)
        region_list.append(reg)
    mask = in_only_one_region(region_list, wcs_list, coord)
    logger.info(f"mask keeps {mask.sum()}/{len(mask)} objects ")
    assert len(mask) == len(coord)
    return mask

def in_only_one_region(region_list, wcs_list, coord):
    mask_list = []
    for reg, fwcs in zip(region_list, wcs_list):
        mask = reg.contains(coord, fwcs)
        mask_list.append(mask)
    arr = np.stack(mask_list, axis=1)
    output_mask = ( np.sum(arr, axis=1) == 1 )
    #combined_mask = ( combined_mask == 1 )
    return output_mask
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from dxs.utils.image import uniform_sphere
    from dxs import paths

    field = "SA"
    tiles = [x for x in range(1,13)]
    band = "J"
    
    mosaic_list = [paths.get_mosaic_path(field, tile, band) for tile in tiles]
    randoms = SkyCoord(
        uniform_sphere((332., 336.), (-1.2, 1.8), size=100_000), unit="degree"
    )
    
    mask = in_only_one_tile(mosaic_list, randoms)
    good_randoms = randoms[ mask ]
    fig, ax = plt.subplots()
    ax.scatter(randoms.ra, randoms.dec, s=1)
    ax.scatter(good_randoms.ra, good_randoms.dec, s=1)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
    
