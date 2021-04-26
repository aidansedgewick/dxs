import logging
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from regions import SkyRegion, PixCoord, PolygonSkyRegion, CompoundSkyRegion, write_ds9, read_ds9
from spherical_geometry.polygon import SphericalPolygon

from dxs.utils.image import build_mosaic_wcs

import matplotlib.pyplot as plt

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
    mask = in_only_one_region(region_list, coord, wcs_list=wcs_list)
    logger.info(f"mask keeps {mask.sum()}/{len(mask)} objects ")
    assert len(mask) == len(coord)
    return mask

def in_only_one_region(
    coord: SkyCoord, region_list: List[SkyRegion], wcs_list: List[WCS] = None
):
    if all(isinstance(p, Path) or isinstance(p, str) for p in region_list):
        region_list = [read_ds9(p) for p in region_list]

    if wcs_list is None:
        wcs_list = []
        for region in region_list:
            rwcs = guess_wcs_from_region(region)
            wcs_list.append(rwcs)
    if not len(region_list) == len(wcs_list):
        raise ValueError("region_list {len(region_list)} and wcs_list {len(wcs_list)} must be same length")
    mask_list = []
    for region, fwcs in zip(region_list, wcs_list):
        region_mask = region.contains(coord, fwcs)
        mask_list.append(region_mask)
    arr = np.stack(mask_list, axis=1)
    mask = ( np.sum(arr, axis=1) == 1 )
    assert len(mask) == len(coord)
    #combined_mask = ( combined_mask == 1 )
    return mask

def guess_wcs_from_region(region: SkyRegion, pixel_scale: float = 0.2):
    if isinstance(region, PolygonSkyRegion):
        center = SkyCoord(
            ra=0.5 * (np.min(region.vertices.ra) + np.max(region.vertices.ra)), 
            dec=0.5 * (np.min(region.vertices.dec) + np.max(region.vertices.dec)), 
            unit="degree"
        )
        cr = (
            np.max(region.vertices.ra) - np.min(region.vertices.ra), 
            np.max(region.vertices.dec) - np.min(region.vertices.dec)
        )
        size = (
            cr[0].value * pixel_scale / 3600.,
            cr[1].value * pixel_scale / 3600. * np.cos(center.dec * np.pi / 180.)
        )
    else:
        raise NotImplementedError(f"use PolygonSkyRegion, not {type(region)}")
    return build_mosaic_wcs(center, size, pixel_scale)


def make_tile_region(
    stack_list, pointings=None, ccds=None, output_path=None, n_cpus=1, steps=2
):
    if isinstance(stack_list, str):
        stack_list = Path(stack_list)
    if isinstance(stack_list, Path):
        stack_list = [stack_list]

    cdds = ccds or [1,2,3,4]

    regions = []
    for stack in stack_list:
        with fits.open(stack) as f:
            for ccd in ccds:
                fwcs = WCS(f[ccd].header)
                footprint = fwcs.calc_footprint()
                region = PolygonSkyRegion(
                    SkyCoord(
                        ra=footprint[:,0].flatten(), 
                        dec=footprint[:,1].flatten(),
                        unit="degree",
                    )
                )
                regions.append(region)

    if len(regions) > 1:
        pass

    if output_path is not None:
        write_ds9(regions, output_path)
    return region

def spherical_poly_wrapper(polygons):
    compound = SphericalPolygon.multi_intersection(polygons)
    compound_part_points = [points for points in compound.points]
    if len(compound_part_points) == 0:
        return None
    return compound
                

if __name__ == "__main__":
    print("no main block")
    
