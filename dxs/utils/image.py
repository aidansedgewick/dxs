import yaml
from pathlib import Path

import numpy as np
import pandas as pd

import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.nddata.utils import Cutout2D

import dxs.paths as paths

survey_config = paths.config_path / "survey_config.yaml"
with open(survey_config, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

stack_data = pd.read_csv(paths.header_data_path)

def get_hdu_name(stack_path, ccd, weight=False):
    weight = ".weight" if weight else ""
    return f"{stack_path.stem}_{ccd:02d}{weight}.fits"

def prepare_hdus(
    stack_path, 
    resize=True, edges=25.0,
    segmentation_path=None,
    normalise_exptime=True,
    subtract_bgr=True, bgr_size=32,
):
    stack_path = Path(stack_path) # convert to pathlib Path for niceness.
    results = []
    with fits.open(stack_path) as f:
        for ii, ccd in enumerate(survey_config["ccds"]):
            hdu_name = get_hdu_name(stack_path, ccd)
            hdu_path = paths.temp_hdus_path / hdu_name  
            results.append(hdu_path)          
            if hdu_path.exists():
                continue
            
            data = f[ccd].data.copy()
            header = f[ccd].header

            if resize:
                fwcs = wcs.WCS(header)
                xlen, ylen = header['NAXIS1'], header['NAXIS2']
                position = (ylen//2, xlen//2)
                size = (int(ylen-2*edges), int(xlen-2*edges))
                cutout = Cutout2D(data, position, size, wcs=fwcs)
                data_hdu = fits.PrimaryHDU(data=cutout.data,header=header)
                data_hdu.header.update(cutout.wcs.to_header())
            else:
                data_hdu = fits.PrimaryHDU(data=data,header=header)
            data_hdu.writeto(hdu_path, overwrite=True)    
    return results

def get_stack_data(field, tile, band, pointing=None):
    """
    
    """
    queries = [f"(field=='{field}')"]
    if tile is not None:
        if not isinstance(tile, list):
            tile = [tile]
        queries.append(f"(tile in @tile)")
    if band is not None:
        if not isinstance(band, list):
            band = [band]
        queries.append(f"(band in @band)")
    if pointing is not None:
        if not isinstance(pointing, list):
            pointing = [pointing]
        queries.append(f"(pointing in @pointing)")    
    query = "&".join(q for q in queries)
    return stack_data.query(query)

def calculate_mosaic_geometry(field, tile, ccds=None, factor=None, border=None):
    relevant_stacks = get_stack_data(field, tile, band=None)
    print(relevant_stacks)
    stack_list = [paths.input_data_path / f"{x}.fit" for x in relevant_stacks["filename"]]

    ccds = ccds or [0]
    ra_values = []
    dec_values = []
    for ii, stack_path in enumerate(stack_list):
        with fits.open(stack_path) as f:
            for ccd in ccds:
                fwcs = wcs.WCS(f[ccd].header)
                footprint = fwcs.calc_footprint()
                ra_values.extend(footprint[:,0])
                dec_values.extend(footprint[:,1])
    ra_limits = (np.min(ra_values), np.max(ra_values))
    dec_limits = (np.min(dec_values), np.max(dec_values))

    # center is easy.
    center = (np.mean(ra_limits), np.mean(dec_limits))
    
    # image size takes a bit more thought because of spherical things.
    cos_dec = np.cos(center[1]*np.pi / 180.)
    plate_factor = 3600. / survey_config["pixel_scale"]
    x_size = abs(ra_limits[1] - ra_limits[0]) * plate_factor * cos_dec
    y_size = abs(dec_limits[1] - dec_limits[0]) * plate_factor
    image_size = (int(x_size), int(y_size))
    if factor is not None:
        image_size = (image_size[0]*factor, image_size[1]*factor)
    if border is not None:
        image_size = (image_size[0]+border, image_size[1]+border)
    return center, image_size    

def add_keys(mosaic_path, data, hdu=0, verbose=False):
    with fits.open(mosaic_path, mode="update") as mosaic:
        for key, val in data.items():
            if verbose:
                print(f"Update {key} to {val}")
            mosaic[hdu].header[key.upper()] = val
        mosaic.flush()






