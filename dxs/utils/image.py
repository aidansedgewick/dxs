import os
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS

import dxs.paths as paths

#survey_config = paths.config_path / "survey_config.yaml"
#with open(survey_config, "r") as f:
#    survey_config = yaml.load(f, Loader=yaml.FullLoader)

#stack_data = pd.read_csv(paths.header_data_path)

import matplotlib.pyplot as plt

### bits related to survey area/randoms.

def uniform_sphere(ra_limits, dec_limits, size=1):
    """
    Get points randomly distributed on the surface of a (unit) sphere.
    Returns Nx2 numpy array with columns [ra, dec].
    Use this over astropy as astropy only does the whole sphere...

    Parameters
    ----------
    ra_limits
        right asc. limits to generate points in.
    dec_limits
        declination limits to generate points in.
    size
        number of points (default 1)
    """
    zlim = np.sin(np.pi * np.asarray(dec_limits) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size=size)
    DEC = (180. / np.pi) * np.arcsin(z)
    #DEC = DEClim[0] + (DEClim[1] - DEClim[0]) * np.random.random(size) # NO!
    RA = ra_limits[0] + (ra_limits[1] - ra_limits[0]) * np.random.random(size)
    return np.column_stack([RA, DEC])

def objects_in_coverage(image_path, ra, dec, minimum_coverage=0.0, hdu=0):
    with fits.open(image_path) as img:
        header = img[hdu].header
        fwcs = WCS(header)
        data = abs(img[hdu].data)
    ra = ra.flatten()
    dec = dec.flatten()
    coord = np.column_stack([ra, dec])
    pix = fwcs.wcs_world2pix( coord, 0 ).astype(int)
    #pix[:,0] = np.clip(pix[:,0], 0, header["NAXIS1"])
    #pix[:,1] = np.clip(pix[:,1], 0, header["NAXIS2"])
    mask = np.full(len(pix), False)
    xmask = (0 < pix[:,0]) & (pix[:,0] < header["NAXIS1"])
    ymask = (0 < pix[:,1]) & (pix[:,1] < header["NAXIS2"])
    pix_mask = xmask & ymask

    pix = pix[ pix_mask ]
    mask[ pix_mask ] = data[pix[:,1], pix[:,0]] > minimum_coverage
    return mask

def objects_in_multi_coverage(image_list, ra, dec, minimum_coverage=0.0, hdu=0):
    full_mask = np.full(len(ra), True)
    if not isinstance(minimum_coverage, list):
        minimum_coverage = [minimum_coverage] * len(image_list)
    if not isinstance(hdu, list):
        hdu = [hdu] * len(image_list)
    for ii, image_path in enumerate(image_list):
        mask = objects_in_coverage(
            image_path, ra, dec, minimum_coverage=minimum_coverage[ii], hdu=hdu[ii]
        )
        full_mask = full_mask * mask
    return full_mask

def calc_survey_area(image_list, ra_limits=None, dec_limits=None, N=100_000):
    if not isinstance(image_list, list):
        image_list = [image_list]
    if not all([ra_limits, dec_limits]):
        ra_values = []
        dec_values = []
        for image_path in image_list:
            with fits.open(image_path) as img:
                fwcs = WCS(img[0].header)
                footprint = fwcs.calc_footprint()
                ra_values.extend(footprint[:,0])
                dec_values.extend(footprint[:,1])
        ra_limits = (np.min(ra_values), np.max(ra_values))
        dec_limits = (np.min(dec_values), np.max(dec_values))

    area_box = calc_spherical_rectangle_area(ra_limits, dec_limits)
    randoms = uniform_sphere(ra_limits, dec_limits, size=N)
    survey_mask = objects_in_multi_coverage(image_list, randoms[:,0], randoms[:,1])
    factor = len(randoms[ survey_mask ]) / len(randoms)
    survey_area = factor * area_box
    return survey_area

def calc_spherical_rectangle_area(ra_limits, dec_limits):
    dra = ra_limits[1] - ra_limits[0]
    ddec = np.sin(dec_limits[1] * np.pi / 180.) - np.sin(dec_limits[0] * np.pi / 180.)
    area_box = dra * ddec * 180. / np.pi
    return area_box    

def mosaic_difference(path1, path2, save_path=None, show=True, header=1, hdu1=0, hdu2=0):
    path1 = Path(path1)
    path2 = Path(path2)
    if save_path is None:
        save_path = paths.temp_swarp_path / f"diff_{path1.stem}_{path2.stem}.fits"
    save_path = Path(save_path)
    with fits.open(path1) as mosaic1:
        data1 = mosaic1[hdu1].data
        header1 = mosaic1[hdu1].header
    with fits.open(path2) as mosaic2:
        data2 = mosaic2[hdu2].data
        header2 = mosaic2[hdu2].header
    if header==1:
        header = header1
    elif header==2:
        header = header2
    else:
        raise ValueError("Choose to keep first header (header=1), or second (header=2)")
    data = data1-data2
    output_hdu = fits.PrimaryHDU(data=data, header=header)
    output_hdu.writeto(save_path, overwrite=True)
   
    ds9_command = build_ds9_command([save_path, path1, path2])
    print(f"view image with \n    {ds9_command} &")

#def mosaic_quotient(path1, path2, save_path=None, header=1, hdu1=0, hdu2=0):
#    path1 = Path(path1)
#    if save_path is None:
#        save_path = paths.temp_swarp_path / f"diff_{path1.stem}_{path2.stem}.fits"
#    path2

def scale_mosaic(path, value, save_path=None, hdu=0, round_val=None):
    path = Path(path)
    save_path = Path(save_path)
    with fits.open(path) as mosaic:
        data = mosaic[hdu].data
        data = data*value
        if round_val is not None:
            data = np.around(data, decimals=round_val)
        header = mosaic[hdu].header
    output_hdu = fits.PrimaryHDU(data, header)
    output_hdu.writeto(save_path, overwrite=True)

def make_normalised_weight_map(weight_path, coverage_path, output_path):

    with fits.open(weight_path) as weight:
        #shape = weight[0].data.shape
        data = weight[0].data #.flatten()
        with fits.open(coverage_path) as coverage:
            cov = coverage[0].data.astype(int) #.flatten()
            cov_vals = np.unique(cov)
            for cval in cov_vals:
                print(cval)
                if cval < 1:
                    continue
                mask = (cov == cval)
                cdat = data[ mask ].flatten()
                med = np.median(cdat)
                data[ mask ] = data[ mask ] / med
        #data = data.reshape(shape)
        new_hdu = fits.PrimaryHDU(data=data, header=weight[0].header)
    new_hdu.writeto(output_path, overwrite=True)

### other

default_ds9_flags = ["single", "zscale", "cmap bb", "wcs skyformat degrees", "multiframe"]

def build_ds9_command(path_list, flags=None, relative=True):
    flags = flags or default_ds9_flags 
    if not isinstance(path_list, list):
        path_list = [path_list]
    if not isinstance(flags, list):
        flags = [flags]

    if relative:    
        print_path_list = []
        for path in path_list:
            path = Path(path)
            try:
                print_path_list.append( path.relative_to(Path.cwd()) )
            except:
                print_path_list.append( path )
    else:
        print_path_list = path_list    
    flag_str = " ".join(f"-{x}" for x in flags)
    path_str = " ".join(str(path) for path in print_path_list)
    cmd = f"ds9 {flag_str} {path_str}"
    return cmd
    




       
