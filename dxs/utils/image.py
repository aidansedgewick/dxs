import logging
import os
import tqdm
import yaml
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from scipy.ndimage import binary_dilation

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord 
from astropy.coordinates import concatenate as skycoord_concatenate
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from regions import PolygonPixelRegion, PolygonSkyRegion, PixCoord, read_ds9

from dxs import paths

logger = logging.getLogger("image_utils")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)


###==================== mosaics ======================###

def build_mosaic_wcs(
    center, size: Tuple[int, int], pixel_scale: float, proj="TAN", **kwargs
):
    """
    size as (XPIX, YPIX) - because WCS of rules...
    pixel_scale in ARCSEC.
    """
    """
    w = WCS(naxis=2)
    w.wcs.crpix = [size[0]/2, size[1]/2]
    w.wcs.cdelt = [-pixel_scale / 3600., pixel_scale / 3600.]
    if isinstance(center, SkyCoord):
        w.wcs.crval = [center.ra.value, center.dec.value]
    else:
        w.wcs.crval = [center[0], center[1]]
    w.wcs.ctype = [
        "RA" + "-" * (6-len(proj)) + proj, "DEC" + "-" * (5-len(proj)) + proj
    ]
    w.naxis1 = size[0]
    w.naxis2 = size[1]"""
    
    cr1 = center.ra.value if isinstance(center, SkyCoord) else center[0]
    cr2 = center.dec.value if isinstance(center, SkyCoord) else center[1]

    wcs_dict = {
        "CTYPE1": "RA" + "-" * (6-len(proj)) + proj,
        "CUNIT1": "deg",
        "CDELT1": -pixel_scale / 3600.,
        "CRPIX1": size[0] / 2,
        "CRVAL1": cr1,
        "NAXIS1": size[0],
        "CTYPE2": "DEC" + "-" * (5-len(proj)) + proj,
        "CUNIT2": "deg",
        "CDELT2": pixel_scale / 3600.,
        "CRPIX2": size[1] / 2,
        "CRVAL2": cr2,
        "NAXIS2": size[1],
        **kwargs,
    }
    w = WCS(wcs_dict)
    w.fix()
    return w


###============= bits related to survey area/randoms. =============###

def calc_spherical_rectangle_area(ra_limits, dec_limits):
    dra = ra_limits[1] - ra_limits[0] # could * by pi / 180 here, but would have to undo in last line...
    ddec = np.sin(dec_limits[1] * np.pi / 180.) - np.sin(dec_limits[0] * np.pi / 180.)
    area_box = dra * ddec * 180. / np.pi # if radians in first line, would * (180 / pi)**2
    return area_box    


def uniform_sphere(
    ra_limits, dec_limits, size: int = 1, density: float = None,
):
    """
    Get points randomly distributed on the surface of a (unit) sphere.
    Returns Nx2 numpy array with columns [ra, dec].
    Use this over astropy, as astropy only does the whole sphere...

    Parameters
    ----------
    ra_limits
        right asc. limits to generate points in.
    dec_limits
        declination limits to generate points in.
    size
        number of points (default 1)
    density
        if provided, generate this many points per sq. deg.
    """

    if density is not None:
        area_box = calc_spherical_rectangle_area(ra_limits, dec_limits)
        size = int(area_box * density)
    logger.info(f"generate {size:,} randoms")
    zlim = np.sin(np.pi * np.asarray(dec_limits) / 180.) # sin(dec) is uniformly distributed.
    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size=size)
    DEC = (180. / np.pi) * np.arcsin(z)
    #DEC = dec_limits[0] + (dec_limits[1] - dec_limits[0]) * np.random.random(size) # NO!
    RA = ra_limits[0] + (ra_limits[1] - ra_limits[0]) * np.random.random(size)
    return np.column_stack([RA, DEC])

def single_image_coverage(
    image_path, ra, dec, minimum_coverage=0.0, absolute_value=True, hdu=0
):
    with fits.open(image_path) as img:
        header = img[hdu].header
        fwcs = WCS(header)
        if absolute_value:
            data = abs(img[hdu].data)
    ra = ra.flatten()
    dec = dec.flatten()
    coord = np.column_stack([ra, dec])
    pix = fwcs.wcs_world2pix( coord, 0 ).astype(int)
    mask = np.full(len(pix), False)
    xmask = (0 < pix[:,0]) & (pix[:,0] < header["NAXIS1"])
    ymask = (0 < pix[:,1]) & (pix[:,1] < header["NAXIS2"])
    pix_mask = xmask & ymask

    pix = pix[ pix_mask ]
    mask[ pix_mask ] = data[pix[:,1], pix[:,0]] > minimum_coverage
    return mask

def objects_in_coverage(
    image_list, ra, dec, minimum_coverage=0.0, absolute_value=True, hdu=0
):
    """
    Given a [list of] image(s) and ra, dec coords (in degrees),
    find out if each coord is "in the data" in each of the images - ie, is the value
    of [all of] the images greater than some minimum value at each ra, dec.
    Return a boolean array of len(ra)
    
    Parameters
    ----------
    image_list
        list of fits images
    ra
        length N array
    dec
        length N array
    minimum_coverage
        what value(s) do the data need to be greater than at the coordinate?
        can provide single value, or list of len(image_list)
    absolute_value
        take the absolute value for the data? bool (or list of bool, len(image_list))
    hdu
        hdu to use in each image (single value, or list len(image_list)).
    """
    # TODO use skycoord?

    if len(ra) != len(dec):
        lra, dec = len(ra), len(dec)
        raise ValueError(f"len(ra) ({lra}) not equal to len(dec) ({ldec})")
    
    full_mask = np.full(len(ra), True)
    if not isinstance(image_list, list):
        image_list = [image_list]
    if not isinstance(minimum_coverage, list):
        minimum_coverage = [minimum_coverage] * len(image_list)
    if not isinstance(absolute_value, list):
        absolute_value = [absolute_value] * len(image_list)
    if not isinstance(hdu, list):
        hdu = [hdu] * len(image_list)
    for ii, image_path in enumerate(image_list):
        mask = single_image_coverage(
            image_path, ra, dec, 
            minimum_coverage=minimum_coverage[ii], 
            absolute_value=absolute_value[ii],
            hdu=hdu[ii],
        )
        full_mask = full_mask * mask
    return full_mask

def calc_survey_area(
    image_list, ra_limits=None, dec_limits=None, density=1e4, return_randoms=False
):
    """
    Monte-carlo to find survey area from list of fits files.
    Calculate the area of the spherical rectangle which bounds all of the images.
    Generate N random points (on the surface of a sphere), 
    where N is density * area_rectangle.
    See what fraction of them fall within the data,
    Return fraction * area_recangle. 

    Paramters
    ---------
    image_list
        list of fits files to calculate survey area from.
    ra_limits
        optionally provide a tuple of min/max of the survey ra - else calculated
        from the footprint of the fits files - values in degrees!
    dec_limits
        optionally provide a tuple of min/max of the survey dec - else calculated
        from the footprint of the fits files - values in degrees!
    density
        number of points per sq. deg. used for monte carlo.
    """
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
    randoms = uniform_sphere(ra_limits, dec_limits, density=density)
    survey_mask = objects_in_coverage(image_list, randoms[:,0], randoms[:,1])
    factor = len(randoms[ survey_mask ]) / len(randoms)
    survey_area = factor * area_box
    if return_randoms:
        return survey_area, randoms[ survey_mask ]
    return survey_area
   

###================== coverage mosaics ==================###

def make_good_coverage_map(
    coverage_map_path, 
    minimum_coverage,
    output_path=None,
    weight_map_path=None, 
    minimum_weight=0.8, # ??? a complete guess.
    dilation_structure=None, 
    dilation_iterations=50,
    hdu=0
):
    """
    modify an image 3 steps: [sq. brackets show defaults]
        1:
            set any pixels less than minimum_coverage (required argument).
        2:
            set any pixels to zero if corresponding weight_map pixel 
            are less than minimimum_weight=[0.8] (if weight_map_path) is provided.
        3.
            dilate the zero regions with scipy's binary dilation
                -- but NOT the outside 2 pixels. we leave these in case of swarp error, etc.
            kwargs for this are dilation_structure=[None], dilation_iterations=[50].
            
    """

    coverage_map_path = Path(coverage_map_path)
    logger.info(f"modify {coverage_map_path.name}")
    with fits.open(coverage_map_path) as f:
        data = f[hdu].data #.astype(int)
        data = np.around(data, decimals=0)
        fwcs = WCS(f[hdu].header)
        data[data < minimum_coverage] = 0.
        if weight_map_path is not None:
            with fits.open(weight_map_path) as weight:
                wdat = weight[0].data
                if wdat.shape != data.shape:
                    raise ValueError(f"weight shape ({wdat.shape}) not the same as input shape ({data.shape})")
                data[ wdat < minimum_weight ] = 0.
        if dilation_iterations > 0:
            data = dilate_zero_regions(
                data, structure=dilation_structure, iterations=dilation_iterations
            )    
        good_coverage = fits.PrimaryHDU(data=data, header=f[hdu].header)
        if output_path is None:
            output_path = coverage_map_path.with_suffix(".good_cov.fits")
        good_coverage.writeto(output_path, overwrite=True)

def dilate_zero_regions(data, structure=None, iterations=50):
    mask = (data[2:-2, 2:-2] == 0)
    old_zeros = mask.sum()    
    mask = binary_dilation(
        mask, structure=structure, iterations=iterations
    )
    new_zeros = mask.sum()
    data[2:-2, 2:-2][ mask ] = 0.
    logger.info(f"mask further {new_zeros-old_zeros:,} pix")
    return data

###=================== bright star masking ===============###

def mask_regions_in_mosaic(
    mosaic_path,
    region_list,
    expand=1000,
    output_path=None,
    skip_checks=False,
):
    logger.info("start masking regions")
    if len(region_list) == 0:
        logger.info("No regions to mask.")
        return None
    with fits.open(mosaic_path) as f:
        data = f[0].data.copy()
        header = f[0].header
        wcs = WCS(header)
        ylen, xlen = data.shape
    if not skip_checks:
        expanded_wcs = Cutout2D(
            data, position=(xlen//2, ylen//2), size=(xlen+expand, ylen+expand), wcs=wcs,
            mode="partial", fill_value=1
        ).wcs
        
        footprint = wcs.calc_footprint()    
        footprint_region = PolygonSkyRegion(SkyCoord(footprint, unit="degree")).to_pixel(wcs)

        logger.info(f"find relevant regions from {len(region_list)}")
        # ugly ugly blergh - but faster than concat...
        ra_vals = np.array([region.center.ra.degree for region in region_list])
        dec_vals = np.array([region.center.dec.degree for region in region_list])
        sky_coord = SkyCoord(ra=ra_vals, dec=dec_vals, unit="degree")
        contained_by = sky_coord.contained_by(expanded_wcs)
        region_list = [region for region, contained in zip(region_list, contained_by) if contained]
    else:
        logger.info("skip checking for relevant regions")


    logger.info(f"masking {len(region_list)} regions")
    for sky_region in tqdm.tqdm(region_list):
        pix_region = sky_region.to_pixel(wcs)
        bbox = pix_region.bounding_box
        if bbox.ixmax < 0 or bbox.ixmin > xlen or bbox.iymin > ylen or bbox.iymax < 0:
            continue

        mask = pix_region.to_mask(mode="center") # is the center of each pixel in the mask?
        slicer = mask.bbox.slices

        mask_data = mask.data
        if data[slicer].shape != mask_data.shape:
            xmin, xmax, ymin, ymax = mask.bbox.extent
            data_xmin, data_xmax = max(int(xmin), 0), min(int(xmax), xlen-1)
            data_ymin, data_ymax = max(int(ymin), 0), min(int(ymax), ylen-1) 

            mask_xmin = 0 - int(xmin) if xmin < 0 else 0
            mask_ymin = 0 - int(ymin) if ymin < 0 else 0
            mask_xmax = mask_xmin + (data_xmax - data_xmin)
            mask_ymax = mask_ymin + (data_ymax - data_ymin)
            try:
                slicer = np.s_[data_ymin:data_ymax, data_xmin:data_xmax]
                mask_data = mask_data[mask_ymin:mask_ymax, mask_xmin:mask_xmax]
            except:
                pass
            if data[slicer].shape != mask_data.shape:
                continue

        bool_mask = np.invert(mask_data.astype(bool)).astype(int)
        data[slicer] = data[slicer] * bool_mask

    if output_path is None:
        output_path = mosaic_path
    output_hdu = fits.PrimaryHDU(data=data, header=header)
    output_hdu.writeto(output_path, overwrite=True)


###=================== compare mosaics ===================###

def mosaic_compare(
    path1, path2, func="diff", save_path=None, show=True, header=1, hdu1=0, hdu2=0
):
    path1 = Path(path1)
    path2 = Path(path2)
    if save_path is None:
        save_path = paths.scratch_swarp_path / f"{func}_{path1.stem}_{path2.stem}.fits"
    save_path = Path(save_path)
    with fits.open(path1) as mosaic1:
        data1 = mosaic1[hdu1].data
        header1 = mosaic1[hdu1].header
    with fits.open(path2) as mosaic2:
        data2 = mosaic2[hdu2].data
        header2 = mosaic2[hdu2].header
    try:
        out_header = [header1, header2][header-1]
    except:
        raise ValueError("Choose to keep first header (header=1), or second (header=2)")
    if func in ["diff", "difference"]:
        data = np.subtract(data1, data2)
    elif func in ["quot", "quotient"]:
        data = np.divide(data1, data2)
    else:
        raise ValueError("func=  'quot' or 'diff'")
    output_hdu = fits.PrimaryHDU(data=data, header=out_header)
    output_hdu.writeto(save_path, overwrite=True)
   
    ds9_command = build_ds9_command(save_path, path1, path2)
    print(f"view image with \n    {ds9_command} &")

def mosaic_quotient(path1, path2, save_path=None, header=1, hdu1=0, hdu2=0):
    path1 = Path(path1)
    path2 = Path(path2)
    if save_path is None:
        save_path = paths.scratch_swarp_path / f"quot_{path1.stem}_{path2.stem}.fits"
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
    data = data1/data2
    output_hdu = fits.PrimaryHDU(data=data, header=header)
    output_hdu.writeto(save_path, overwrite=True)
   
    ds9_command = build_ds9_command(save_path, path1, path2)
    print(f"view image with \n    {ds9_command} &")

def scale_mosaic(path, value, save_path=None, hdu=0, round_val=None):
    path = Path(path)
    save_path = Path(save_path)
    logger.info("scaling mosaic")
    with fits.open(path) as mosaic:
        data = mosaic[hdu].data
        data = data*value
        if round_val is not None:
            data = np.around(data, decimals=round_val)
        header = mosaic[hdu].header
    output_hdu = fits.PrimaryHDU(data, header)
    output_hdu.writeto(save_path, overwrite=True)   

### other

default_ds9_flags = [
    "single", "zscale", "cmap bb", "wcs skyformat degrees", "multiframe", "lock frame wcs"    
]

def build_ds9_command(*mosaic_paths, flags=None, relative=True):
    flags = flags or default_ds9_flags
    path_list = []
    for x in mosaic_paths:
        if isinstance(x, tuple) or isinstance(x, list):
            path_list.extend(x)
        else:
            path_list.append(x)
    if not isinstance(flags, list):
        flags = [flags]

    if relative:    
        print_path_list = []
        for path in mosaic_paths:
            path = Path(path)
            try:
                print_path_list.append( path.relative_to(Path.cwd()) )
            except:
                print_path_list.append( path )
    else:
        print_path_list = mosaic_paths   
    flag_str = " ".join(f"-{x}" for x in flags)
    path_str = " ".join(str(path) for path in print_path_list)
    cmd = f"ds9 {flag_str} {path_str}"
    return cmd
    




       
