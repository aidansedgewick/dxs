import logging
import sys
import yaml
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


from dxs.utils.misc import tile_parser, tile_encoder, print_header
from dxs.utils.image import build_ds9_command

from dxs import paths

logger = logging.getLogger("field_mask")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

def field_mask_pipeline(
    field, tiles, band, 
    output_code=0, 
    input_extension=".cov.good_cov.fits", 
    output_suffix="good_cov_mask",
    resolution=1.0,
    require_all=True,
):

    output_stem = paths.get_mosaic_stem(field, output_code, band, suffix=output_suffix)
    output_path = paths.masks_path / f"{output_stem}.fits"
    print(output_path)


    input_paths = []
    for tile in tiles:
        p = paths.get_mosaic_path(field, tile, band)
        use_path = p.with_suffix(input_extension)
        if not use_path.exists():
            logger.info(f"{use_path} not found")
            if require_all:
                raise ValueError(f"{use_path} not found")
        else:
            input_paths.append(use_path)
    if len(input_paths) == 0:
        logger.info("no mosaics - continue")
        return None
    
    input_list = []
    for mosaic_path in input_paths:
        with fits.open(mosaic_path) as mos:
            t = (mos[0].data, WCS(mos[0].header))
            input_list.append(t)
        
    wcs_out, shape_out = mosaicking.find_optimal_celestial_wcs(
        input_list, resolution = resolution * u.arcsec
    )
    logger.info("starting reprojection")
    output_array, footprint = mosaicking.reproject_and_coadd(
        input_list, wcs_out, shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="mean"
    )
    logger.info("finished reprojection")
    header = wcs_out.to_header()
    output_hdu = fits.PrimaryHDU(data=output_array, header=header)
    output_hdu.writeto(output_path, overwrite=True)

    #output_mask_paths.append(output_path)
    logger.info(f"written {output_path}")

    return output_path
    

if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("fields")
    parser.add_argument("bands")
    parser.add_argument("--tiles", default=None)
    parser.add_argument("--resolution", default=1.0, type=float)
    parser.add_argument("--input-extension", default=".cov.good_cov.fits")
    parser.add_argument("--output-suffix", default="_good_cov_mask")
    parser.add_argument("--n_cpus", default=None, type=int)
    parser.add_argument("--require-all", default=True, action="store_false")
    args = parser.parse_args()

    fields = args.fields.split(",")
    print(fields)
    if args.tiles is not None:
        tspl = args.tiles.split()
        if len(tspl) != len(fields):
            raise ValueError("number of tile-strs must be equal to len(fields)")
        tile_combos = []
        output_codes = []
        for ts in tspl:
            combo = tile_parser(ts) # converts eg. "1,2,7-10" to [1,2,7,8,9,10]
            output_codes.append( tile_encoder(combo) )
            tile_combos.append( combo )

    else:
        tile_combos = []
        for field in fields:
            combo = survey_config["merge"]["default_tiles"].get(field, None)
            if combo is None:
                raise ValueError(f"no default tiles in survey_config.merge for {field}")
            tile_combos.append(combo)
        output_codes = [0 for field in fields]
    bands = args.bands.split(",")    

    for field, tile_combo, output_code in zip(fields, tile_combos, output_codes):
        for band in bands:
            print_header(f"{field} {output_code} {band}")
            print(f"use tiles {tile_combo}")

            field_mask_pipeline(
                field, tile_combo, band, 
                output_code=output_code,
                input_extension=args.input_extension,
                output_suffix=args.output_suffix,
                resolution=args.resolution,
                require_all=args.require_all,
            )






