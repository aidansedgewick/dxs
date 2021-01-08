import json
import yaml
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from astromatic_wrapper.api import Astromatic

#from dxs.utils.image import (
#    prepare_hdus, get_stack_data, calculate_mosaic_geometry, add_keys
#)
from dxs.utils.misc import check_modules, format_flags

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

stack_data = pd.read_csv(paths.header_data_path)

class MosaicBuilder:

    """
    Class for building mosiacs. Calls SWarp.
    """

    def __init__(
        self, field: str, tile: int, band: str, prefix: str=None,
        mosaic_stem=None, mosaic_dir=None, n_cpus=None, 
        swarp_config=None, swarp_config_path=None
    ):
        check_modules("swarp") # do we have swarp available?
        self.field = field
        self.tile = tile
        self.band = band
        
        self.mosaic_stem = mosaic_stem or self.get_mosaic_stem(field, tile, band, prefix=prefix)
        if mosaic_dir is not None:
            mosaic_dir = Path(mosaic_dir)
        self.mosaic_dir = mosaic_dir or self.get_mosaic_dir(field, tile, band)
        self.mosaic_dir.mkdir(exist_ok=True, parents=True)
        self.mosaic_path = self.mosaic_dir / f"{self.mosaic_stem}.fits"

        self.swarp_list_path  = self.mosaic_dir / "swarp_list.txt"
        self.swarp_run_parameters_path = self.mosaic_dir / "swarp_run_parameters.json"
        self.swarp_config = swarp_config or {}
        self.swarp_config_file = swarp_config_path or paths.config_path / "swarp/mosaic.swarp"

        self.relevant_stacks = get_stack_data(self.field, self.tile, self.band)

        self.n_cpus = n_cpus

    def build(self):
        self.prepare_all_hdus()
        config = self.build_swarp_config()
        config.update(self.swarp_config)
        config = format_flags(config)
        self.swarp = Astromatic(
            "SWarp", 
            str(paths.temp_swarp_path), # I think this is ignored anyway?!
            config=config, 
            config_file=str(self.swarp_config_file),
            store_output=True,
        )
        swarp_list_name = "@"+str(self.swarp_list_path)
        kwargs = self.swarp.run(swarp_list_name)
        print(kwargs)
        with open(self.swarp_run_parameters_path, "w+") as f:
            json.dump(kwargs, f, indent=2)
        
    def prepare_all_hdus(self, stack_list=None):
        if stack_list is None:
            stack_list = self.get_stack_list()    
        if self.n_cpus is None:    
            hdu_list = []        
            for stack_path in stack_list:
                results = prepare_hdus(stack_path)
                hdu_list.extend(results)
        else:
            with Pool(self.n_cpus) as pool:
                results = pool.map(prepare_hdus, stack_list)
                hdu_list = [p for result in results for p in result]
        for hdu_path in hdu_list:
            assert hdu_path.exists()
        with open(self.swarp_list_path, "w+") as f:
            f.writelines([str(hdu_path)+"\n" for hdu_path in hdu_list])

    def get_stack_list(self):
        stack_list = [paths.input_data_path / f"{x}.fit" for x in self.relevant_stacks["filename"]]
        return stack_list
                
    def build_swarp_config(self):
        config = {}
        config["imageout_name"] = self.mosaic_path
        weightout_name = self.mosaic_dir / f"{self.mosaic_stem}.weight.fits"
        config["weightout_name"] = weightout_name
        config["resample_dir"] = paths.temp_swarp_path
        center, size = calculate_mosaic_geometry(
            self.field, self.tile, ccds=survey_config["ccds"]
        )
        config["center_type"] = "MANUAL"
        config["center"] = center #f"{center[0]:.6f},{center[1]:.6f}"
        config["image_size"] = size #f"{size[0]},{size[1]}"
        config["pixelscale_type"] = "MANUAL"
        config["pixel_scale"] = survey_config["pixel_scale"] #f"{pixel_scale:.6f}"        
        config["nthreads"] = self.n_cpus
        return config

    def add_extra_keys(self, normalise_exptime=False):
        data = {}
        data["seeing"] = self.relevant_stacks["seeing"].median()
        if normalise_exptime:
            data["magzpt"] = self.relevant_stacks["magzpt"].median()
        else:
            exptime_factor = 2.5*np.log10(self.relevant_stacks["exptime"])
            magzpt_col = self.relevant_stacks["magzpt"] + exptime_factor
            data["magzpt"] = magzpt_col.median() # works as still pd.Series...
        add_keys(self.mosaic_path, data, hdu=0, verbose=True)


def get_mosaic_stem(field, tile, band, prefix=None):
    if prefix is None:
        prefix = ""
    return f"{prefix}{field}{tile:02d}{band}"

def get_mosaic_dir(field, tile, band):
    return paths.mosaics_path / get_mosaic_stem(field, tile, band)

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

def prepare_hdus(
    stack_path, 
    resize=True, edges=25.0,
    segmentation_path=None,
    normalise_exptime=True,
    subtract_bgr=True, bgr_size=32,
):
    stack_path = Path(stack_path) # convert to pathlib Path for niceness.
    results = []

    if segmentation_path is not None:
        mask

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

def get_hdu_name(stack_path, ccd, weight=False):
    weight = ".weight" if weight else ""
    return f"{stack_path.stem}_{ccd:02d}{weight}.fits"

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

if __name__ == "__main__":
    builder = MosaicBuilder("EN", 4, "J", n_cpus=8)
    builder.build()
    builder.add_extra_keys()









