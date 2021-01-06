import json
import yaml
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from astromatic_wrapper.api import Astromatic

from dxs.utils.image import (
    prepare_hdus, get_stack_data, calculate_mosaic_geometry, add_keys
)
from dxs.utils.misc import check_modules, format_astromatic_flags

from dxs import paths

survey_config = paths.config_path / "survey_config.yaml"
with open(survey_config, "r") as f:
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

    @staticmethod
    def get_mosaic_stem(field, tile, band, prefix=None):
        if prefix is None:
            prefix = ""
        return f"{prefix}{field}{tile:02d}{band}"

    @staticmethod
    def get_mosaic_dir(field, tile, band):
        return paths.mosaics_path / MosaicBuilder.get_mosaic_stem(field, tile, band)

    def build(self):
        self.prepare_all_hdus()
        config = self.build_swarp_config()
        config.update(self.swarp_config)
        config = format_astromatic_flags(config)
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

    def add_extra_keys(self,):
        data = {}
        data["seeing"] = self.relevant_stacks["seeing"].median()
        data["magzpt"] = self.relevant_stacks["magzpt"].median()
        add_keys(self.mosaic_path, data, hdu=0, verbose=True)

if __name__ == "__main__":
    builder = MosaicBuilder("EN", 4, "J", n_cpus=8)
    builder.build()
    builder.add_extra_keys()









