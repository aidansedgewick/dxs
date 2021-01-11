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

survey_config = paths.config_path / "survey_config.yaml"
with open(survey_config, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

stack_data = pd.read_csv(paths.header_data_path)
default_ds9_flags = ["single", "zscale", "cmap bb", "wcs skyformat degrees", "multiframe"]


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


def build_ds9_command(paths, flags=None, relative=True):
    flags = flags or default_ds9_flags 
    if not isinstance(paths, list):
        paths = [paths]
    if not isinstance(flags, list):
        flags = [flags]

    if relative:    
        print_paths = []
        for path in paths:
            path = Path(path)
            try:
                print_paths.append( path.relative_to(Path.cwd()) )
            except:
                print_paths.append( path )
    else:
        print_paths = paths    
    flag_str = " ".join(f"-{x}" for x in flags)
    path_str = " ".join(str(path) for path in print_paths)
    cmd = f"ds9 {flag_str} {path_str}"
    return cmd
    




       
