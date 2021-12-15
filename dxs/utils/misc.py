import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord

from dxs import paths

logger = logging.getLogger("utils.misc")

class ModuleMissingError(Exception):
    pass

class AstropyFilter(logging.Filter):
    def filter(self, message):
        return not "FITSFixedWarning" in message.getMessage()

def haversine(c1, c2):
    dlat = c1.dec - c2.dec
    dlon = c1.ra - c2.ra
    d1 = np.sin(dlat / 2.)
    d2 = np.sin(dlon / 2.)

    arg = d1 * d1 + np.cos(c1.dec) * np.cos(c2.dec) * d2 * d2

    return 2. * np.arcsin( np.sqrt(arg) )

class GreatCircleArc:
    # https://math.stackexchange.com/questions/383711/parametric-equation-for-great-circle
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.d = haversine(c1, c2)
        #np.arccos( 
        #    np.sin(c1.dec) * np.sin(c2.dec) + np.cos(c1.dec) * np.cos(c2.dec) * np.cos(c1.ra - c2.ra)
        #)
        
    def get_waypoints(self, f):
        A = np.sin((1. - f) * self.d) / np.sin(self.d)
        B = np.sin(f * self.d) / np.sin(self.d)

        x = A * np.cos(self.c1.dec) * np.cos(self.c1.ra) + B * np.cos(self.c2.dec) * np.cos(self.c2.ra)
        y = A * np.cos(self.c1.dec) * np.sin(self.c1.ra) + B * np.cos(self.c2.dec) * np.sin(self.c2.ra)
        z = A * np.sin(self.c1.dec)  + B * np.sin(self.c2.dec)

        w_dec = np.arctan2(z, np.sqrt(x*x + y*y))
        w_ra = np.arctan2(y, x)

        return SkyCoord(w_ra, w_dec)

module_suggestions = {
    "swarp": "try   module load swarp",
    "sex": "try   module load sextractor",
    "stilts": "try   module load starlink",
}

def check_modules(*modules):
    missing_modules = []
    present_modules = {}
    for module in modules:
        # check if it exists
        exe_path = shutil.which(module)
        if exe_path is None:
            missing_modules.append(module)
        else:
            logger.info(f"Using {module} at {exe_path}")
    if len(missing_modules) > 0:
        suggestions = []
        for module in missing_modules:
            suggestion = module_suggestions.get(module, "no suggestions available")
            suggestions.append(f"Missing '{module}': {suggestion}")
        suggestion_string = "\n".join(x for x in suggestions)
        raise ModuleMissingError(
            f"\033[31mMissing modules\033[0m {missing_modules}.\n{suggestion_string}"
        )

def format_flags(config, capitalise=True, float_precision=6):
    """
    Capitalise config keys.
    Convert pathlib.Path types to str
    Convert int to str
    Convert float to six decimal place string
    Convert tuple of int to comma-separated string
    Convert tuple of float to comma-sep string with six decimal places.
    Confert SkyCoord to commasep float-format "ra,dec"
    Do nothing to string.
    """
    formatted_config = {}
    for param, value in config.items():
        if capitalise:
            key = param.upper()
        else:
            key = param
        if value is None:
            value = "None"
        if isinstance(value, str):
            formatted_config[key] = value
        elif isinstance(value, Path):
            formatted_config[key] = str(value) # if it's a pathlib Path, fix it for JSON later.
        elif isinstance(value, float):
            formatted_config[key] = f"{value:.{float_precision}f}"
        elif isinstance(value, int):
            formatted_config[key] = str(value)
        elif isinstance(value, SkyCoord):
            formatted_config[key] = f"{value.ra.value:.{float_precision}f},{value.dec.value:.{float_precision}f}"
        elif isinstance(value, tuple):
            if all(isinstance(x, int) for x in value):
                formatted_config[key] = ','.join(f"{x}" for x in value)
            elif all(isinstance(x, float) for x in value):
                formatted_config[key] = ','.join(f"{x:.{float_precision}f}" for x in value)
            else:
                mixed_types = [type(x) for x in value]
                raise ValueError(f"Don't know how to format mixed types {mixed_types} - check element types?")
        else:
            raise ValueError(f"Don't know how to format type {type(value)}")
    return formatted_config

def create_file_backups(file_list: List[Path], temp_dir: Path):
    if not isinstance(file_list , list):
        file_list = [file_list]
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    new_paths = []
    for file_path in file_list:
        file_path = Path(file_path)
        new_file_path = temp_dir / file_path.name
        if file_path == new_file_path:
            new_file_path = temp_dir / f"copy_{filepath.name}"
        new_paths.append(new_file_path)
        logger.info(f"backup {file_path} to {new_file_path}")
        shutil.copy2(file_path, new_file_path)
    return new_paths

def get_git_info():
    """
    find git branch name and commit ID
    """
    dxs_git = Path(__file__).parent.parent.parent / ".git"
    branch_cmd = f"git --git-dir {dxs_git} rev-parse --abbrev-ref HEAD".split()
    branch = "unavailable"
    try:
        branch = (
            subprocess.run(branch_cmd, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
    except Exception as e:
        print("Couldn't find git branch", e)
    local_SHA_cmd = f'git --git-dir {dxs_git} log -n 1 --format="%h"'.split()
    local_SHA = "unavailable"
    try:
        local_SHA = (
            subprocess.run(local_SHA_cmd, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
    except Exception as e:
        print("could not record local git SHA", e)
    return branch, local_SHA

def print_header(string, edge="###", newlines=3, return_string=False):
    try:
        columns = shutil.get_terminal_size((80,20)).columns
    except:
        columns = 80
    N = int(columns - len(string) - 2 - 2*len(edge)) // 2
    header_str = (
        "\n"*newlines + edge + N*"=" + f" {string} " + N*"=" + edge + "\n"
    )
    if return_string:
        return header_str
    print(header_str)

def tile_parser(s):
    init_list = s.split(",")
    out_list = []
    for x in init_list:
        if "-" in x:
            x1, x2 = x.split("-")
            out_list.extend([ x for x in range(int(x1), int(x2) + 1) ])
        else:
            out_list.append(int(x))
    return out_list

def tile_encoder(l):
    if isinstance(l, str):
        l = tile_parser(l)
    tot = sum([2 ** (x-1) for x in l])
    tile_code = "{:03x}".format(tot)
    return tile_code
    
def tile_decoder(s):
    value = int(s, 16)
    b = "{:b}".format(value)
    tiles = [ii+1 for ii, bit in enumerate(b[::-1]) if bool(int(bit))]
    return tiles
    

###========= helpful common(?) 1d-array problems.

def calc_mids(arr):
    return 0.5*(arr[:-1]+arr[1:])

def calc_widths(arr):
    return arr[1:] - arr[:-1]

def calc_range(arr, axis=None):
    return (arr.min(axis=axis), arr.max(axis=axis))

def mad(arr):
    med = np.median(arr)
    dev = arr - med
    mad_val = np.median(abs(dev))
    return mad_val






