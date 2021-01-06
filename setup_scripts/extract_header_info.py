"""
20210104
"""

import time
import sys
import os
import glob
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import Angle

from dxs import paths

stack_paths = glob.glob(str(paths.input_data_path / "*.fit"))

print(f"Extracting data from {len(stack_paths)} stacks")

bands = {"J": 1, "H": 2, "K": 3}
fields = {"SA": 1, "LH": 2, "EN": 3, "XM": 4}
field_longnames = {'SA22': 1, 'Lockman': 2, 'ElaisN1': 3, 'XMM-LSS': 4}
pointings = {"0_0": 1, "0_1": 2, "1_0": 3, "1_1": 4}
ccds = [1,2,3,4]

def extract_data_from_primary_header(header, bad_header=False):
    data = {}
    object_tag = header["OBJECT"]
    # list[0] == "DXS" every row - not so interesting!
    _, longname, field_tile, tag_pointing, tag_band = object_tag.split(" ")


    data["field"] = field_tile[:2]
    data["tile"] = field_tile[2:]
    if bad_header:
        data["tile"] = 9
    data["band"] = header["FILTER"]
    data["pointing"] = tag_pointing[:3]
    data["obj_band"] = tag_band[0]
    data["exptime"] = header["EXP_TIME"]
    data["MJD"] = header["MJD-OBS"]

    return data

def extract_data_from_ccd_header(header, bad_header=False, suffix=None):
    data = {}
    suffix = f"_{suffix}" if suffix is not None else ""

    try:
        magzpt = header["MAGZPT"]
    except:
        magzpt = -1.0
    data["magzpt" + suffix] = magzpt
    try:
        seeing = header["SEEING"]*header["PIXLSIZE"]/2.0
    except:
        seeing = -1.0
    data["seeing" + suffix] = seeing

    return data

def extract_header_data(f, bad_header):
    header_data = extract_data_from_primary_header(
        f[0].header, bad_header=bad_header
    )

    header_1_data = extract_data_from_ccd_header(f[1].header)
    header_data.update(header_1_data)
    for jj, ccd in enumerate(ccds):
        ccd_header_data = extract_data_from_ccd_header(f[ccd].header, suffix=ccd)
        header_data.update(ccd_header_data)
    return header_data

if __name__ == "__main__":
    corrupt_fits = []
    header_data_list = []
    bad_band = 0

    bad_headers_path = paths.config_path / "bad_headers.csv"
    if bad_headers_path.exists():
        bad_headers_df = pd.read_csv(bad_headers_path)
        bad_headers = [Path(x).stem for x in bad_headers_df["fname"]]
    else:
        bad_headers = []

    for ii, stack_path in enumerate(stack_paths):
        header_data = None
        print(f"Extracting data from {ii}/{len(stack_paths)} stacks")
        stack_path = Path(stack_path)
        try:
            bad_header = (stack_path.stem in bad_headers)
            with fits.open(stack_path) as f:
                header_data = extract_header_data(f, bad_header)
            if header_data["band"] != header_data["obj_band"]:
                bad_band += 1
            header_data["filename"] = stack_path.stem
            header_data["index"] = ii
            header_data_list.append(header_data)
        except:
            corrupt_fits.append(stack_path.name + "\n")
        del f # Do this manually?!
        
                

    dataframe = pd.DataFrame(header_data_list)

    header_data_path = paths.header_data_path # defaults to paths.config_path / "dxs_header_data.csv"
    dataframe.to_csv(header_data_path)

    corrupt_frames_path = paths.config_path / "dxs_corrupt_fits.txt"

    print(f"there are {bad_band} mislablled filters")

    print(f"there are {len(corrupt_fits)} corrupt frames, named in \n    {corrupt_frames_path}")

    with open(corrupt_frames_path, "w+") as f:
        f.writelines(corrupt_fits)



