"""
20210104
"""

import json
import time
import sys
import os
import glob
import gc
import yaml
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy import units as u
from astropy.coordinates import Angle

import dxs.paths as paths # don't use from dxs, as this goes through __init__

stack_paths = glob.glob(str(paths.stack_data_path / "*.fit"))
stack_paths = sorted(stack_paths)

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

def extract_data_from_primary_header(header, stack_name=""):
    data = {"rejected": False, "rejection_reason": None}
    object_tag = header["OBJECT"]
    # list[0] == "DXS" every row - not so interesting!

    data["exptime"] = header["EXP_TIME"]
    data["MJD"] = header["MJD-OBS"]
    data["project"] = header["PROJECT"]
    data["band"] = header["FILTER"]

    try:
        _, long_name, field_tile, tag_pointing, tag_band = object_tag.split(" ")
        ftb = field_tile + header["FILTER"]
    except Exception:
        ftb = None
        print(f"{stack_name} bad tag {object_tag}")
        data["tag"] = header["OBJECT"]
        data["obj_band"] = "X"
        return data
        #long_name, field_tile, tag_pointing, tag_band = None, None, None, None
    field_code = field_tile[:2]
    tile = field_tile[2:]
    code_from_lookup = survey_config["field_to_code"][long_name]
    if code_from_lookup != field_code:
        print(f"{stack_name} {ftb} long name {long_name} doesn't corrospond to code {field_code}")
    field_code = code_from_lookup
    if header["FILTER"] != tag_band:
        hf = header['FILTER']
        #print(f"{stack_name} {ftb} header FILTER {hf} doesn't match object code {tag_band}")

    data["field"] = field_code
    data["tile"] = int(tile)
    data["pointing"] = tag_pointing[:3]
    data["obj_band"] = tag_band[0] # Sometimes  K is listed as K4 - a typo.

    data["tag"] = data["field"] + f"{data['tile']:02d}" + data["band"]
    return data

def extract_data_from_ccd_header(header, suffix=None):
    data = {}
    suffix = f"_{suffix}" if suffix is not None else ""
    try:
        magzpt = header["MAGZPT"]
    except:
        magzpt = -1.0
    data["magzpt" + suffix] = magzpt
    try:
        seeing = header["SEEING"] * header["PIXLSIZE"] / 2.0
    except:
        seeing = -1.0
    data["seeing" + suffix] = seeing
    return data

def extract_header_data(stack_path):
    try:
        with fits.open(stack_path) as f:
            pass
    except Exception:
        print(f"CORRUPT STACK: {stack_path.stem}")
        return {"rejected": True, "rejection_reason": "corrupt"}
    with fits.open(stack_path) as f:
        primary_header = f[0].header
        header_data = extract_data_from_primary_header(
            primary_header, stack_name=stack_path.stem
        )
        del primary_header
        if header_data["rejected"]:
            return header_data
        #del f[0].header
        for jj, ccd in enumerate(survey_config["ccds"]):
            ccd_header = f[ccd].header
            ccd_header_data = extract_data_from_ccd_header(ccd_header, suffix=ccd)
            header_data.update(ccd_header_data)
            del ccd_header
            #del f[ccd].header

        del f
    return header_data

def process_stack(stack_path, index):
    stack_path = Path(stack_path)
    header_data = extract_header_data(stack_path)
    if header_data is None:
        raise ValueError("extract_header_data should not reutn None")
    header_data["filename"] = stack_path.stem
    header_data["index"] = index
    tag = header_data.get("tag", "NO TAG")

    bad_header_fixes = bad_headers.get(stack_path.stem, None)
    if bad_header_fixes is not None:
        for k, v in bad_header_fixes.items():
            if k not in header_data:
                print(f"{stack_path.stem} {tag}: add {k}={v} to header_data")
                #raise ValueError("trying to replace non-existant, {k}")
            else:
                print(f"{stack_path.stem} {tag}: replace {k} {header_data[k]} with {v}")
            if k == "tile":
                header_data[k] = int(v)
            else:
                header_data[k] = v

    if header_data["rejected"]:
        header_data["tile"] = -1
        reason = header_data.get("rejection_reason", "no reason given...?!")
        print(f"{stack_path.stem} {tag}: REJECT, {reason}")
        
    return header_data

def _process_stack_wrapper(arg):
    return process_stack(*arg)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-cpus", type=int, default=1)
    
    args = parser.parse_args()

    bad_headers_path = paths.base_path / "setup_scripts/bad_headers.yaml"
    print(bad_headers_path)
    if bad_headers_path.exists():
        with open(bad_headers_path, "r") as f:
        #    bad_headers = json.load(f)        
            bad_headers = yaml.load(f, Loader=yaml.FullLoader)
    else:
        bad_headers = {}

    print(f"Defined in {bad_headers_path}")
    print(f"Looking out for {len(bad_headers)} bad headers to fix... ")
    print(f"     defined in {bad_headers_path}\n")
    print(f"Extracting data from {len(stack_paths)} stacks\n\n\n")


    wget_path = paths.base_path / "setup_scripts/dxs.wget"

    wget_deprec_lookup = {}
    deprec_idxes = ["mf", 1, 2, 3, 4]
    if wget_path.exists():
        with open(wget_path, "r") as f:
            for ii, line in enumerate(f):
                lspl = line.split("#")
                fname = lspl[0].split("-O")[1].strip().split(".")[0]
                deprec_codes = [int(x) for x in lspl[1].split(",")[1:]]

                wget_deprec_lookup[fname] = {
                    idx: code for idx, code in zip(deprec_idxes, deprec_codes)
                }

    #interval = 25
    t1 = time.time()
    if args.n_cpus == 1:
        header_data_list = []
        for ii, stack_path in enumerate(stack_paths):        
            header_data = process_stack(stack_path, ii)
            header_data_list.append(stack_path)
    else:
        pool_args = [(p, ii) for ii, p in enumerate(stack_paths)]
        with Pool(args.n_cpus) as pool:
            header_data_list = pool.map(_process_stack_wrapper, pool_args)

    for dat in header_data_list:
        for ii in ["mf", 1, 2, 3, 4]:
            dat[f"deprec_{ii}"] = wget_deprec_lookup.get(dat["filename"], {}).get(ii, 0)

    t2 = time.time()
    print(f"processed {len(stack_paths)} stacks in {(t2-t1)/60.} min")
               
    dataframe = pd.DataFrame(header_data_list)
    dataframe.set_index("index", inplace=True)
    corrupt_data = dataframe[ dataframe["tile"] < 0 ]
    corrupt_stacks = [ x for x in corrupt_data["filename"] ]

    dataframe = dataframe[ dataframe["tile"] > 0]
    dataframe.sort_index(inplace=True)

    dataframe.drop(["rejected", "rejection_reason"], inplace=True, axis=1)

    reorder = [
        "field", "tile", "band", "pointing", "exptime", "MJD",
        "magzpt_1", "magzpt_2", "magzpt_3", "magzpt_4",
        "seeing_1", "seeing_2", "seeing_3", "seeing_4",
        "deprec_mf", "deprec_1", "deprec_2", "deprec_3", "deprec_4",
        "filename", "project", "obj_band", "tag", 
    ]

    print(dataframe.columns)

    missing = set(x for x in dataframe.columns if x not in reorder)
    if len(missing) > 0:
        raise ValueError(f"you're forgetting about {missing} in reorder")
        
   
    reordered_df = dataframe[reorder]

    header_data_path = paths.header_data_path # defaults to paths.config_path / "dxs_header_data.csv"
    reordered_df.to_csv(header_data_path, float_format="%.5f", index=False)


    corrupt_frames_path = paths.config_path / "dxs_corrupt_fits.txt"

    dataframe["bad_band"] = dataframe["band"] != dataframe["obj_band"]

    print(f"there are {dataframe['bad_band'].sum()} mis-matched filter/object tag")

    print(f"there are {len(corrupt_stacks)} corrupt frames, named in \n    {corrupt_frames_path}")

    with open(corrupt_frames_path, "w+") as f:
        lines = [x+"\n" for x in corrupt_stacks]
        f.writelines(lines)



