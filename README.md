# dxs

## Pipeline for creating mosaics and catalogs from UKIDSS-DXS

### Setup

1. Clone this repo
2. Preferably start a new python3.6 virtualenv
3. Install this repository (and requirements):
   - `python3 -m pip install -r requirements.txt`
   - `python3 -m pip install -e .`
   - install python3 package `astromatic_wrapper`. If you can't do this with pip, 
     then there's a forked repo at https://github.com/aidansedgewick/astromatic_wrapper which you should be able
     able to clone (outside of this directory) and install with `python3 -m pip install -e .`
4. `cd setup_scripts` then `./get_data.sh`. This will take a few hours.
5. Still in `setup_scripts`, `python3 extract_header_info.py`. Will take ~20 mins.
6. Optionally prepare the panstarrs catalogs:
   - copy the "raw" catalogs into `input_data/external/panstarrs`.
   - in `setup_scripts`, do `python3 process_panstarrs.py`

Make sure `SWarp`, `SExtractor` and `stilts` are in the path (eg. `module load swarp/2.38`).

### Creating mosaics.

If you're running on a cluster with a scheduler, there is a "runner script" to help with this.
Modify `configuration/system_config.yaml` to suit your system.

Then: `python3 dxs/runner/setup_run.py [field_list] [tile_list] [band_list] scripts/mosaic_pipeline.py --run_name [some_meaningful_name]`
Follow the instructions on screen. 
For all mosaics, use:
  field_list `EN,LH,SA,XM`
  tile_list 1-12
  band_list `J,H,K`
Subsets also work. eg EN,SA 1,2,4,8, J

You need lots (O(10TB)!) of scratch space to do this all in one go,
so probably best to do this in a few parts,
and clear temp_data/hdus every time.

If you're not using a scheduler, you can run:
    `python3 scripts/mosaic_pipeline.py [field] [tile] [band]` 
to create a single mosaic. Mosaics are ~1Gb each.

### Creating catalogs

These don't take so long.

You can still create them with the scheduler, however:
    `python3 dxs/runner/setup_run.py [field_list] [tile_list] [band_list] scripts/basic_pipeline.py --run_name [meaningful_name]`
and follow the instructions again.
