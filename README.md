# dxs

## Pipeline for creating mosaics and catalogs from UKIDSS-DXS

### Setup.

1. Clone this repo
2. Preferably start a new python3.6 virtualenv
   - `python3 -m virtualenv dxsenv`
   - `source dxsenv/bin/activate` (do this step every time)
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

Then create a "run_config". (See example in dxs/runner/blank_config.yaml).
Or, there are couple of pre-made ones in eg dxs/runner/configs/EN_mosaics.yaml

Then: `python3 dxs/runner/setup_run.py -c /path/to/run_config.yaml`
Follow the instructions on screen. 
For all mosaics, use:
  fields `EN,LH,SA,XM`
  tiles 1-12
  bands `J,H,K`
Subsets also work. eg EN,SA 1,2,4,8 J

You need lots (O(10Tb)!) of scratch space to do this all in one go,
so probably best to do this in a few parts,
and clear temp_data/hdus every time.

If you're not using a scheduler, you can run:
    `python3 scripts/mosaic_pipeline.py [field] [tile] [band]` 
to create a single mosaic. Mosaics are ~1Gb each.

It took me about 1hr per mosaic to run the full "pipeline" on a 16 core node.

### Creating catalogs.

These don't take as long.

You can still create them with the scheduler, however:
    `python3 dxs/runner/setup_run.py -c /path/to/run_config.yaml`
and follow the instructions again.

To match other optical/MIR catalogs, you'll need to modify `scripts/basic_pipeline.py`

Should be as easy as adding a new `pair_matcher.match_catalog("path/to/catalog.fits")`
(Adding extra kwargs as required.)

### Analysis.

to be added.
