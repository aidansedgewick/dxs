from pathlib import Path

file_path = Path(__file__).absolute()

base_path = file_path.parent.parent

# INPUT DATA
input_data_path = file_path.parent.parent / "input_data"
stack_data_path = input_data_path / "stacks"

# SCRATCH PATHS
scratch_data_path = file_path.parent.parent / "scratch_data"
scratch_hdus_path = scratch_data_path / "hdus"
scratch_swarp_path = scratch_data_path / "swarp"
scratch_sextractor_path = scratch_data_path / "sextractor"
scratch_stilts_path = scratch_data_path / "stilts"

# CONFIGS
config_path = file_path.parent.parent / "configuration"
header_data_path = config_path / "dxs_header_data.csv"

# TESTS
tests_path = file_path.parent.parent / "test_dxs"
scratch_test_path = tests_path / "scratch_tests"

# (OUTPUT) DATA
data_path = file_path.parent.parent / "data"
mosaics_path =  data_path / "mosaics"
catalogs_path = data_path / "catalogs"
masks_path = data_path / "masks"


# RUNNER
runner_path = file_path.parent.parent / "runner"

print(f"\033[36minput_data_path\033[0m {input_data_path}")
print(f"\033[36mdata_path\033[0m {data_path}")
print(f"\033[36mconfig_path\033[0m {config_path}\n")

# HELPER FUNCS

def create_all_paths():
    input_data_path.mkdir(exist_ok=True, parents=True)
    stack_data_path.mkdir(exist_ok=True, parents=True)

    scratch_data_path.mkdir(exist_ok=True, parents=True) 
    scratch_hdus_path.mkdir(exist_ok=True, parents=True)   
    scratch_swarp_path.mkdir(exist_ok=True, parents=True)
    scratch_sextractor_path.mkdir(exist_ok=True, parents=True)
    scratch_stilts_path.mkdir(exist_ok=True, parents=True)

    scratch_test_path.mkdir(exist_ok=True, parents=True)

    data_path.mkdir(exist_ok=True, parents=True)
    mosaics_path.mkdir(exist_ok=True, parents=True)
    catalogs_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)

    runner_path.mkdir(exist_ok=True, parents=True)

def get_mosaic_stem(field, tile, band, prefix=None, suffix=None):
    prefix = prefix or ""
    suffix = suffix or ""
    # return f"{prefix}{field}{tile:02d}{band}{suffix}"
    if isinstance(tile, int):
        tile_code = f"{tile:02d}"
    else:
        tile_code = tile
    return f"{prefix}{field}{tile_code}{band}{suffix}"

def get_mosaic_dir(field, tile, band):
    return mosaics_path / get_mosaic_stem(field, tile, band)

def get_mosaic_path(field, tile, band, prefix=None, suffix=None, extension=".fits"):
    mosaic_dir = get_mosaic_dir(field, tile, band)
    mosaic_stem = get_mosaic_stem(field, tile, band, prefix=prefix, suffix=suffix)
    return mosaic_dir / f"{mosaic_stem}{extension}"

def get_catalog_stem(
    field, tile, detection_band, measurement_band=None, prefix=None, suffix=None
):
    prefix = prefix or ""
    if measurement_band is not None:
        measurement_band = f"{measurement_band}fp"
    measurement_band = measurement_band or ''
    suffix = suffix or ""
    #return f"{prefix}{field}{tile:02d}{detection_band}{measurement_band}{suffix}"
    if isinstance(tile, int):
        tile_code = f"{tile:02d}"
    else:
        tile_code = tile
    return f"{prefix}{field}{tile_code}{detection_band}{measurement_band}{suffix}"

def get_catalog_dir(field, tile, detection_band):
    return catalogs_path / get_catalog_stem(field, tile, "")

def get_catalog_path(
    field, tile, detection_band, measurement_band=None, 
    prefix=None, suffix=None, extension=".cat.fits"
):
    catalog_dir = get_catalog_dir(field, tile, detection_band)
    catalog_stem = get_catalog_stem(
        field, tile, detection_band, measurement_band=measurement_band, 
        prefix=prefix, suffix=suffix
    )
    return catalog_dir / f"{catalog_stem}{extension}"

path_summary_str = (
    f"PATHS\n==========================\n"
    + f"base_path = {base_path} (all other paths relative to here)\n"
    #+ f"(all other paths relative to base_path)\n"
    + f"\ninput paths:\n"
    + f" input_data_path = {input_data_path.relative_to(base_path)}\n"
    + f" stack_data_path = {stack_data_path.relative_to(base_path)}\n"
    + f" config_path = {config_path.relative_to(base_path)}\n"
    + f"\nscratch paths:\n"
    + f" scratch_data_path = {scratch_data_path.relative_to(base_path)}\n"
    + f" scratch_hdus_path = {scratch_hdus_path.relative_to(base_path)}\n"
    + f" scratch_swarp_path = {scratch_swarp_path.relative_to(base_path)}\n"
    + f" scratch_sextractor_path = {scratch_sextractor_path.relative_to(base_path)}\n"
    + f" scratch_stilts_path = {scratch_stilts_path.relative_to(base_path)}\n"
    + f" scratch_test_path = {scratch_test_path.relative_to(base_path)}\n"
    + f"\ndata paths:\n"
    + f" data_path = {data_path.relative_to(base_path)}\n"
    + f" catalogs_path = {catalogs_path.relative_to(base_path)}\n"
    + f" mosaics_path = {mosaics_path.relative_to(base_path)}\n"
    + f" masks_path = {masks_path.relative_to(base_path)}\n"
)


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--make-paths", default=False, action="store_true")    
    args = parser.parse_args()
    
    if args.make_paths:
        print("creating all paths")
        create_all_paths()

    print(path_summary_str)
