from pathlib import Path

file_path = Path(__file__).absolute()

base_path = file_path.parent.parent

# INPUT DATA
input_data_path = file_path.parent.parent / "input_data"
stack_data_path = input_data_path / "stacks"
if stack_data_path.is_dir() is False:
    raise ValueError(f"No data directory {data_path}. \nEither run setup_scripts/get_data.sh or manually add the data in this location. ")

# TEMP PATHS
temp_data_path = file_path.parent.parent / "temp_data"
temp_hdus_path = temp_data_path / "hdus"
temp_swarp_path = temp_data_path / "swarp"
temp_sextractor_path = temp_data_path / "sextractor"
temp_stilts_path = temp_data_path / "stilts"

# CONFIGS
config_path = file_path.parent.parent / "configuration"
header_data_path = config_path / "dxs_header_data.csv"

# TESTS
tests_path = file_path.parent.parent / "test_dxs"
temp_test_path = tests_path / "temp"

# (OUTPUT) DATA
data_path = file_path.parent.parent / "data"
mosaics_path =  data_path / "mosaics"
catalogs_path = data_path / "catalogs"
masks_path = data_path / "masks"


# RUNNER
runner_path = file_path.parent.parent / "runner"

print(f"input_data_path {input_data_path}\nconfig_path {config_path}\n")

# HELPER FUNCS

def create_all_paths():
    temp_data_path.mkdir(exist_ok=True, parents=True) 
    temp_hdus_path.mkdir(exist_ok=True, parents=True)   
    temp_swarp_path.mkdir(exist_ok=True, parents=True)
    temp_sextractor_path.mkdir(exist_ok=True, parents=True)
    temp_stilts_path.mkdir(exist_ok=True, parents=True)

    temp_test_path.mkdir(exist_ok=True, parents=True)

    data_path.mkdir(exist_ok=True, parents=True)
    mosaics_path.mkdir(exist_ok=True, parents=True)
    catalogs_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)

    runner_path.mkdir(exist_ok=True, parents=True)

def get_mosaic_stem(field, tile, band, prefix=None, suffix=None):
    prefix = prefix or ""
    suffix = suffix or ""
    return f"{prefix}{field}{tile:02d}{band}{suffix}"

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
    return f"{prefix}{field}{tile:02d}{detection_band}{measurement_band}{suffix}"

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

if __name__ == "__main__":
    print("creating all paths")
    create_all_paths()

