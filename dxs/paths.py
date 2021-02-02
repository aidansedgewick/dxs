from pathlib import Path

file_path = Path(__file__).absolute()

base_path = file_path.parent

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

# CONFIGS
config_path = file_path.parent.parent / "configuration"
header_data_path = config_path / "dxs_header_data.csv"

# (OUTPUT) DATA
data_path = file_path.parent.parent / "data"
mosaics_path =  data_path / "mosaics"
catalogs_path = data_path / "catalogs"


# RUNNER
runner_path = file_path.parent.parent / "runner"


print(f"input_data_path {input_data_path}")
print(f"config_path {config_path}")

# HELPER FUNCS

def create_all_paths():
    temp_data_path.mkdir(exist_ok=True, parents=True) 
    temp_hdus_path.mkdir(exist_ok=True, parents=True)   
    temp_swarp_path.mkdir(exist_ok=True, parents=True)
    temp_sextractor_path.mkdir(exist_ok=True, parents=True)

    data_path.mkdir(exist_ok=True, parents=True)
    mosaics_path.mkdir(exist_ok=True, parents=True)
    catalogs_path.mkdir(exist_ok=True, parents=True)

    runner_path.mkdir(exist_ok=True, parents=True)

def get_mosaic_stem(field, tile, band, prefix=None):
    prefix = prefix or ""
    return f"{prefix}{field}{tile:02d}{band}"

def get_mosaic_dir(field, tile, band):
    return mosaics_path / get_mosaic_stem(field, tile, band)

def get_mosaic_path(field, tile, band, prefix=None, extension=".fits"):
    mosaic_dir = get_mosaic_dir(field, tile, band)
    mosaic_stem = get_mosaic_stem(field, tile, band, prefix=prefix)
    return mosaic_dir / f"{mosaic_stem}{extension}"

def get_catalog_stem(field, tile, detection_band, measurement_band=None, prefix=None):
    prefix = prefix or ""
    if measurement_band is not None:
        measurement_band = f"{measurement_band}fp"
    measurement_band = measurement_band or ''
    return f"{prefix}{field}{tile:02d}{detection_band}{measurement_band}"

def get_catalog_dir(field, tile, detection_band):
    return catalogs_path / get_catalog_stem(field, tile, "")

if __name__ == "__main__":
    print("creating all paths")
    create_all_paths()

