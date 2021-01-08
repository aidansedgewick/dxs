from pathlib import Path

file_path = Path(__file__).absolute()

input_data_path = file_path.parent.parent / "input_data"
stack_data_path = input_data_path / "stacks"
if stack_data_path.is_dir() is False:
    raise ValueError(f"No data directory {data_path}. \nEither run setup_scripts/get_data.sh or manually add the data in this location. ")

temp_data_path = file_path.parent.parent / "temp_data"
temp_data_path.mkdir(exist_ok=True, parents=True)
temp_hdus_path = temp_data_path / "hdus"
temp_hdus_path.mkdir(exist_ok=True, parents=True)
temp_swarp_path = temp_data_path / "swarp"
temp_swarp_path.mkdir(exist_ok=True, parents=True)
temp_sextractor_path = temp_data_path / "sextractor"
temp_sextractor_path.mkdir(exist_ok=True, parents=True)

config_path = file_path.parent.parent / "configuration"
header_data_path = config_path / "dxs_header_data.csv"

data_path = file_path.parent.parent / "data"
mosaics_path =  data_path / "mosaics"
catalogs_path = data_path / "catalogs"

print(f"input_data_path {input_data_path}")
print(f"config_path {config_path}")

