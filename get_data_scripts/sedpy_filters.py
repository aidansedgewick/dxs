import shutil
from glob import glob
from pathlib import Path

import sedpy

from dxs import paths

sedpy_data_path = Path(sedpy.__file__).parent / "data/filters"
input_sedpy_filters_path = paths.input_data_path / "sedpy_filters"

def copy_all_filters():

    print(f"sedpy_setup: copying filter profiles from {input_sedpy_filters_path} to {sedpy_data_path}")

    glob_pattern = str(input_sedpy_filters_path / "*.par")

    par_files = glob(glob_pattern)
    
    for filter_par in par_files:
        shutil.copy2(filter_par, sedpy_data_path)

copy_all_filters()
    
