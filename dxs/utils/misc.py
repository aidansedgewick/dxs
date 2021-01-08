import os
import shutil
from pathlib import Path
from typing import List

import numpy

class ModuleMissingError(Exception):
    pass

def check_modules(modules):
    if not isinstance(modules, list):
        modules = [modules] # convert to list, for looping.
    missing_modules = []
    present_modules = {}
    for module in modules:
        # check if it exists
        exe_path = shutil.which(module)
        if exe_path is None:
            missing_modules.append(module)
        else:
            print(f"Using {module} at {exe_path}")
    if len(missing_modules) > 0:
        raise ModuleMissingError(f"Missing modules {missing_modules}. Please load.")

def format_flags(config, capitalise=True, float_precision=6):
    """
    Capitalise config keys.
    Convert pathlib.Path types to str
    Convert int to str
    Convert float to six decimal place string
    Convert tuple of int to comma-separated string
    Convert tuple of float to comma-sep string with six decimal places.
    Do nothing to string.
    """
    formatted_config = {}
    for param, value in config.items():
        if capitalise:
            key = param.upper()
        else:
            key = param
        if isinstance(value, str):
            formatted_config[key] = value
        elif isinstance(value, Path):
            formatted_config[key] = str(value) # if it's a pathlib Path, fix it for JSON later.
        elif isinstance(value, float):
            formatted_config[key] = f"{value:.{float_precision}f}"
        elif isinstance(value, int):
            formatted_config[key] = str(value)
        elif isinstance(value, tuple):
            if all(isinstance(x, int) for x in value):
                formatted_config[key] = ','.join(f"{x}" for x in value)
            elif all(isinstance(x, float) for x in value):
                formatted_config[key] = ','.join(f"{x:.{float_precision}f}" for x in value)
            else:
                raise ValueError(f"Don't know how to format type {type(value)} - check element types?")
        else:
            raise ValueError(f"Don't know how to format type {type(value)}")
    return formatted_config

def create_file_backups(file_list: List[Path], temp_dir: Path):
    temp_dir = Path(temp_dir)
    new_paths = []
    for file_path in file_list:
        file_path = Path(file_path)
        new_file_path = temp_dir / file_path.name
        new_paths.append(new_file_path)
        shutil.copy2(file_path, new_file_path)
    return new_paths

def get_git_info():
    pass


