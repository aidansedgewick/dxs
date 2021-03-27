from os import remove, rmdir
from pathlib import Path
import json

import pytest

import numpy as np

from dxs.utils import misc
from dxs import paths

def test__check_modules():
    with pytest.raises(Exception):
        misc.check_modules("non_existant_module")
    # check for modules we absolutely know are here.
    misc.check_modules("python3", "git")

def test__format_flags():
    test_config = {
        "tuple_int_test": (100, 400, 9000),
        "tuple_float_test": (1234.543, 6000.),
        "none_test": None,
        "float_test": 3.14159265358979324,
        "int_test": 200,
        "str_test": "my_string123",
    }
    
    output_config = misc.format_flags(test_config)
    assert output_config["TUPLE_INT_TEST"] == "100,400,9000"
    assert output_config["TUPLE_FLOAT_TEST"] == "1234.543000,6000.000000"
    assert output_config["NONE_TEST"] == "None"
    assert output_config["FLOAT_TEST"] == "3.141593"
    assert output_config["INT_TEST"] == "200"
    assert output_config["STR_TEST"] == "my_string123"

def test__make_backup_paths():
    test_data1 = {"a": "string1", "b": "string2", "c": "string3"}
    test_path1 = Path(__file__).parent / "some_test_data.json"
    with open(test_path1, "w+") as f:
        json.dump(test_data1, f)
    test_data2 = {"d": "data_data", "e": "eeeee", "f": "super_secret_data"}
    test_path2 = Path(__file__).parent / "some_other_test_data.json"
    with open(test_path2, "w+") as f:
        json.dump(test_data2, f)

    temp_dir = Path(__file__).parent / "where_temp_files_live/subdir"
    new_test_paths = misc.create_file_backups([test_path1, test_path2], temp_dir=temp_dir)

    assert temp_dir.is_dir()
    copy1_path = temp_dir / "some_test_data.json"
    copy2_path = temp_dir / "some_other_test_data.json"
    assert copy1_path.exists()
    assert copy2_path.exists()
    assert copy1_path in new_test_paths
    assert copy2_path in new_test_paths
    # clean up
    remove(copy1_path)
    remove(test_path1)
    remove(copy2_path)
    remove(test_path2)
    rmdir(temp_dir)
    rmdir(temp_dir.parent)
    # make sure we've cleaned up.
    assert not copy1_path.exists()
    assert not copy2_path.exists()
    assert not temp_dir.parent.exists()

def test__tile_parser():
    assert misc.tile_parser("1-12") == [1,2,3,4,5,6,7,8,9,10,11,12]
    assert misc.tile_parser("1-4,8-10") == [1,2,3,4,8,9,10]

def test__calc_mids():
    arr1 = np.array([4,5,6,8])
    assert np.allclose(misc.calc_mids(arr1), np.array([4.5, 5.5, 7.]))
    arr2 = np.array([4.5, 10.0, 1000.0, 1001.0])
    assert np.allclose(misc.calc_mids(arr2), np.array([7.25, 505.0, 1000.5]))

def test__calc_range():
    arr = np.array([-100., -201.0, 3.14, 3.])
    assert misc.calc_range(arr) == (-201.0, 3.14)

