import os
import pytest

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from dxs.utils import table as table_utils
from dxs.utils.image import build_mosaic_wcs

from dxs import paths

def test__table_to_numpynd():
    tab = Table(
        {
            "x": [2., 3., 4., 5.,], 
            "y": [11., -12., 13., 14.,], 
            "z": [0., -10., 7., np.nan]
        }
    )
    
    expected_arr = np.array([
        [2., 11., 0.],
        [3., -12., -10.],
        [4., 13., 7.],
        [5., 14., np.nan]
    ])

    result = table_utils.table_to_numpynd(tab)
    assert np.allclose(result, expected_arr, equal_nan=True)

def test__fix_column_names():
    catalog_path = paths.scratch_test_path / "tableutils_fix_names.cat.fits"
    if catalog_path.exists():
        os.remove(catalog_path)
    assert not catalog_path.exists()

    catalog = Table({
        "FLUX": np.random.uniform(0, 1, 10),
        "BLAH_BLAH": np.random.uniform(0, 1, 10),
        "TEST_COL": np.random.uniform(5, 6, 10),
    })

    catalog.write(catalog_path)


    col_lookup = {
        "BLAH_BLAH": "different_blah"
    }
    table_utils.fix_column_names(
        catalog_path, column_lookup=col_lookup, band="Q", prefix="www", suffix="mmm"
    )
    result = Table.read(catalog_path)

    assert set(result.columns) == set(["FLUX", "Q_wwwdifferent_blahmmm", "TEST_COL"])    

def test__explode_columns_in_fits():
    pass

def test__explode_column():
    pass

def test__remove_objects_in_bad_coverage():
    catalog_path = paths.scratch_test_path / "tableutils_bad_coverage.cat.fits"
    if catalog_path.exists():
        os.remove(catalog_path)
    assert not catalog_path.exists()

    catalog = Table({
        "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "coverage": [1, 1, 11, 12, 12, 10, 8, 14, 3, 5],
        "random_vals": np.random.uniform(0, 1, 10),
    })

    catalog.write(catalog_path)

    table_utils.remove_objects_in_bad_coverage(
        catalog_path, coverage_column="coverage", minimum_coverage=11
    )

    result = Table.read(catalog_path)
    assert len(result) == 4
    assert np.allclose(result["val"].data, np.array([3, 4, 5, 8]))
    assert np.allclose(result["coverage"].data, np.array([11, 12, 12, 14]))

def test__add_column_to_catalog():
    catalog_path = paths.scratch_test_path / "tableutils_add_col.cat.fits"
    if catalog_path.exists():
        os.remove(catalog_path)
    assert not catalog_path.exists()
    catalog = Table({
        "x": np.random.uniform(0, 1, 10),
        "y": np.random.uniform(0, 1, 10),
    })
    catalog.write(catalog_path)

    table_utils.add_column_to_catalog(
        catalog_path, {"data1": 3., "data2": [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]}
    )

    result = Table.read(catalog_path)
    assert np.allclose(result["data1"].data, 3.)
    assert np.allclose(
        result["data2"].data, np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
    )

def test__add_map_value():
    catalog_path = paths.scratch_test_path / "tableutils_map_value.cat.fits"
    if catalog_path.exists():
        os.remove(catalog_path)
    assert not catalog_path.exists()

    catalog = Table({
        "random_data": np.random.uniform(0, 1, 7),
        "bad_ra": np.random.uniform(89.9, 90.1, 7),
        "bad_dec": np.random.uniform(44.9, 45.1, 7),
        "xpix": [0, 10, 12, 50, 25, 90, 90],
        "ypix": [0, 10, 10, 50, 90, 25, 75],
    })
    catalog.write(catalog_path)

    image_size = (100, 100)    
    hdu_wcs = build_mosaic_wcs(
        SkyCoord(ra=90., dec=45., unit="deg"), (100, 100), pixel_scale=1.0
    )
    header = hdu_wcs.to_header()
    
    data = np.zeros(image_size) + 1
    
    data[:50, 50:] = 10    
    data[50:, 50:] = 20
    data[10, 10] = 100

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu_path = paths.scratch_test_path / "tableutils_map_value_map.fits"
    if hdu_path.exists():
        os.remove(hdu_path)
    assert not hdu_path.exists()
    hdu.writeto(hdu_path)

    with pytest.raises(ValueError):
        table_utils.add_map_value_to_catalog(
            catalog_path, hdu_path, "map_value", ra="bad_ra", dec="bad_dec", xpix="xpix", ypix="ypix"
        )
        

    table_utils.add_map_value_to_catalog(
        catalog_path, hdu_path, "map_value", xpix="xpix", ypix="ypix"
    )

    result = Table.read(catalog_path)
    assert np.allclose(
        result["map_value"].data, [1, 100, 1, 20, 1, 10, 20]
    )

