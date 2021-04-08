import numpy as np

from astropy.table import Table

from dxs import CrosstalkProcessor

from dxs import paths


def test__cp_init():

    test_stacks = []
    test_table = Table()
    test_crosstalk_catalog_path = paths.temp_test_path / "crosstalk_init.cat.fits"
    cp = CrosstalkProcessor(
        test_stacks, 
        test_table, 
        crosstalk_catalog_path=test_crosstalk_catalog_path
    )
    expected_orders = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
    assert set(cp.crosstalk_orders) == set(expected_orders)

def test__get_stars_in_frame():

    test_stacks = []
    test_table = Table()
    test_crosstalk_catalog_path = paths.temp_test_path / "crosstalk_init.cat.fits"
    cp = CrosstalkProcessor(
        test_stacks, 
        test_table, 
        crosstalk_catalog_path=test_crosstalk_catalog_path
    )

    short_star_table = Table(
        {"ra": [180.0], "dec": [0.0], "xpix": [1000], "ypix": [1000], "parent_id": [1]}
    )
    crosstalk_pixels = cp.get_crosstalk_pixels(short_star_table)

    # can't scatter over center x line / y line = 4096 / 2 = 2048
    # so crosstalks of crosstalks at positions 
    #  3048,  2792,  2536,  2280,  2024,  1768,  1512,  1256,  1000,
    #     744,   488,   232,   -24,  -280,  -536,  -792, -1048
    expected_xpix = set([2024, 1768,  1512,  1256,   744,   488,   232])

    assert set(crosstalk_pixels["xpix"]) == expected_xpix
    
