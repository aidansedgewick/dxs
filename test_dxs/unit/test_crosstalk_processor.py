import os

import numpy as np

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

from dxs import CrosstalkProcessor
from dxs.mosaic_builder import build_mosaic_header
from dxs import paths


def test__crosstalk_processor_init():

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

def test__get_crosstalk_pixels():

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
    
def test__get_stars_in_frame_from_wcs():
    
    star_table_path = paths.input_data_path / "external/tmass/tmass_SA22_stars.csv"
    star_table = Table.read(star_table_path, format="ascii")

    test_crosstalk_catalog_path = paths.temp_test_path / "crosstalk_init.cat.fits"
    test_stacks = []
    cp = CrosstalkProcessor(
        test_stacks, 
        star_table, 
        crosstalk_catalog_path=test_crosstalk_catalog_path
    )

    center = (334.0, 0.0)
    pixel_scale = 2.0
    size = ( int(1. / (pixel_scale / 3600.)), int(1. / (pixel_scale / 3600.)) )
    test_header = build_mosaic_header(center, size, pixel_scale)
    test_wcs = WCS(test_header)

    stars_in_frame = cp.stars_in_frame_from_wcs(
        test_wcs, star_table, mag_column="k_m", mag_limit=12.0, xlen=size[0], ylen=size[1]
    )
    assert len(stars_in_frame) == 130

def test__get_stars_in_stack():

    pixel_scale = 0.2
    size = ( 4156, 4156 )
    dth = (size[0] // 4) * pixel_scale / 3600.

    c1 = (334.0, 1.0)
    test_header1 = build_mosaic_header(c1, size, pixel_scale)
    #test_wcs1 = WCS(test_header1)

    c2 = (334.0, 2.0)
    test_header2 = build_mosaic_header(c2, size, pixel_scale)
    #test_wcs2 = WCS(test_header2)

    h0 = fits.Header()
    h0["OBJECT"] = "DXS FF00 X 0_0" # <dxs> <fieldtile> <band> <pointing>

    primary_hdu = fits.PrimaryHDU(data=None, header=h0)
    img1 = fits.ImageHDU(data=np.zeros(size), header=test_header1)
    img2 = fits.ImageHDU(data=np.zeros(size), header=test_header2)
    test_stack = fits.HDUList([primary_hdu, img1, img2])

    test_stack_path = paths.temp_test_path / "test_stack.fits"
    test_stack.writeto(test_stack_path, overwrite=True)

    star_ra_vals = ([
        c1[0] - dth, c1[0] - dth, c1[0] + dth, c1[0] + dth,
        c2[0] - dth, c2[0] - dth, c2[0] + dth, c2[0] + dth,
    ])
    star_dec_vals = np.array([
        c1[1] - dth, c1[1] + dth, c1[1] - dth, c1[1] + dth,
        c2[1] - dth, c2[1] + dth, c2[1] - dth, c2[1] + dth,
    ])
    mag_vals = np.full(len(star_ra_vals), 10.0)

    star_table = Table(
        {"ra": star_ra_vals, "dec": star_dec_vals, "k_m": mag_vals}
    )

    test_stacks = []
    test_crosstalk_catalog_path = paths.temp_test_path / "crosstalk_init.cat.fits"
    cp = CrosstalkProcessor(
        test_stacks,
        star_table,
        crosstalk_catalog_path=test_crosstalk_catalog_path
    )
    dpix = np.array([-4, -3, -2, -1, 1, 2, 3, 4]) * 256
    xtalk_ra_list = [
        np.full(len(dpix), c1[0] - dth),
        (c1[0] - dth) + dpix * pixel_scale / 3600.,
        (c1[0] + dth) + dpix * pixel_scale / 3600.,
        np.full(len(dpix), c1[0] + dth),
        np.full(len(dpix), c2[0] - dth),
        (c2[0] - dth) + dpix * pixel_scale / 3600.,
        (c2[0] + dth) + dpix * pixel_scale / 3600.,
        np.full(len(dpix), c2[0] + dth),
    ]
    xtalk_dec_list = [
        (c1[1] - dth) + dpix * pixel_scale / 3600.,
        np.full(len(dpix), c1[1] + dth),
        np.full(len(dpix), c1[1] - dth),
        (c1[1] + dth) + dpix * pixel_scale / 3600.,
        (c2[1] - dth) + dpix * pixel_scale / 3600.,
        np.full(len(dpix), c2[1] + dth),
        np.full(len(dpix), c2[1] - dth),
        (c2[1] + dth) + dpix * pixel_scale / 3600.,
    ]
    expected_crosstalk_table = Table(
        {
            "crosstalk_ra": np.concatenate(xtalk_ra_list),
            "crosstalk_dec": np.concatenate(xtalk_dec_list),
        }
    )
    crosstalk_table = cp.get_crosstalks_in_stack(
        test_stack_path, mag_column="k_m", ccds=[1, 2]
    )
    assert len(crosstalk_table) == len(expected_crosstalk_table)
    assert np.allclose(
        np.sort(crosstalk_table["crosstalk_ra"]), 
        np.sort(expected_crosstalk_table["crosstalk_ra"])
    )
    assert np.allclose(
        np.sort(crosstalk_table["crosstalk_dec"]),
        np.sort(expected_crosstalk_table["crosstalk_dec"])
    )
    os.remove(test_stack_path)
    assert not test_stack_path.exists()

def test__collate_crosstalks():
    
    n_test_stacks = 10
    stack_paths = []
    
    pixel_scale = 0.2
    size = ( 4156, 4156 )
    dth = (size[0] // 4) * pixel_scale / 3600.    

    for ii in range(n_test_stacks):
        if ii < 3:
            base_ra = 90.
            pointing = "1_0"
        else:
            # shifted over by one "crosstalk"
            base_ra = 90. - 256 * pixel_scale / 3600.
            pointing = "0_0"

        base_dec = 1.
        center = (
            base_ra + 1e-4 * np.random.uniform(-1, 1), 
            base_dec + 1e-4 * np.random.uniform(-1, 1)
        )

        h0 = build_mosaic_header(center, size, pixel_scale)
        h0["OBJECT"] = f"dxs TT00 X {pointing}"

        data = np.zeros(size)
        hdu = fits.PrimaryHDU(data=data, header=h0)
        hdu_path = paths.temp_test_path / f"collate_test_stack_{ii}.fits"
        hdu.writeto(hdu_path, overwrite=True)
        stack_paths.append(hdu_path)

    star_table = Table(
        {"ra": [90. - dth, 270. - dth], "dec": [1. - dth, 1. - dth], "k_m": [10., 10.]}
    )
    temp_collated_path = paths.temp_test_path / "xtalk_collate.cat.fits"
    cp = CrosstalkProcessor(
        stack_paths, star_table, crosstalk_catalog_path=temp_collated_path
    )
    cp.collate_crosstalks("k_m", ccds=[0])

    collated_crosstalks = Table.read(temp_collated_path)
    assert len(collated_crosstalks) == 16

    







