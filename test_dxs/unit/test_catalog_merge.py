import numpy as np
import pytest

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from easyquery import Query

from dxs import catalog_merge

from dxs.utils.image import (
    build_mosaic_wcs, uniform_sphere, calc_survey_area, calc_spherical_rectangle_area
)

from dxs import paths


def test__contained_in_multiple_wcs():
    """
    This is a poor test.
    # TODO: make this better somehow?
    # ideally would select the "expected" randoms in multiple WCS by 
    looking at dec > some_value, but I can't figure out how to deal with wcs_max_dec
    is a function of ra.
    """

    size = int(1.1 * 3600)
    
    c1 = SkyCoord(ra=180., dec=0.5, unit="deg")
    c2 = SkyCoord(ra=180., dec=-0.5, unit="deg") 
    s1 = ( size, int(size * np.cos(c1.dec)) )
    s2 = ( size, int(size * np.cos(c2.dec)) )

    wcs1 = build_mosaic_wcs(c1, s1[::-1], pixel_scale=1.0)
    wcs2 = build_mosaic_wcs(c2, s2[::-1], pixel_scale=1.0)

    r_arr = uniform_sphere((179., 181.), (-1.5, 1.5), density=1e4)
    randoms = SkyCoord(ra=r_arr[:,0], dec=r_arr[:,1], unit="deg")

    in_wcs1 = wcs1.footprint_contains(randoms)
    in_wcs2 = wcs2.footprint_contains(randoms)

    with pytest.raises(ValueError):
        failing_mask = catalog_merge.contained_in_multiple_wcs(randoms, [wcs1, wcs2], at_least_one=True)
        # ie, some randoms are outside the wcses.

    good_randoms = randoms[ in_wcs1 | in_wcs2 ]
    mask = wcs1.footprint_contains(good_randoms) & wcs2.footprint_contains(good_randoms)
    assert sum(mask) > 0 # there are randoms in both wcses.

    result = catalog_merge.contained_in_multiple_wcs(good_randoms, [wcs1, wcs2], at_least_one=True)
    np.testing.assert_array_equal(mask, result)

    """
    ## compare AREAS...???

    data1 = np.ones(s1)
    hdu1 = fits.PrimaryHDU(data=data1, header=wcs1.to_header())
    hdu1_path = paths.scratch_test_path / "catmerge_multi_wcs1.fits"
    hdu1.writeto(hdu1_path, overwrite=True)

    data2 = np.ones(s2)
    hdu2 = fits.PrimaryHDU(data=data2, header=wcs2.to_header())
    hdu2_path = paths.scratch_test_path / "catmerge_multi_wcs2.fits"
    hdu2.writeto(hdu2_path, overwrite=True)
    
    area_from_hdus = calc_survey_area([hdu1_path, hdu2_path], density=1e5)

    rect_area = calc_spherical_rectangle_area((179., 181.), (-1.5, 1.5))
    overlap_area = rect_area * sum(result) / len(randoms) # ALL randoms, not just good.
    assert np.isclose(overlap_area, area_from_hdus, rtol=0.01) # fairly lenient...
    """

    
def test__select_group_argmax():

    tab = Table({
        "group_id": [1,  1,  1,  1,  2,  2,  3,  3,  4 ],
        "test_val": [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        "snr_val":  [1., 2., 9., 1., 2., 8., 3., 3., 4.],
        ##          |        ^     |     ^ | ^     | ^ |
    })
    
    result = catalog_merge.select_group_argmax(tab, "group_id", "snr_val")
    
    assert len(result) == 4

    assert np.isclose(result[ result["group_id"] == 1]["test_val"], 3.)
    assert np.isclose(result[ result["group_id"] == 1]["snr_val"], 9.)

    assert np.isclose(result[ result["group_id"] == 2]["test_val"], 6.)
    assert np.isclose(result[ result["group_id"] == 2]["snr_val"], 8.)

    assert np.isclose(result[ result["group_id"] == 3]["test_val"], 7.) 
    assert np.isclose(result[ result["group_id"] == 3]["snr_val"], 3.)

    assert np.isclose(result[ result["group_id"] == 4]["test_val"], 9.)
    assert np.isclose(result[ result["group_id"] == 4]["snr_val"], 4.)

def test__check_for_merge_error():
    tab = Table({
        "GroupID": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "mag": [9.9, 10.05, 10.1, 9.95, 0.2, 0.21, 0.19, 1.0, 1.0, 1.1]
    })
    val_check_diff = catalog_merge.check_for_merge_errors(
        tab, "GroupID", "mag"
    )
    assert np.allclose(val_check_diff, [0.2, 0.02, 0.1])
        
    


def test__merge_catalogs():
    c1 = SkyCoord(ra=214.5, dec=44.5, unit="deg")
    c2 = SkyCoord(ra=215.5, dec=44.5, unit="deg")
    c3 = SkyCoord(ra=215.5, dec=45.5, unit="deg")
    c4 = SkyCoord(ra=214.5, dec=45.5, unit="deg")

    size = int(1.1 * 3600)
    pixel_scale = 1.0

    s1 = (size, int(size * np.cos(c1.dec)) ) 
    s2 = (size, int(size * np.cos(c2.dec)) )
    s3 = (size, int(size * np.cos(c3.dec)) )
    s4 = (size, int(size * np.cos(c4.dec)) )

    wcs1 = build_mosaic_wcs(c1, s1[::-1], pixel_scale)
    wcs2 = build_mosaic_wcs(c2, s2[::-1], pixel_scale)
    wcs3 = build_mosaic_wcs(c3, s3[::-1], pixel_scale)
    wcs4 = build_mosaic_wcs(c4, s4[::-1], pixel_scale)

    r_arr = uniform_sphere((212., 218.), (42., 48.), density=1e3)   
    randoms = SkyCoord(ra=r_arr[:,0], dec=r_arr[:,1], unit="deg")
    mag_vals = np.random.uniform(16, 24, len(r_arr))

    in_wcs1 = wcs1.footprint_contains(randoms)
    in_wcs2 = wcs2.footprint_contains(randoms)
    in_wcs3 = wcs3.footprint_contains(randoms)
    in_wcs4 = wcs4.footprint_contains(randoms)

    catalogs = []
    for ii, mask in enumerate([in_wcs1, in_wcs2, in_wcs3, in_wcs4], 1):
        cat = Table({
            "id": np.arange(sum(mask)), 
            "ra": r_arr[mask, 0], 
            "dec": r_arr[mask, 1], 
            "mag": mag_vals[mask],
            "snr": np.random.uniform(1, 2, sum(mask)),
            "tile": np.full(sum(mask), ii)
        })
        mod4_mask = cat["id"] % 4 == ii
        cat[ mod4_mask ]["snr"] = cat[ mod4_mask ]["snr"] + 10.
        catalogs.append(cat)

    cat1, cat2, cat3, cat4 = catalogs

    in_any_wcs = ( in_wcs1 | in_wcs2 | in_wcs3 | in_wcs4 )
    expected_output_len = sum(in_any_wcs) 

    # weak test...
    assert sum([len(x) for x in [cat1, cat2, cat3, cat4]]) > expected_output_len

    input_data_list = [
        (cat1, wcs1), (cat2, wcs2), (cat3, wcs3), (cat4, wcs4)
    ]
    output_path = paths.scratch_test_path / "merge_test.cat.fits"
    catalog_merge.merge_catalogs(
        input_data_list, output_path, "id", "ra", "dec", "snr", 
        value_check_column="mag",
        merge_error=0.5,
    )

    merged = Table.read(output_path)

    assert len(merged) == expected_output_len

    grid = np.column_stack([in_wcs1, in_wcs2, in_wcs3, in_wcs4])
    expected_overlap = sum(np.sum(grid, axis=1) > 1)
    in_overlap = sum(merged["GroupSize"] > 0)
    assert expected_overlap == in_overlap

    center_square_mask = ( in_wcs1 & in_wcs2 & in_wcs3 & in_wcs4 )
    in_center_square = sum(center_square_mask)
    assert len(merged[ merged["GroupSize"] == 4 ]) == in_center_square







