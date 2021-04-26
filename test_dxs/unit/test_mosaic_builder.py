import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS

from dxs import mosaic_builder
from dxs.utils.image import build_mosaic_header

from dxs import paths


magzpt_data = {f"magzpt_{ii}": np.random.normal(20.0, 0.1, 1000) for ii in [1,2,3,4]}
seeing_data = {f"seeing_{ii}": np.random.normal(0.8, 0.1, 1000) for ii in [1,2,3,4]}
example_stack_data = pd.DataFrame(
    {**magzpt_data, **seeing_data, "exptime": np.full(1000, 10.)}
)
print(example_stack_data)

def test_get_stack_data():
    stack_bad_field = mosaic_builder.get_stack_data("NA", 1, "J")
    assert len(stack_bad_field) == 0
    stack_bad_tile = mosaic_builder.get_stack_data("SA", 13, "J")
    assert len(stack_bad_tile) == 0
    stack_bad_filter = mosaic_builder.get_stack_data("EN", 4, "H")
    assert len(stack_bad_tile) == 0
    SA04K_stack_data = mosaic_builder.get_stack_data("SA", 4, "K")
    assert len(SA04K_stack_data) == 122
    assert set(SA04K_stack_data["pointing"]) == set(["0_0", "0_1", "1_0", "1_1"])

def test_get_neighbor_stacks():
    pass

def test_calc_mosaic_geometry():
    size = (360, 360)
    pix_scale = 1. #arcsec
    centers = [
        (180. + 0.033, 0.033), (180. - 0.033, 0.033), 
        (180. + 0.033, -0.033), (180. - 0.033, -0.033)
    ]
    hdu_list = []
    for ii, center in enumerate(centers):
        data = np.random.uniform(0, 1, size)
        header = build_mosaic_header(center, size, pix_scale)
        hdu_path = paths.temp_test_path / f"calc_geom_test_hdu_{ii}.fits"
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(hdu_path, overwrite=True)
        hdu_list.append(hdu_path)        

    center, mosaic_size = mosaic_builder.calculate_mosaic_geometry(
        hdu_list, pixel_scale=1. # arcsec
    )
    assert np.allclose(np.array([180., 0.]), np.array([center.ra.value, center.dec.value]), atol=1e-3)
    assert np.allclose((600., 600.), np.array(mosaic_size), rtol=0.02) # spherical-ness.

    center2, mosaic_size2 = mosaic_builder.calculate_mosaic_geometry(
        hdu_list, pixel_scale=1., factor=2.0, border=57 # arcsec
    )
    assert np.allclose(np.array(mosaic_size2), np.array(mosaic_size)*2 + 57, atol=1)
    assert np.allclose(mosaic_size2, np.array([1257., 1257.]), rtol=0.02)

def test_add_keys():
    empty_header = build_mosaic_header(
        (330., 0.), (1000, 1000), pixel_scale=1. / 3600.
    )
    data = np.random.uniform(0, 1, (1000, 1000))
    print(empty_header)
    empty_fits = fits.PrimaryHDU(data=data, header=empty_header)
    fits_path = paths.temp_test_path / "add_keys_test.fits"
    empty_fits.writeto(fits_path, overwrite=True)
    
    keys_to_add = {"test1": 100, "test2": ("some_data", "this is a comment")}    

    mosaic_builder.add_keys(fits_path, keys_to_add, verbose=True)
    with fits.open(fits_path) as f:
        assert f[0].header["TEST1"] == 100
        assert f[0].header["TEST2"] == "some_data"
        assert f[0].header.comments["TEST2"] == "this is a comment"

def test_mosaic_builder():
    builder_for_bad_spec = mosaic_builder.MosaicBuilder.from_dxs_spec("SA", 13, "K")
    assert builder_for_bad_spec is None

    swarp_config = {"center": (180., 1.234), "image_size": (12345, 21000)}
    SA04K_builder = mosaic_builder.MosaicBuilder.from_dxs_spec(
        "SA", 4, "K", swarp_config=swarp_config, include_neighbors=False
    )
    assert len(SA04K_builder.stack_list) == 122
    SA04K_builder.initialise_astromatic()
    print(SA04K_builder.cmd_kwargs)
    assert "-NTHREADS 1" in SA04K_builder.cmd_kwargs["cmd"]
    assert "-CENTER 180.000000,1.234000" in SA04K_builder.cmd_kwargs["cmd"]
    assert "-IMAGE_SIZE 12345,21000" in SA04K_builder.cmd_kwargs["cmd"]
    assert SA04K_builder.cmd_kwargs["code"] == "SWarp"

    
def test_magzpt():
    magzpt = mosaic_builder.MosaicBuilder.calc_magzpt(example_stack_data)
    assert np.isclose(magzpt, 22.5, atol=0.01)
    magzpt_no_exptime = mosaic_builder.MosaicBuilder.calc_magzpt(
        example_stack_data, magzpt_inc_exptime=False
    )
    assert np.isclose(magzpt_no_exptime, 20.0, atol=0.01)
    

def test_seeing():
    seeing = mosaic_builder.MosaicBuilder.calc_seeing(example_stack_data)
    assert np.isclose(seeing, 0.8, atol=0.01)

def test_hdu_preparer():
    pass

def test_prepare_stack():
    pass

