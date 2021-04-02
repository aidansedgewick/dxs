import numpy as np
import pandas as pd

from astropy.io import fits

from dxs.mosaic_builder import build_mosaic_header
from dxs.utils import image

from dxs import paths

def test__calc_spherical_rectangle_area():
    whole_sphere = image.calc_spherical_rectangle_area((0., 360.), (-90., 90.))
    sphere_ster = 4 * np.pi * (180. / np.pi) ** 2
    assert np.isclose(whole_sphere, sphere_ster)

    tiny_area = image.calc_spherical_rectangle_area((179.99, 180.01), (-0.01, 0.01))
    assert np.isclose(tiny_area, 4e-4)
    
    small_high_area = image.calc_spherical_rectangle_area((150.0, 154.0), (80.0, 84.0))
    area_from_calculator = 2.226317
    assert np.isclose(small_high_area, area_from_calculator)

def test__uniform_sphere():
    # test density option
    ra_lims_1 = (178., 182.)
    dec_lims_1 = (-2., 2.)
    area_1 = 15.99675
    assert np.isclose(image.calc_spherical_rectangle_area(ra_lims_1, dec_lims_1), area_1)
    density_1 = 1e5
    random_coords_1 = image.uniform_sphere(ra_lims_1, dec_lims_1, density=density_1)
    n_randoms_1 = int(area_1 * density_1)
    assert np.isclose(len(random_coords_1), n_randoms_1)

    # look at distribution in RA

    n_bins_1 = 10
    bins_1 = np.linspace(ra_lims_1[0], ra_lims_1[1], n_bins_1 + 1)
    n_per_bin_1 = n_randoms_1 / n_bins_1
    assert bins_1[-1] == ra_lims_1[1] # make sure we're endpoint inclusive.
    hist_1, _ = np.histogram(random_coords_1[:, 0], bins=bins_1)
    assert np.allclose(hist_1, n_per_bin_1, rtol=0.02)

    # look for uniform distribution in sin(dec)

    ra_lims_2 = (180., 180.2)
    dec_lims_2 = (50., 90.)
    density_2 = 1e5
    random_coords_2 = image.uniform_sphere(ra_lims_2, dec_lims_2, density=density_2)
    num_randoms_2 = len(random_coords_2)
    n_bins_2 = 10
    bins_2 = np.linspace(
        np.sin(dec_lims_2[0] * np.pi / 180.), 
        np.sin(dec_lims_2[1] * np.pi / 180.), 
        n_bins_2 + 1
    )

    hist_2, _ = np.histogram(np.sin(random_coords_2[:, 1] * np.pi / 180.), bins=bins_2)
    num_per_bin_2 = num_randoms_2 / n_bins_2
    assert np.allclose(hist_2, num_per_bin_2, rtol=0.02)

def test__single_coverage():
    # make an empty image.
    image_size = (7200, 7200)
    header = build_mosaic_header(center=(180.0, 0.0), size=image_size, pixel_scale=1.0)
    data = np.zeros(image_size)
    data[1800:5400, 1800:5400] = 5.0
    hdu = fits.PrimaryHDU(data=data, header=header)
    test_path = paths.temp_test_path / "single_coverage_test.fits"   
    hdu.writeto(test_path, overwrite=True)

    randoms = image.uniform_sphere((179., 181.), (-1., 1.), density=1e6)
    mask = image.single_image_coverage(test_path, randoms[:, 0], randoms[:, 1])
    assert np.isclose(len(randoms) / 4, mask.sum(), rtol=0.001)

    masked_randoms = randoms[ mask ]

    delta = 0.5 #np.sin(0.5 * np.pi / 180.) * 180. / np.pi
    ra_mask = ( 180. - delta < masked_randoms[:, 0]) & (masked_randoms[:, 0] < 180. + delta)
    dec_mask = ( -delta < masked_randoms[:, 1]) & (masked_randoms[:, 1] < delta)
    assert all( ra_mask & dec_mask )
    
def test__multi_image_coverage():
    pass

def test__objects_in_coverage():
    pass

def test__calc_survey_area():
    pass

def test__make_good_coverage_map():
    pass







