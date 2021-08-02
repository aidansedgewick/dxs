import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, concatenate
from astropy.io import fits
from astropy.wcs import WCS

from regions import PolygonSkyRegion, CircleSkyRegion, RectangleSkyRegion

from dxs.utils import misc
from dxs.utils import image

from dxs import paths


def test__build_mosaic_wcs():
    wcs1 = image.build_mosaic_wcs(
        SkyCoord(ra=313.2, dec=42.0, unit="deg"), size=(1800, 900), pixel_scale = 1.0
    )

    assert wcs1.calc_footprint().shape == (4, 2)

    assert wcs1.pixel_shape == (1800, 900)

    h = wcs1.to_header()
    assert np.isclose(h["CRVAL1"], 313.2)
    assert np.isclose(h["CRVAL2"], 42.0)
    assert np.isclose(h["CRPIX1"], 900)
    assert np.isclose(h["CRPIX2"], 450)
    
    # unsure how to properly test this...
    

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

def test__single_image_coverage():
    image_size = (1800, 1800)
    image_center = SkyCoord(ra=215., dec=30., unit="deg")
    pixel_scale = 1.0
    fwcs = image.build_mosaic_wcs(image_center, image_size[::-1], pixel_scale)
    header = fwcs.to_header()

    data = np.zeros(image_size)

    max_ra = 215.25
    max_dec = 30.25
    min_ra = 214.75
    min_dec = 29.75

    vertices = [ # go "anti-clockwise" in the diagram.
        (min_ra, max_dec), (max_ra, max_dec), (max_ra, min_dec), (min_ra, min_dec)
    ]

    vs = [v for v in vertices] + [vertices[0]]
    dx = 0.05
    coord_list = []
    for v1, v2 in zip(vs[:-1], vs[1:]):
        const_ra = np.isclose(v1[0], v2[0])
        const_dec = np.isclose(v1[1], v2[1])
        cosdec = np.cos(v1[1] * np.pi / 180.)
        N_cdec = int(abs(v1[0] - v2[0]) / (dx * cosdec))
        N_cra = int(abs(v1[1] - v2[1]) / dx)
        if const_ra and not const_dec:
            N = N_cra
        elif const_dec and not const_ra:
            N = N_cdec
        elif not const_ra and not const_dec:
            N = max(N_cdec, N_cra)
        elif const_ra and const_dec:
            raise ValueError("two consecutive points are the same.")
        
        cosdec = np.cos(v1[0] * np.pi / 180.)
        segment = [
            SkyCoord(ra=ra_ii, dec=dec_ii, unit="deg") for ra_ii, dec_ii in zip(
                np.linspace(v1[0], v2[0], N, endpoint=False), 
                np.linspace(v1[1], v2[1], N, endpoint=False)
            )
        ]
        coord_list.extend(segment)

    sph_coords = concatenate(coord_list)
    region = PolygonSkyRegion(sph_coords)
    pix_region = region.to_pixel(fwcs)
    mask = pix_region.to_mask()
    
    data = data + mask.to_image(image_size)

    hdu = fits.PrimaryHDU(data=data, header=header)
    coverage_path = paths.scratch_test_path / "test_single_coverage.fits"
    hdu.writeto(coverage_path, overwrite=True)

    randoms = image.uniform_sphere((214., 216.), (29., 31.), density=1e4)
    ra_mask = (min_ra < randoms[:,0]) & (randoms[:,0] < max_ra)
    dec_mask = (min_dec < randoms[:,1]) & (randoms[:,1] < max_dec)
    
    good_randoms = randoms[ ra_mask & dec_mask ]

    coverage_mask = image.single_image_coverage(
        coverage_path, randoms[:,0], randoms[:,1]
    )
    randoms_from_coverage = randoms[ coverage_mask ]

    assert np.isclose(len(good_randoms), len(randoms_from_coverage), rtol=0.005)


#def test__multi_image_coverage():
#    pass

#def test__objects_in_coverage():
#    pass


def test__calc_survey_area():
    ## high in the sky.
    image_size = (900, 900)
    img_center = SkyCoord(ra=180.0, dec=50.0, unit="degree")

    fwcs = image.build_mosaic_wcs(
        center=img_center, 
        size=image_size[::-1], 
        pixel_scale=120.0
    )
    h = fwcs.to_header()

    data = np.zeros(image_size)

    ## make this shape
        
    ##       B*     A*
    ## |-----|  s1  |------| dec_max
    ## |   C*|------|      |
    ## |               |---| s2_dec_max
    ## |               |
    ## |               | s2
    ## |               |
    ## |               |---| s2_dec_min
    ## |               D*  |
    ## |-------------------| dec_min
    #  ra_max         ra_min
    
    max_ra = 194.
    max_dec = 61.
    min_ra = 164.
    min_dec = 41. 
    s1_dec = 56. # C*
    s1_min_ra = 174.  # A*
    s1_max_ra = 184. # B*
    s2_ra = 169. # D*
    s2_min_dec = 46.
    s2_max_dec = 56.

    lsp = np.linspace

    vertices = [ # go "anti-clockwise" in the diagram.
        (min_ra, max_dec), (s1_min_ra, max_dec), (s1_min_ra, s1_dec), (s1_max_ra, s1_dec),
        (s1_max_ra, max_dec), (max_ra, max_dec), (max_ra, min_dec), (min_ra, min_dec),
        (min_ra, s2_min_dec), (s2_ra, s2_min_dec), (s2_ra, s2_max_dec), (min_ra, s2_max_dec),
    ]

    vs = [v for v in vertices] + [vertices[0]]
    dx = 0.50
    coord_list = []
    for v1, v2 in zip(vs[:-1], vs[1:]):
        const_ra = np.isclose(v1[0], v2[0])
        const_dec = np.isclose(v1[1], v2[1])
        cosdec = np.cos(v1[1] * np.pi / 180.)
        N_cdec = int(abs(v1[0] - v2[0]) / (dx * cosdec))
        N_cra = int(abs(v1[1] - v2[1]) / dx)
        if const_ra and not const_dec:
            N = N_cra
        elif const_dec and not const_ra:
            N = N_cdec
        elif not const_ra and not const_dec:
            N = max(N_cdec, N_cra)
        elif const_ra and const_dec:
            raise ValueError("two consecutive points are the same.")
        
        cosdec = np.cos(v1[0] * np.pi / 180.)
        segment = [
            SkyCoord(ra=ra_ii, dec=dec_ii, unit="deg") for ra_ii, dec_ii in zip(
                np.linspace(v1[0], v2[0], N, endpoint=False), 
                np.linspace(v1[1], v2[1], N, endpoint=False)
            )
        ]
        coord_list.extend(segment)

    sph_coords = concatenate(coord_list)
    region = PolygonSkyRegion(sph_coords)
    pix_region = region.to_pixel(fwcs)
    mask = pix_region.to_mask()
    
    data = data + mask.to_image(image_size)

    hdu = fits.PrimaryHDU(data=data, header=h)
    image_path_1 = paths.scratch_test_path / "survey_area_test_1.fits"
    hdu.writeto(image_path_1, overwrite=True)

    main_area = image.calc_spherical_rectangle_area((min_ra, max_ra), (min_dec, max_dec))
    s1_area = image.calc_spherical_rectangle_area((s1_min_ra, s1_max_ra), (s1_dec, max_dec))
    s2_area = image.calc_spherical_rectangle_area((min_ra, s2_ra), (s2_min_dec, s2_max_dec))

    cov_area = main_area - s1_area - s2_area

    est_area = image.calc_survey_area(image_path_1, density=1e4)
    
    assert np.isclose(cov_area, est_area, rtol=0.005)


def test__make_good_coverage_map():
    image_center = SkyCoord(ra=90., dec=45., unit="deg")
    image_size = (1000, 1000)
    pixel_scale = 1.0
    
    fwcs = image.build_mosaic_wcs(image_center, image_size[::-1], pixel_scale)
    header=fwcs.to_header()

    data = np.zeros(image_size)

    data[:,:] = 1.

    data[200:800, 200:800] = 4. # big box, 600 x 600
    data[250:750, 250:750] = 7. # smaller box 500 x 500 
    data[400:600, 250:750] = 9. # tall stripe, height of big box. 200 x 500

    # so we're sure what we're putting in.
    assert np.sum(data == 1. ) == 4 * 600 * 200 + 4 * 200 * 200
    assert np.sum(data == 4.) == 4 * 500 * 50 + 4 * 50 * 50 # edge stripes + corners
    assert np.sum(data == 7.) == 2 * 500 * 150
    assert np.sum(data == 9.) == 200 * 500

    input_coverage_path = paths.scratch_test_path / "input_test_mkcov.fits"
    output_coverage_path = paths.scratch_test_path / "output_test_mkcov.fits"

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(input_coverage_path, overwrite=True)

    image.make_good_coverage_map(
        input_coverage_path,
        minimum_coverage=7.,
        output_path=output_coverage_path,
        dilation_iterations=20
    )

    with fits.open(output_coverage_path) as f:
        rdat = f[0].data


    ## 4 4 4 4 4 4 4 4 4 4          0 0 0 0 0 0 0 0 0 0
    ## 4 7 7 7 9 9 7 7 7 4          0 0 0 0 0 0 0 0 0 0
    ## 4 7 7 7 9 9 7 7 7 4          0 0 7 7 9 9 7 7 0 0
    ## 4 7 7 7 9 9 7 7 7 4          0 0 7 7 9 9 7 7 0 0
    ## 4 7 7 7 9 9 7 7 7 4   --->   0 0 7 7 9 9 7 7 0 0
    ## 4 7 7 7 9 9 7 7 7 4          0 0 7 7 9 9 7 7 0 0 
    ## 4 7 7 7 9 9 7 7 7 4          0 0 7 7 9 9 7 7 0 0
    ## 4 7 7 7 9 9 7 7 7 4          0 0 0 0 0 0 0 0 0 0
    ## 4 4 4 4 4 4 4 4 4 4          0 0 0 0 0 0 0 0 0 0

    assert np.sum(rdat == 1.) == 0
    assert np.sum(rdat == 4.) == 0
    assert np.sum(rdat == 7.) == 2 * (500 - 20 - 20) * (150 - 20) # -20 for stripe of 20 taken off.
    # take 20 off bottom, and top, 20 off right, x2 for "mirror".
    assert np.sum(rdat == 9.) == 200 * (500 - 20 - 20)
    # width of rows of 9s have not changed - only number of rows.

def test__mask_regions_in_mosaic():
    image_center = SkyCoord(ra=90., dec=45., unit="deg")    
    dec_size = 3600
    ra_size = int(3600 * np.cos(image_center.dec))

    image_size = (dec_size, ra_size)
    pixel_scale = 1.0
    
    fwcs = image.build_mosaic_wcs(image_center, image_size[::-1], pixel_scale)
    header = fwcs.to_header()

    data = abs(np.random.normal(5, 1, image_size)) + 0.1 
    assert np.sum(data < 1e-8) == 0 # NOTHING is zero.

    hdu = fits.PrimaryHDU(data = data, header=header)
    input_path = paths.scratch_test_path / "test_mask_regions_input.fits"
    output_path = paths.scratch_test_path / "test_mask_regions_output.fits"
    hdu.writeto(input_path, overwrite=True)


    expand_border = 500

    c1 = image_center
    c1_xpix, c1_ypix = c1.to_pixel(fwcs)
    circle_radius = 20. 
    c1_kwargs = {"radius": circle_radius * u.arcsec}
    assert c1.contained_by(fwcs) # duh

    c2 = SkyCoord(ra=90.2, dec=45.25, unit="deg")
    c2_xpix, c2_ypix = c2.to_pixel(fwcs)
    c2_kwargs = {"width": 0.2 * u.deg, "height": 40. * u.arcsec, "angle": 45. * u.deg}
    assert c2.contained_by(fwcs)

    c3 = SkyCoord(ra=89.4, dec=45., unit="deg")
    c3_xpix, c3_ypix = c3.to_pixel(fwcs)
    c3_kwargs = {"width": 0.2 * u.deg, "height": 40. * u.arcsec}
    assert not c3.contained_by(fwcs) # deliberately outside the image.
    assert ((ra_size < c3_xpix) & (c3_xpix < ra_size + expand_border)) # far away, but not too far.

    c4 = SkyCoord(ra=90.8, dec=45., unit="deg")
    c4_kwargs = {"width": 0.4 * u.deg, "height": 40. * u.arcsec}
    c4_xpix, c4_ypix = c4.to_pixel(fwcs)
    assert not c4.contained_by(fwcs)
    assert c4_xpix < 0 - expand_border # very faw away

    test_regions = [
        CircleSkyRegion(center=c1, **c1_kwargs),
        RectangleSkyRegion(center=c2, **c2_kwargs),
        RectangleSkyRegion(center=c3, **c3_kwargs),
        RectangleSkyRegion(center=c4, **c4_kwargs),
    ]

    image.mask_regions_in_mosaic(
        input_path, test_regions, output_path=output_path
    )

    with fits.open(output_path) as f:
        rdat = f[0].data


    # First, check the circle in the center has some zeros
    c1y = int(c1_ypix)
    c1x = int(c1_xpix)
    circle_center = np.sum(rdat[c1y-5:c1y+5, c1x-5:c1x+5])
    assert np.sum(circle_center) < 1e-16

    # Now check the number of zero pixels is consistent(ish) with pi r**2.
    xpix_width = circle_radius * pixel_scale #* np.cos(c1.dec) # SkyRegion accounts for this!
    ypix_width = circle_radius * pixel_scale
    N_zero_pix_expected = int(np.pi * ypix_width * xpix_width) # pi*a*b = ellipse area.
    circle_postage_stamp = rdat[c1y-30:c1y+30, c1x-30:c1x+30]
    N_zero_pix_actual = np.sum(circle_postage_stamp <1e-15)
    assert np.isclose(N_zero_pix_actual, N_zero_pix_expected, rtol=0.005)


    # look at the diagonal rectangle. Has it correctly gone diagonal?
    # (ie, not zero'ed the whole bounding box).
    c2_tp1 = SkyCoord(ra=90.15, dec=45.22, unit="deg") # still within the bbox of shape 2.
    c2tp1_xpix, c2tp1_ypix = c2_tp1.to_pixel(fwcs)
    c2tp1x = int(c2tp1_xpix)
    c2tp1y = int(c2tp1_ypix)
    assert (rdat[c2tp1y-20:c2tp1y+20, c2tp1x-20:c2tp1x+20] > 0.1).all()

    c3_test_point = SkyCoord(ra=89.52, dec=45., unit="deg")
    c3tp_xpix, c3tp_ypix = c3_test_point.to_pixel(fwcs)
    c3tpx = int(c3tp_xpix)
    c3tpy = int(c3tp_ypix)
    rectangle_portion = rdat[c3tpy-10:c3tpy+10, c3tpx-10:c3tpx+10]
    assert np.sum(rectangle_portion) < 1e-6





