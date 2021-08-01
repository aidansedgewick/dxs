import os
import pytest

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits

from dxs.utils.image import build_mosaic_wcs

from dxs.hdu_preprocessor import HDUPreprocessor, HDUPreprocessorError, get_hdu_name

from dxs import paths

def make_data_header(image_size):
    pixel_scale = 1.0
    wcs = build_mosaic_wcs(SkyCoord(180., 45., unit="deg"), image_size[::-1], pixel_scale)
    header = wcs.to_header()
    data = np.random.normal(4, 0.1, image_size)

    return data, header

def test__hdu_preprocessor_initialisation():
    
    image_size = (500, 500)
    data, header = make_data_header(image_size)

    data[0,100] = 1000

    hdu_output_path = paths.scratch_test_path / "hduprep_init.fits"
    if hdu_output_path.exists():
        os.remove(hdu_output_path)
    assert not hdu_output_path.exists()

    hdu = fits.PrimaryHDU(data=data, header=header)

    hdup = HDUPreprocessor(
        hdu, hdu_output_path, 
    )

    assert hdup.data.shape == (500, 500)
    assert hdup.data[0,100] == 1000
    assert hdup.hdu_output_path.name == "hduprep_init.fits"
    assert hdup.xlen == 500
    assert hdup.ylen == 500

    assert hdup.edges is None

    assert hdup.fill_value is None
    assert np.isclose(hdup.fill_value_var, 1e-3)

    assert hdup.subtract_bgr is False
    assert hdup.bgr_size is None
    assert hdup.filter_size == (1, 1)
    assert np.isclose(hdup.sigma, 3.)

    assert hdup.mask_sources is False
    assert hdup.mask_wcs is None
    assert hdup.mask_map is None

    assert hdup.exptime is None
    assert hdup.add_flux_scale is False
    assert hdup.magzpt is None
    assert hdup.reference_magzpt is None
    
    hdup.prepare_hdu()
    assert hdu_output_path.exists()
    with fits.open(hdu_output_path) as f:
        assert "EXP_TIME" not in f[0].header
        assert "MAGZPT" not in f[0].header
        assert "FLXSCALE" not in f[0].header
    
def test__hdu_preprocessor_fill():
    
    image_size = (1000, 1000)
    data, header = make_data_header(image_size)
    hdu = fits.PrimaryHDU(data=data, header=header)

    hdu_output_path1 = paths.scratch_test_path / "hduprep_fill_test1.fits"
    if hdu_output_path1.exists():
        os.remove(hdu_output_path1)
    assert not hdu_output_path1.exists()
    hdu_proc1 = HDUPreprocessor(hdu, hdu_output_path1, fill_value=3., fill_value_var=1e-4)
    hdu_proc1.prepare_hdu()
    assert hdu_output_path1.exists()    

    with fits.open(hdu_output_path1) as f1:
        mean_val = np.mean(f1[0].data)
        std_val = np.std(f1[0].data)
        assert np.isclose(mean_val, 3.)
        assert np.isclose(std_val, 1e-4, rtol=0.005)

    hdu_output_path2 = paths.scratch_test_path / "hduprep_fill_test2.fits"
    if hdu_output_path2.exists():
        os.remove(hdu_output_path2)
    assert not hdu_output_path2.exists()
    hdu_proc2 = HDUPreprocessor(
        hdu, hdu_output_path2, fill_value="exptime", exptime=10.
    )
    assert np.isclose(hdu_proc2.fill_value, 10.)
    assert np.isclose(hdu_proc2.exptime, 10.)
    hdu_proc2.prepare_hdu()
    assert hdu_output_path2.exists()
    
    with fits.open(hdu_output_path2) as f2:
        mean_val = np.mean(f2[0].data)
        assert np.isclose(mean_val, 10.)
        assert np.isclose(f2[0].header["EXP_TIME"], 10.)
        
    with pytest.raises(HDUPreprocessorError):
        hdu_output_path3 = paths.scratch_test_path / "hduprep_fill_test3.fits"
        hdu_proc3 = HDUPreprocessor(hdu, hdu_output_path3, fill_value="exptime")
        # NOT providing exptime... test it raises the error.

    with pytest.raises(HDUPreprocessorError):
        hdu_output_path4 = paths.scratch_test_path / "hduprep_fill_test4.fits"
        hdu_proc4 = HDUPreprocessor(hdu, hdu_output_path3, fill_value=3., subtract_bgr=True)
        # Try to subtract fill value

    with pytest.raises(HDUPreprocessorError):
        hdu_output_path5 = paths.scratch_test_path / "hduprep_fill_test5.fits"
        hdu_proc5 = HDUPreprocessor(hdu, hdu_output_path3, fill_value=3., mask_sources=True)
        # Try to subtract fill value
        
def test__trim_edges():
    image_size = (500, 500)
    data, header = make_data_header(image_size)
    hdu = fits.PrimaryHDU(data=data, header=header)

    hdu_output_path = paths.scratch_test_path / "hduprep_edges_test.fits"
    if hdu_output_path.exists():
        os.remove(hdu_output_path)
    hdu_proc = HDUPreprocessor(
        hdu, hdu_output_path, edges=25
    )
    hdu_proc.prepare_hdu()
    assert hdu_output_path.exists()
    with fits.open(hdu_output_path) as f:
        rdat = f[0].data
        assert rdat.shape == (450, 450)

def test__get_bgr():
    image_size = (500, 500)
    data, header = make_data_header(image_size)
    hdu = fits.PrimaryHDU(data=data, header=header)

    hdu_output_path = paths.scratch_test_path / "hdu_prep_getbgr.fits"
    if hdu_output_path.exists():
        os.remove(hdu_output_path)
    assert not hdu_output_path.exists()
    hdu_proc = HDUPreprocessor(hdu, hdu_output_path, bgr_size=25)
    
    assert hdu_proc.bgr_size == (25, 25)
    bgr = hdu_proc.get_background()
    bgr_map = bgr.background

    assert bgr_map.shape == (500, 500)

    assert not hdu_output_path.exists() # we should NOT have written any data.
    
def test__bgr_subtract():
    """
    This test is not great...
    """

    image_size = (1024, 1024)
    data, header = make_data_header(image_size)

    assert np.isclose(np.mean(data), 4., rtol=0.01) # average data is 4.
    xsize = image_size[1]

    # add a wave.
    wave = 1. * np.sin(np.linspace(0, 4 * 2*np.pi, xsize))
    data = data + wave

    hdu_input_path = paths.scratch_test_path / "hduprep_bgr_subtract_input.fits"
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(hdu_input_path, overwrite=True)

    rms_data = np.sqrt(np.mean(data * data))

    rms_sin = 1. / np.sqrt(2.)
    rms_const = 4.
    rms_expect = np.sqrt(rms_sin * rms_sin + rms_const * rms_const)
    
    assert np.isclose(rms_data, rms_expect, rtol=0.005)

    hdu_output_path = paths.scratch_test_path / "hduprep_bgr_subtract.fits"
    if hdu_output_path.exists():
        os.remove(hdu_output_path)
    assert not hdu_output_path.exists()


    hdu_proc = HDUPreprocessor(
        hdu, hdu_output_path, subtract_bgr=True, bgr_size=32
    )
    assert hdu_proc.bgr_size == (32, 32)
    assert hdu_proc.filter_size == (1, 1)
    hdu_proc.prepare_hdu()
    assert hdu_output_path.exists()

    with fits.open(hdu_output_path) as f:
        rdat = f[0].data
        
    ## mean value of background in 25x25 squares? 
    ## move by 0.001 each pixel. so 0.0125?    

    #assert np.iscnp.mean(rdat[:,0])

    rdat_mean = np.mean(rdat)
    assert np.isclose(rdat_mean, 0., atol=0.005)
    
    #rms_output = np.sqrt(np.mean(rdat * rdat))
    #assert np.isclose(rms_output, 0.1, rtol=0.1) # the rms should be like the input stdev.

def test__get_source_mask():

    image_size = (500, 500)
    data, header = make_data_header(image_size)
    hdu = fits.PrimaryHDU(data=data, header=header)
    
    # first test exception if no mask_map/mask_hdu is provided
    hdu_output_path1 = paths.scratch_test_path / "hduprep_mask_sources1.fits"
    if hdu_output_path1.exists():
        os.remove(hdu_output_path1)
    assert not hdu_output_path1.exists()

    with pytest.raises(HDUPreprocessorError):
        failing_hdu_proc1 = HDUPreprocessor(
            hdu, hdu_output_path1, mask_sources=True, mask_map=np.zeros(image_size)
        )
    with pytest.raises(HDUPreprocessorError):
        failing_hdu_proc2 = HDUPreprocessor(
            hdu, hdu_output_path1, mask_sources=True, mask_header=header
        )

    ## test we actually get a map the right shape back.
    mask1_size = (800, 600) # having an oversized mask is the usual case.
    mask1_wcs = build_mosaic_wcs(
        SkyCoord(ra=180., dec=45., unit="deg"), mask1_size[::-1], 1.0
    )
    mask1_map = np.full(mask1_size, False) # No masking.
    mask1_map[390:410, 290:310] = True
    mask1_header = mask1_wcs.to_header()
    hdu_proc1 = HDUPreprocessor(
        hdu, hdu_output_path1, 
        mask_sources=True, mask_header=mask1_header, mask_map=mask1_map
    )
    
    source_mask1 = hdu_proc1.get_source_mask()
    assert source_mask1.shape == (500, 500)
    assert source_mask1[240:260, 240:260].all()
    assert not source_mask1[230:270, 230:270].all() # Check we've not gone mad...
    assert np.sum(source_mask1) == 20 * 20 ## ie, the right number.
 
    ## what happens when the mask is a different resolution?
    mask2_size = (250, 250)
    mask2_wcs = build_mosaic_wcs(
        SkyCoord(ra=180., dec=45., unit="deg"), mask2_size[::-1], 2.0
    )
    mask2_header = mask2_wcs.to_header()
    mask2_map = np.full(mask2_size, False)
    mask2_map[120:130, 120:130] = True

    hdu_proc2 = HDUPreprocessor(
        hdu, hdu_output_path1, 
        mask_sources=True, mask_header=mask2_header, mask_map=mask2_map
    )

    source_mask2 = hdu_proc2.get_source_mask()
    assert source_mask2.shape == (500, 500) # ie, upscaled to data shape.
    assert np.sum(source_mask2) == 20 * 20 # 10x10 upscaled.


    ## what happens when the mask does not cover the whole hdu data??
    mask3_size = (600, 600)
    mask3_wcs = build_mosaic_wcs(
        SkyCoord(ra=180., dec=45. + 500. / 3600., unit="deg"), mask3_size[::-1], 1.0
    )
    mask3_header = mask3_wcs.to_header()
    mask3_map = np.full(mask3_size, True)

    hdu_proc3 = HDUPreprocessor(
        hdu, hdu_output_path1,
        mask_sources=True, mask_header=mask3_header, mask_map=mask3_map
    )
    source_mask3 = hdu_proc3.get_source_mask()
    assert source_mask3.shape == (500, 500) # we still get the same shape back.
    assert np.sum(source_mask3[-10:, :]) == 500 * 10 # everything along the top row is masked.
    assert np.sum(source_mask3[:10, :]) == 0  # bottom row (outside mask_map) is filled with False

    ### we should NOT have written anything along the way.
    assert not hdu_output_path1.exists()

def test__mask_sources():
    pass

def test__magzpts():
    image_size = (500, 500)

    cleanup_paths = []

    ## only give magzpt.
    data1, header1 = make_data_header(image_size)
    header1["MAGZPT"] = 25.0
    hdu1 = fits.PrimaryHDU(data=data1, header=header1)
    hdu_output_path1 = paths.scratch_test_path / "hdu_prep_magzpts1.fits"
    cleanup_paths.append(hdu_output_path1)
    hdu_proc1 = HDUPreprocessor(hdu1, hdu_output_path1)
    assert np.isclose(hdu_proc1.magzpt, 25.0)
    hdu_proc1.prepare_hdu()
    with fits.open(hdu_output_path1) as f1:
        assert np.isclose(f1[0].header["MAGZPT"], 25.0)

    ## magzpt and exptime
    data2, header2 = make_data_header(image_size)
    header2["MAGZPT"] = 24.0
    exptime2 = 10.
    hdu2 = fits.PrimaryHDU(data=data2, header=header2)
    hdu_output_path2 = paths.scratch_test_path / "hdu_prep_magzpts2.fits"
    cleanup_paths.append(hdu_output_path2)
    hdu_proc2 = HDUPreprocessor(hdu2, hdu_output_path2, exptime=exptime2)
    assert np.isclose(hdu_proc2.magzpt, 26.5) # 24 + 2.5log10(10)
    hdu_proc2.prepare_hdu()
    with fits.open(hdu_output_path2) as f2:
        assert np.isclose(f2[0].header["MAGZPT"], 24.0)
        assert np.isclose(f2[0].header["EXP_TIME"], 10.)
        
    ## magzpt, exptime and ref.
    data3, header3 = make_data_header(image_size)
    header3["MAGZPT"] = 27.0
    exptime3 = 10.
    hdu3 = fits.PrimaryHDU(data=data3, header=header3)
    hdu_output_path3 = paths.scratch_test_path / "hdu_prep_magzpts3.fits"
    cleanup_paths.append(hdu_output_path3)
    hdu_proc3 = HDUPreprocessor(
        hdu3, hdu_output_path3, exptime=exptime3, reference_magzpt=25.0
    )
    assert np.isclose(hdu_proc3.magzpt, 29.5) # 24 + 2.5log10(10)
    assert np.isclose(hdu_proc3.reference_magzpt, 25.)
    hdu_proc3.prepare_hdu()
    with fits.open(hdu_output_path3) as f3:
        assert np.isclose(f3[0].header["MAGZPT"], 27.0)
    
    # exptime and ref only.
    data4, header4 = make_data_header(image_size)
    exptime4 = 10.
    hdu4 = fits.PrimaryHDU(data=data4, header=header4)
    hdu_output_path4 = paths.scratch_test_path / "hdu_prep_magzpts4.fits"
    cleanup_paths.append(hdu_output_path4)
    hdu_proc4 = HDUPreprocessor(
        hdu4, hdu_output_path4, exptime=exptime4, reference_magzpt=26.0
    )
    assert np.isclose(hdu_proc4.magzpt, 28.5) # 24 + 2.5log10(10)
    assert np.isclose(hdu_proc4.reference_magzpt, 26.)
    hdu_proc4.prepare_hdu()
    with fits.open(hdu_output_path4) as f4:
        assert "MAGZPT" not in f4[0].header

    # AB_conversion & overwrite.
    data5, header5 = make_data_header(image_size)
    header5["MAGZPT"] = 20.0
    exptime5 = 10.
    hdu5 = fits.PrimaryHDU(data=data5, header=header5)
    hdu_output_path5 = paths.scratch_test_path / "hdu_prep_magzpts5.fits"
    cleanup_paths.append(hdu_output_path5)
    hdu_proc5 = HDUPreprocessor(
        hdu5, hdu_output_path5, exptime=exptime5, 
        AB_conversion = 1.900, overwrite_magzpt=True
    )
    assert np.isclose(hdu_proc5.magzpt, 24.4) # 20 + 2.5log10(10) + 1.9
    hdu_proc5.prepare_hdu()
    with fits.open(hdu_output_path5) as f5:
        assert np.isclose(f5[0].header["MAGZPT"], 24.4)

    for path_ii in cleanup_paths:
        assert path_ii.exists()
        os.remove(path_ii)
        assert not path_ii.exists()
    
def test__add_flux_scale():
    image_size = (500, 500)
    data, header = make_data_header(image_size)
    header["MAGZPT"] = 22.5
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu_output_path = paths.scratch_test_path / "hduprep_add_flxscale.fits"
    
    with pytest.raises(HDUPreprocessorError):
        failing_hdu_proc1 = HDUPreprocessor(
            hdu, hdu_output_path, add_flux_scale=True
        )

    hdu_proc = HDUPreprocessor(
        hdu, hdu_output_path, exptime=10., 
        add_flux_scale=True, reference_magzpt=26.
    )
    hdu_proc.prepare_hdu()

    # flxscale will be 10** -0.4 * ( (22.5+2.5log10(10.) - 26.) = 10**0.4s
    with fits.open(hdu_output_path) as f:
        assert np.isclose(f[0].header["FLXSCALE"], 2.51188643150958)

def test__prepare_stack():
    stack_path = paths.scratch_test_path / "hduprep_test_stack.fits"
    if stack_path.exists():
        os.remove(stack_path)
    assert not stack_path.exists()

    image_size = (200, 200)
    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header["EXP_TIME"] = 10.
    hdu_list = fits.HDUList([prim_hdu])
    for ii in range(4):
        data, header = make_data_header(image_size)
        header["MAGZPT"] = 20. + 0.5 * ii # so each one is different.
        data = data + ii
        hdu_ii = fits.ImageHDU(data=data, header=header)
        hdu_list.append(hdu_ii)
    hdu_list.writeto(stack_path)

    expected_paths = [
        paths.scratch_test_path / f"XYZhduprep_test_stack_0{ii}.fits" for ii in [1,2,4]
    ]
    for ep in expected_paths:
        if ep.exists():
            os.remove(ep)
        assert not ep.exists()

    ccds = [1,2,4] # SKIP CCD 3!!!
    kwargs = {"reference_magzpt": 24., "add_flux_scale": True}

    output_list = HDUPreprocessor.prepare_stack(
        stack_path, hdu_prefix="XYZ", ccds=ccds, output_dir=paths.scratch_test_path, **kwargs
    )
    assert len(output_list) == 3
    assert set(output_list) == set(expected_paths)
    
    with fits.open(expected_paths[0]) as f1:
        assert np.isclose(f1[0].header["MAGZPT"], 20.)
        #expected flxscale: 10^(-0.4*(20+(0.5*0)+2.5log10(10.) - 24 )
        assert np.isclose(f1[0].header["FLXSCALE"], 3.981071705534973)

    with fits.open(expected_paths[1]) as f2:
        assert np.isclose(f2[0].header["MAGZPT"], 20.5)
        #expected flxscale: 10^(-0.4*(20+(0.5*1)+2.5log10(10.) - 24 )
        assert np.isclose(f2[0].header["FLXSCALE"], 2.51188643150958)

    with fits.open(expected_paths[2]) as f3:
        assert np.isclose(f3[0].header["MAGZPT"], 21.5)
        #expected flxscale: 10^(-0.4*(20+(0.5*3)+2.5log10(10.) - 24 )
        assert np.isclose(f3[0].header["FLXSCALE"], 1.)

def test__get_hdu_name():
    assert get_hdu_name("stack", 1) == "stack_01.fits"
    assert get_hdu_name("stack", 2, weight=True) == "stack_02.weight.fits"
    assert get_hdu_name("stack", 3, prefix="NA13L") == "NA13Lstack_03.fits"
    test_path = paths.scratch_test_path / "stack.many.extensions.fits"
    all_kwarg_test = get_hdu_name(test_path, 4, weight=True, prefix="XYZ")
    assert all_kwarg_test == "XYZstack.many.extensions_04.weight.fits"













