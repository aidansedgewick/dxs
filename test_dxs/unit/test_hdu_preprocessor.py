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

    assert hdup.fill_value is None
    assert np.isclose(hdup.fill_value_var, 1e-3)
    assert hdup.exptime is None
    assert hdup.add_flux_scale is False
    
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
    
def test__bgr_subtract():
    """
    This test is not great...
    """

    image_size = (2048, 2048)
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
    
def test__mask_sources():
    
    




