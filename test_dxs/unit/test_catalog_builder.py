import os
import pytest

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits

from dxs import catalog_builder
from dxs.utils.image import build_mosaic_wcs, uniform_sphere

from dxs import paths

def make_data_header():
    hdu_wcs = build_mosaic_wcs(
        SkyCoord(ra=90., dec=30., unit="deg"), (1000, 1000), 1.0
    )
    header = hdu_wcs.to_header()
    data = np.random.uniform(5, 1, (1000, 1000))
    return data, header
        
    
def test__catalog_extractor_init():
    mosaic_path = paths.scratch_test_path / "ce_init/catext_init.fits"

    ## some cleanup.
    cat_aux_dir = paths.catalogs_path / "catext_init/aux"
    if cat_aux_dir.exists():
        os.rmdir(cat_aux_dir)
    assert not cat_aux_dir.exists()
    cat_dir = paths.catalogs_path / "catext_init"
    if cat_dir.exists():
        os.rmdir(cat_dir)
    assert not cat_dir.exists()

    catext = catalog_builder.CatalogExtractor(
        mosaic_path, 
        sextractor_config_file=paths.scratch_test_path / "s_config.sex",
        #sextractor_param_file=paths.scratch_test_path / "s_parameters.param"
    )

    assert str(catext.detection_mosaic_path) == str(mosaic_path)
    assert catext.measurement_mosaic_path is None

    exp_cat_path = str(paths.catalogs_path / "catext_init/catext_init.cat.fits")
    assert str(catext.catalog_path) == exp_cat_path
    assert cat_dir.exists()
    assert cat_aux_dir.exists()

    assert catext.use_weight
    exp_weight_path = str(paths.scratch_test_path / "ce_init/catext_init.weight.fits")
    assert str(catext.weight_path) == exp_weight_path

    exp_config_file = str(paths.scratch_test_path / "s_config.sex")
    assert str(catext.sextractor_config_file) == exp_config_file
    exp_param_file = str(paths.config_path / "sextractor/indiv.param")
    assert str(catext.sextractor_parameter_file) == exp_param_file

    exp_seg_path = str(paths.scratch_test_path / "ce_init/catext_init.seg.fits")
    assert str(catext.segmentation_mosaic_path) == exp_seg_path

    os.rmdir(cat_aux_dir)
    assert not cat_aux_dir.exists()
    os.rmdir(cat_dir)
    assert not cat_dir.exists()

def test__catalog_extractor_from_dxs_spec():
    test_config = {"awooooo": 99}
    config_file = paths.config_path / "blahblahblah.config"
    param_file = paths.config_path / "waahwaahwaah.param"
    ce = catalog_builder.CatalogExtractor.from_dxs_spec(
        "QQ", 99, "B", 
        measurement_band="C", 
        sextractor_config=test_config,
        sextractor_config_file=config_file,
        sextractor_parameter_file=param_file,
    )
    
    exp_det_path = str(paths.mosaics_path / "QQ99B/QQ99B.fits")
    assert str(ce.detection_mosaic_path) == exp_det_path
    exp_meas_path = str(paths.mosaics_path / "QQ99C/QQ99C.fits")
    assert str(ce.measurement_mosaic_path) == exp_meas_path
    
    exp_cat_path = str(paths.catalogs_path / "QQ99/QQ99BCfp.cat.fits")
    assert str(ce.catalog_path) == exp_cat_path

    assert ce.sextractor_config["awooooo"] == 99
    assert str(ce.sextractor_config_file) == str(config_file)
    assert str(ce.sextractor_parameter_file) == str(param_file)

    cat_aux_dir = paths.catalogs_path / "QQ99/aux"
    assert cat_aux_dir.exists()
    os.rmdir(cat_aux_dir)
    assert not cat_aux_dir.exists()

    cat_dir = paths.catalogs_path / "QQ99"
    assert cat_dir.exists()
    os.rmdir(cat_dir)
    assert not cat_dir.exists()    
    

def test__build_sextractor_config():
    det_mosaic_path = paths.scratch_test_path / "cf_build_det.fits"
    data, det_header = make_data_header()
    det_header["MAGZPT"] = 20.
    det_header["SEEING"] = 1.0
    det_hdu = fits.PrimaryHDU(data=data, header=det_header)
    det_hdu.writeto(det_mosaic_path, overwrite=True)   

    meas_mosaic_path = paths.scratch_test_path / "cf_build_meas.fits"
    data, meas_header = make_data_header()
    meas_header["MAGZPT"] = 25.
    meas_header["SEEING"] = 2.0
    meas_hdu = fits.PrimaryHDU(data=data, header=meas_header)
    meas_hdu.writeto(meas_mosaic_path, overwrite=True)   

    ce = catalog_builder.CatalogExtractor(
        det_mosaic_path,
        measurement_mosaic_path=meas_mosaic_path, 
        sextractor_config={"backphoto_type": "local"}
    )

    config = ce.build_sextractor_config()

    exp_catalog_name = str(paths.catalogs_path / "cf_build_det/cf_build_detcf_build_meas.cat.fits")
    exp_seg_name = str(paths.scratch_test_path / "cf_build_det.seg.fits")

    assert config["CATALOG_NAME"] == exp_catalog_name
    assert config["PARAMETERS_NAME"] == str(paths.config_path / "sextractor/indiv.param")
    assert config["CHECKIMAGE_TYPE"] == "SEGMENTATION"
    assert config["CHECKIMAGE_NAME"] == exp_seg_name
    assert config["WEIGHT_TYPE"] == "background"
    
    assert config["SEEING_FWHM"] == "1.000000"
    assert config["MAG_ZEROPOINT"] == "25.000000"

    os.rmdir(paths.catalogs_path / "cf_build_det/aux")
    os.rmdir(paths.catalogs_path / "cf_build_det")

def test__add_snr():

    mosaic_path = paths.scratch_test_path / "ce_snr_mosaic.fits"
    catalog_path1 = paths.scratch_test_path / "ce_add_snr.cat.fits"
    cat1 = Table({
        "ra": [1., 2., 3., 4., 5.,],
        "dec": [0., 0., 0., 0., 0.,],
        "flux1": [10., 20., 30., 40., 50.],
        "fluxerr1": [10., 10., 5., 4., 1.],
        "flux2": [10., 10., 10., 10., 10.],
        "fluxerr2": [1., 1., 1., 1., np.nan],
    })
    if catalog_path1.exists():
        os.remove(catalog_path1)
    cat1.write(catalog_path1)

    ce1 = catalog_builder.CatalogExtractor(
        mosaic_path, catalog_path=catalog_path1
    )
    ce1.add_snr(["flux1", "flux2"], ["fluxerr1", "fluxerr2"], ["snr1", "snr2"])
    
    rcat1 = Table.read(catalog_path1)
    assert "snr1" in rcat1.columns
    assert "snr2" in rcat1.columns

    assert np.allclose(rcat1["snr1"], [1., 2., 6., 10., 50.])
    assert np.allclose(rcat1["snr2"], [10., 10., 10., 10., 0.])
    
    ## test not giving enough column names.
    
    catalog_path2 = paths.scratch_test_path / "ce_add_snr.cat.fits"
    cat2 = Table({
        "ra": [1., 2., 3., 4., 5.,],
        "dec": [0., 0., 0., 0., 0.,],
        "flux1": [10., 20., 30., 40., 50.],
        "fluxerr1": [10., 10., 10., 10., 10.],
        "flux2": [10., 10., 10., 10., 10.],
        "fluxerr2": [1., 1., 1., 1., 1.],
        
    })
    if catalog_path2.exists():
        os.remove(catalog_path2)
    cat2.write(catalog_path1)

    ce2 = catalog_builder.CatalogExtractor(
        mosaic_path, catalog_path=catalog_path2
    )
    with pytest.raises(ValueError):
        ce2.add_snr(["flux1", "flux2"], ["fluxerr1"], ["snr1", "snr2"])
    with pytest.raises(ValueError):
        ce2.add_snr(["flux1", "flux2"], ["fluxerr1", "fluxerr2"], ["snr1"])
    tab = Table.read(catalog_path2)
    assert "snr1" not in tab.columns
    assert "snr2" not in tab.columns

    # test give non-list input works.
    ce2.add_snr("flux1", "fluxerr1", "snr1")
    tab = Table.read(catalog_path2)
    assert np.allclose(tab["snr1"], [1., 2., 3., 4., 5.])
    assert "snr2" not in tab.columns



### =========== now test catalog_matcher =========== ###

def test__catalog_matcher_init():
    input_path = paths.scratch_test_path / "cat_matcher_init.cat.fits"

    matcher1 = catalog_builder.CatalogMatcher(input_path)
    assert matcher1.catalog_path == input_path
    assert matcher1.ra == "ra"
    assert matcher1.dec == "dec"

    matcher2 = catalog_builder.CatalogMatcher(input_path, ra="ra_test", dec="dec_test")
    assert matcher2.catalog_path == input_path
    assert matcher2.ra == "ra_test"
    assert matcher2.dec == "dec_test"

def test__catalog_matcher_match():
    
    tab1 = Table({
        "id1":  [101 , 102 , 103 , 104 , 105 , 106 , 107 , 108 , 109 ,      ],
        "ra1":  [90. , 90.1, 90.2, 90.3, 90.4, 90.5, 90.6, 90.7, 90.8,      ],
        "dec1": [0.  , 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,        ],
        "J_m":  [10. , 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,       ],
    })
    assert len(tab1) == 9 # remember this for later...

    tab2 = Table({
        "id2":  [101 , 102 , 103 , 104 ,             105 , 106 , 107 , 108  ],
        "ra2":  [90. , 90.1, 90.2, 90.3,             90.6, 90.7, 90.8, 90.9 ],
        "dec2": [0.  , 0.,   0.,   0.,               0.,   0.,   0.,   0.   ],
        "K_m":  [10.5, 11.5, 12.5, 13.5,             16.5, 17.5, 18.5, 19.5,],
    })

    tab1_path = paths.scratch_test_path / "cat_matcher_tab1.cat.fits"
    if tab1_path.exists():
        os.remove(tab1_path)
    assert not tab1_path.exists()
    tab1.write(tab1_path)

    tab2_path = paths.scratch_test_path / "cat_matcher_tab2.cat.fits"
    if tab2_path.exists():
        os.remove(tab2_path)
    assert not tab2_path.exists()
    tab2.write(tab2_path)

    matcher = catalog_builder.CatalogMatcher(tab1_path, ra="ra1", dec="dec1")
    assert matcher.catalog_path == tab1_path

    output_path = paths.scratch_test_path / "cat_matcher_output.cat.fits"
    if output_path.exists():
        os.remove(output_path)
    assert not output_path.exists()

    matcher.match_catalog(
        tab2_path, output_path, ra="ra2", dec="dec2", error=1.0, set_output_as_input=True
    )

    assert "join=all1" in matcher.stilts.cmd
    assert "find=best" in matcher.stilts.cmd

    output = Table.read(output_path)
    output.sort("id1")
    output["ra2"] = output["ra2"].filled(-99.)
    print(output["ra2"].filled(-99.))
    print([np.isfinite(x) for x in output["ra2"]])
    output["dec2"] = output["dec2"].filled(-99.)
    output["K_m"] = output["K_m"].filled(-99.)
    print(output[["ra2", "dec2", "K_m"]])
    assert set(output.columns) == set([
        "id1", "ra1", "dec1", "J_m", "id2", "ra2", "dec2", "K_m", "Separation"])
    assert len(output) == 9
    assert np.allclose(
        output["ra1"].data, [90., 90.1, 90.2, 90.3, 90.4,   90.5,   90.6, 90.7, 90.8,]
    )
    assert np.allclose(
        output["ra2"].data, [90., 90.1, 90.2, 90.3, np.nan, np.nan, 90.6, 90.7, 90.8,   ],
        equal_nan=True
    ) ###                                            ^^^^    ^^^^                    ^^^
    ### no 90.4, 90.5, so they are nan.

    assert np.allclose(output["dec1"], 0.)
    assert np.allclose(
        output["dec2"], [0.,0.,0.,0., np.nan, np.nan, 0., 0., 0.,],
        equal_nan=True
    ) ###                              ^^^^    ^^^^
    jk = output["J_m"] - output["K_m"]
    assert np.allclose(
        jk, [-0.5, -0.5, -0.5, -0.5, np.nan, np.nan, -0.5, -0.5, -0.5],
        equal_nan=True
    ) ### 










