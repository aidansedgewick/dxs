import os
import pytest

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from dxs import mosaic_builder
from dxs.utils.image import build_mosaic_wcs

from dxs import paths


magzpt_data = {f"magzpt_{ii}": np.random.normal(20.0, 0.1, 1000) for ii in [1,2,3,4]}
seeing_data = {f"seeing_{ii}": np.random.normal(0.8, 0.1, 1000) for ii in [1,2,3,4]}
example_stack_data = pd.DataFrame(
    {**magzpt_data, **seeing_data, "exptime": np.full(1000, 10.)}
)
print(example_stack_data)

def test__load_module():
    assert isinstance(mosaic_builder.survey_config, dict)
    assert isinstance(mosaic_builder.neighbors_config, dict)

    assert isinstance(mosaic_builder.header_data, pd.DataFrame)
    assert all(col in mosaic_builder.header_data.columns for col in 
        ["field", "tile", "band", "pointing", "deprec_mf", "magzpt_1", "seeing_1"]
    )
def test_get_stack_data():
    stack_bad_field = mosaic_builder.get_stack_data("NA", 1, "J")
    assert len(stack_bad_field) == 0
    stack_bad_tile = mosaic_builder.get_stack_data("SA", 13, "J")
    assert len(stack_bad_tile) == 0
    stack_bad_filter = mosaic_builder.get_stack_data("EN", 4, "H")
    assert len(stack_bad_tile) == 0
    SA04K_stack_data = mosaic_builder.get_stack_data(
        "SA", 4, "K", include_deprecated_stacks=True
    )
    assert len(SA04K_stack_data) == 122
    assert set(SA04K_stack_data["pointing"]) == set(["0_0", "0_1", "1_0", "1_1"])

def test__get_new_position():
    assert mosaic_builder.get_new_position((2,3), (1,1)) == (3,4)
    assert mosaic_builder.get_new_position((1.5,0.5), (1,0)) == (2.5,0.5)

def test__get_neighbor_tiles():
    assert set(mosaic_builder.get_neighbor_tiles("SA", 1)) == set([2,3,4,5,6,7,8,9])
    assert set(mosaic_builder.get_neighbor_tiles("SA", 3)) == set([1,2,4])
    assert set(mosaic_builder.get_neighbor_tiles("SA", 11)) == set([5,6,7,10,12])
  
    assert set(mosaic_builder.get_neighbor_tiles("LH", 3)) == set([1,2,4,5,6,7,8,9])
    assert set(mosaic_builder.get_neighbor_tiles("LH", 1)) == set([2,3,4,11])
    assert set(mosaic_builder.get_neighbor_tiles("LH", 7)) == set([3,6,8])

    assert set(mosaic_builder.get_neighbor_tiles("EN", 1)) == set([2,3,4,5,6,7,8])
    assert set(mosaic_builder.get_neighbor_tiles("EN", 3)) == set([1,2,4,10,12])
    assert set(mosaic_builder.get_neighbor_tiles("EN", 11)) == set([5,6])
    
    assert set(mosaic_builder.get_neighbor_tiles("XM", 3)) == set([1,2,4,5,6,7,8])
    assert set(mosaic_builder.get_neighbor_tiles("XM", 1)) == set([2,3,4])

def test__get_neighbor_stacks():

    ## test first where we know there is only 1 neighbour, SA07H - with neighbour SA06H.    
    SA07H_neighbors_deprec = mosaic_builder.get_neighbor_stacks(
        "SA", 7, "H", include_deprecated_stacks=True
    )
    assert len(SA07H_neighbors_deprec) == 36
    hdus_used = [
        (row["pointing"], tuple(row["ccds"])) for idx, row in SA07H_neighbors_deprec.iterrows()
    ]
    assert set(hdus_used) == set([("0_0", (1,2)), ("0_1", (1,2))])

    ## try ignoring deprec stacks.
    SA07H_neighbors_nd = mosaic_builder.get_neighbor_stacks(
        "SA", 7, "H", include_deprecated_stacks=False
    )
    # I count 8 deprec stacks in output of: grep SA,6,H,0_* ./configuration/dxs_header_data.csv
    assert len(SA07H_neighbors_nd) == 28 # 
    hdus_used = [
        (row["pointing"], tuple(row["ccds"])) for idx, row in SA07H_neighbors_nd.iterrows()
    ]
    assert set(hdus_used) == set([("0_0", (1,2)), ("0_1", (1,2))])

    print("\n\n\n\n")

    #... now try a more complicated one.
    LH07Kn_dep = mosaic_builder.get_neighbor_stacks(
        "LH", 7, "K", include_deprecated_stacks=True
    )

    # get len with:  grep <pattern> ./configuration/dxs_header_data.csv | wc -l
    assert sum(LH07Kn_dep["tile"] == 8) == 25 # output of pattern LH,8,K,1_*
    assert sum(LH07Kn_dep["tile"] == 3) == 12 # output of pattern LH,3,K,1_0
    assert sum(LH07Kn_dep["tile"] == 6) == 52 # sum of patterns LH,6,K,1_0 and LH,6,K,0_0
    
    hdus_used = [
        (row["tile"], row["pointing"], tuple(row["ccds"])) 
        for idx, row in LH07Kn_dep.iterrows()
    ]
    print(set(hdus_used))
    expected = [
        (6, "0_0", (1,4)), (6, "1_0", (1,4)), 
        (3, "1_0", (4,)), 
        (8, "1_1", (3,4)), (8, "1_0", (3,4)),
    ]

    assert set(hdus_used) == set(expected)

def test__calc_mosaic_geometry():
    size = (360, 360)
    pix_scale = 1. #arcsec
    centers = [
        (180. + 0.033, 0.033), (180. - 0.033, 0.033), 
        (180. + 0.033, -0.033), (180. - 0.033, -0.033)
    ]
    hdu_list = []
    for ii, center in enumerate(centers):
        data = np.random.uniform(0, 1, size)
        hdu_wcs = build_mosaic_wcs(center, size, pix_scale)
        header = hdu_wcs.to_header()
        hdu_path = paths.scratch_test_path / f"calc_geom_test_hdu_{ii}.fits"
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

    # check that it fails if geometry is MASSIVE.
    
    hdu1_center = SkyCoord(ra=130., dec=60., unit="deg")
    hdu2_center = SkyCoord(ra=140., dec=60., unit="deg")
    
    size1 = ( int(3600), int(3600 * np.cos(hdu1_center.dec)) ) 
    size2 = ( int(3600), int(3600 * np.cos(hdu2_center.dec)) )
    
    wcs1 = build_mosaic_wcs(hdu1_center, size1[::-1], 1.0)
    wcs2 = build_mosaic_wcs(hdu2_center, size2[::-1], 1.0)

    hdu1 = fits.PrimaryHDU(data=np.random.normal(5,1, size1), header=wcs1.to_header())
    hdu2 = fits.PrimaryHDU(data=np.random.normal(5,1, size2), header=wcs2.to_header())

    hdu1_path = paths.scratch_test_path / "mb_geom_fail1.fits"
    hdu1.writeto(hdu1_path, overwrite=True)
    hdu2_path = paths.scratch_test_path / "mb_geom_fail2.fits"
    hdu2.writeto(hdu2_path, overwrite=True)

    with pytest.raises(mosaic_builder.MosaicBuilderError):
        mosaic_builder.calculate_mosaic_geometry([hdu1_path, hdu2_path])
    os.remove(hdu1_path)
    os.remove(hdu2_path)
    assert not hdu1_path.exists()
    assert not hdu2_path.exists()

    
    

def test__add_keys():
    hdu_wcs = build_mosaic_wcs(
        (330., 0.), (1000, 1000), pixel_scale=1. / 3600.
    )
    empty_header = hdu_wcs.to_header()
    data = np.random.uniform(0, 1, (1000, 1000))
    print(empty_header)
    empty_fits = fits.PrimaryHDU(data=data, header=empty_header)
    fits_path = paths.scratch_test_path / "add_keys_test.fits"
    empty_fits.writeto(fits_path, overwrite=True)
    
    keys_to_add = {
        "test1": 100, 
        "test2": ("some_data", "this is a comment"),
        "test3": True,
        "test4": False,
        # "test5": (1000, 10000), # This will be val ==1000, comment==10000. Bad idea!
        "test6": None,
    }

    mb = mosaic_builder.MosaicBuilder(
        [], mosaic_path=fits_path, header_keys={"seeing": 1.2345}
    )
    assert isinstance(mb.header_keys, dict)
    assert "seeing" in mb.header_keys

    mb.add_extra_keys(extra_keys=keys_to_add)
    with fits.open(fits_path) as f:
        # test added.
        assert f[0].header["TEST1"] == 100
        assert f[0].header["TEST2"] == "some_data"
        assert f[0].header.comments["TEST2"] == "this is a comment"
        assert f[0].header["TEST3"]
        assert not f[0].header["TEST4"]
        # don't do test 5.
        assert f[0].header["TEST6"] is None

        # test default added.
        assert np.isclose(f[0].header["SEEING"], 1.2345)
        assert not f[0].header["DO_FLXSC"] 
        assert np.isclose(f[0].header["ABOFFSET"], 0.)
        assert f[0].header["FILL_VAL"] is None
        assert f[0].header["TRIMEDGE"] == 0
        assert not f[0].header["BGR_SUB"]
        assert f[0].header["BGRFILTR"] == "default"
        assert f[0].header["BGRSIGMA"] == "default"
        assert "BRANCH" in f[0].header
        assert "LOCALSHA" in f[0].header
        assert "PIPEVERS" in f[0].header

def test__mosaic_builder_init():

    output_path = paths.scratch_test_path / "mb_init/mb_builder_init.fits"
    if output_path.exists():
        os.remove(output_path)
    assert not output_path.exists()

    output_aux_dir = paths.scratch_test_path / "mb_init/aux"
    if output_aux_dir.exists():
        os.rmdir(output_aux_dir)
    assert not output_aux_dir.exists()

    output_dir = paths.scratch_test_path / "mb_init"
    if output_dir.exists():
        os.rmdir(output_dir)
    assert not output_dir.exists()

    input_hdu_list = [paths.scratch_test_path / "mb_init_hdu.fits"]

    mb = mosaic_builder.MosaicBuilder(input_hdu_list, output_path)
    assert isinstance(mb.swarp_config, dict)
    assert isinstance(mb.header_keys, dict)
    assert isinstance(mb.hdu_prep_kwargs, dict)
    
    assert len(mb.swarp_config) == 0
    assert len(mb.header_keys) == 0
    assert len(mb.hdu_prep_kwargs) == 0

    assert output_dir.exists()
    assert output_aux_dir.exists()
    assert str(mb.mosaic_path) == str(output_path)
    
    assert str(mb.swarp_config_file) == str(paths.config_path / "swarp/mosaic.swarp")
    
    assert len(mb.stack_list) == 1
    assert len(mb.ccds_list) == 1
    assert mb.ccds_list[0] is None

def test__prepare_all_hdus():

    ### first, write a couple of (multi-hdu) stacks.

    mb_prep_stacks_dir = paths.scratch_test_path / "mb_prep_stacks"
    mb_prep_stacks_dir.mkdir(parents=True, exist_ok=True)

    stack_path1 = paths.scratch_test_path / "mb_prep_stacks/mb_prep_stack1.fits"
    if stack_path1.exists():
        os.remove(stack_path1)
    assert not stack_path1.exists()
    stack_path2 = paths.scratch_test_path / "mb_prep_stacks/mb_prep_stack2.fits"
    if stack_path2.exists():
        os.remove(stack_path2)
    assert not stack_path2.exists()

    c_coords = [(214.5, 44.5), (215.5, 44.5), (214.5, 44.5), (215.5, 44.5),]
    centers = [SkyCoord(ra=ra, dec=dec, unit="deg") for (ra, dec) in c_coords]
    magzpts = [20., 21., 22., 23.,]
    size = (1000, 1000)

    hdul1 = fits.HDUList([fits.PrimaryHDU()])
    hdul1[0].header["EXP_TIME"] = 10.
    for center, magzpt in zip(centers, magzpts):
        hdu_wcs = build_mosaic_wcs(center, size, pixel_scale=1.0)
        hdu_data = np.random.uniform(5, 1, size)
        header = hdu_wcs.to_header()
        header["MAGZPT"] = magzpt
        hdu = fits.ImageHDU(data=hdu_data, header=header)
        hdul1.append(hdu)
    hdul1.writeto(stack_path1)

    hdul2 = fits.HDUList([fits.PrimaryHDU()])
    hdul2[0].header["EXP_TIME"] = 100.
    for center, magzpt in zip(centers, magzpts):
        hdu_wcs = build_mosaic_wcs(center, size, pixel_scale=1.0)
        hdu_data = np.random.uniform(5, 1, size)
        header = hdu_wcs.to_header()
        header["MAGZPT"] = magzpt
        hdu = fits.ImageHDU(data=hdu_data, header=header)
        hdul2.append(hdu)
    hdul2.writeto(stack_path2)

    stack_df = pd.DataFrame(
        {"filename": ["mb_prep_stack1", "mb_prep_stack2"], "ccds": [[1,2,3,4], [1,3]]}
    )
    assert len(stack_df) == 2

    output_aux_dir = paths.scratch_test_path / "mb_hdu_prep/aux"
    if output_aux_dir.exists():
        os.rmdir(output_aux_dir)
    assert not output_aux_dir.exists()

    output_dir = paths.scratch_test_path / "mb_hdu_prep"
    if output_dir.exists():
        os.rmdir(output_dir)
    assert not output_dir.exists()

    mosaic_path = paths.scratch_test_path / "mb_hdu_prep/mb_prepout.fits"
    s1_hdus = [paths.scratch_hdus_path / f"mb_prep_stack1_{ii:02d}.fits" for ii in [1,2,3,4]]
    s2_hdus = [paths.scratch_hdus_path / f"mb_prep_stack2_{ii:02d}.fits" for ii in [1,3]]
    expected_results = s1_hdus + s2_hdus
    for exp_path in expected_results:
        if exp_path.exists():
            os.remove(exp_path)
        assert not exp_path.exists()

    hdu_prep_kwargs = {"overwrite_magzpt": True, "AB_conversion": 2.0}
    mb = mosaic_builder.MosaicBuilder(
        stack_df, mosaic_path, stack_data_dir=mb_prep_stacks_dir, ext=".fits"
    )
    assert mb.ccds_list == [[1,2,3,4], [1,3]]
    assert len(mb.stack_list) == 2
    results = mb.prepare_all_hdus(**hdu_prep_kwargs)
    assert len(results) == 6
    assert len(set(results)) == 6


    assert set(results) == set(expected_results)

    with fits.open(expected_results[0]) as f0:
        assert np.isclose(f0[0].header["EXP_TIME"], 10.)
        assert np.isclose(f0[0].header["MAGZPT"], 20. + 2.5 + 2.)
    with fits.open(expected_results[1]) as f1:
        assert np.isclose(f1[0].header["EXP_TIME"], 10.)
        assert np.isclose(f1[0].header["MAGZPT"], 21. + 2.5 + 2.)
    with fits.open(expected_results[2]) as f2:
        assert np.isclose(f2[0].header["EXP_TIME"], 10.)
        assert np.isclose(f2[0].header["MAGZPT"], 22. + 2.5 + 2.)
    with fits.open(expected_results[3]) as f3:
        assert np.isclose(f3[0].header["EXP_TIME"], 10.)
        assert np.isclose(f3[0].header["MAGZPT"], 23. + 2.5 + 2.)
    with fits.open(expected_results[4]) as f4:
        assert np.isclose(f4[0].header["EXP_TIME"], 100.)
        assert np.isclose(f4[0].header["MAGZPT"], 20. + 5. + 2.)
    with fits.open(expected_results[5]) as f5:
        assert np.isclose(f5[0].header["EXP_TIME"], 100.)
        assert np.isclose(f5[0].header["MAGZPT"], 22. + 5. + 2.)
    
    ## only really need to do this because they're not in scratch_tests...
    for exp_path in expected_results:
        os.remove(exp_path)
        assert not exp_path.exists()


def test__initialise_astromatic():
    pass

def test__mosaic_builder_from_spec():
    builder_for_bad_spec = mosaic_builder.MosaicBuilder.from_dxs_spec("SA", 13, "K")
    assert builder_for_bad_spec is None

    swarp_config = {"center": (180., 1.234), "image_size": (12345, 21000)}
    SA04K_builder = mosaic_builder.MosaicBuilder.from_dxs_spec(
        "SA", 4, "K", 
        swarp_config=swarp_config, 
        include_neighbors=False, 
        include_deprecated_stacks=True
    )

    assert "seeing" in SA04K_builder.header_keys
    assert "magzpt" in SA04K_builder.header_keys

    assert len(SA04K_builder.stack_list) == 122
    assert set(tuple(x) for x in SA04K_builder.ccds_list) == set( ((1,2,3,4),) )
    SA04K_builder.initialise_astromatic()
    print(SA04K_builder.cmd_kwargs)
    assert "-NTHREADS 1" in SA04K_builder.cmd_kwargs["cmd"]
    assert "-CENTER 180.000000,1.234000" in SA04K_builder.cmd_kwargs["cmd"]
    assert "-IMAGE_SIZE 12345,21000" in SA04K_builder.cmd_kwargs["cmd"]
    assert SA04K_builder.cmd_kwargs["code"] == "SWarp"
    

def test__write_swarp_list():
    input_hdu_list = []
    for ii, ra in enumerate([89.9, 90., 90.1]):
        center = SkyCoord(ra=ra, dec=60., unit="deg")
        hdu_wcs = build_mosaic_wcs(center, (360, 360), 10.)
        hdu = fits.PrimaryHDU(
            data=np.random.normal(0, 1, (360, 360)), 
            header=hdu_wcs.to_header()
        )
        hdu_path = paths.scratch_test_path / f"mosaicbuilder_list_{ii}.fits"
        if hdu_path.exists():
            os.remove(hdu_path)
        assert not hdu_path.exists()
        input_hdu_list.append(hdu_path)
        hdu.writeto(hdu_path)

    output_path = paths.scratch_test_path / "mb_writelist/mb_writelisttest.fits"
    mb = mosaic_builder.MosaicBuilder(
        input_hdu_list, output_path
    )

    expected_path = paths.scratch_test_path / "mb_writelist/aux/mb_writelisttest_swarp_list.txt"
    assert str(mb.swarp_list_path) == str(expected_path)

    expected_tiny_data_path = paths.scratch_hdus_path / "mb_writelisttest_tiny_data.fits"
    if expected_tiny_data_path.exists():
        os.remove(expected_tiny_data_path)
    assert not expected_tiny_data_path.exists()

    mb.write_swarp_list(mb.stack_list)
    with open(expected_path) as f:
        hdu_list = []
        for line in f:
            hdu_list.append(line.strip())
        assert hdu_list[0] == str(expected_tiny_data_path)
        assert len(hdu_list) == 4

    for hdu_path in hdu_list:
        os.remove(hdu_path)
        assert not os.path.exists(hdu_path)
    os.remove(expected_path)
    os.rmdir(paths.scratch_test_path / "mb_writelist/aux")
    os.rmdir(paths.scratch_test_path / "mb_writelist")
    

#def test__pass_kwargs_to_hdu_preparer():
#    pass

def test__magzpt():
    magzpt = mosaic_builder.MosaicBuilder.calc_magzpt(example_stack_data)
    assert np.isclose(magzpt, 22.5, atol=0.01)
    magzpt_no_exptime = mosaic_builder.MosaicBuilder.calc_magzpt(
        example_stack_data, magzpt_inc_exptime=False
    )
    assert np.isclose(magzpt_no_exptime, 20.0, atol=0.01)
    

def test_seeing():
    seeing = mosaic_builder.MosaicBuilder.calc_seeing(example_stack_data)
    assert np.isclose(seeing, 0.8, atol=0.01)


