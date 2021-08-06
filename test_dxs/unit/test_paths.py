from dxs import paths

def test__create_paths_does_not_fail():
    paths.create_all_paths()

def test__paths_are_generated():
    assert paths.input_data_path.is_dir()
    assert paths.stack_data_path.is_dir()

    assert paths.scratch_data_path.is_dir()
    assert paths.scratch_hdus_path.is_dir()
    assert paths.scratch_swarp_path.is_dir()
    assert paths.scratch_sextractor_path.is_dir()
    assert paths.scratch_stilts_path.is_dir()

    assert paths.scratch_test_path.is_dir()

    assert paths.data_path.is_dir()
    assert paths.mosaics_path.is_dir()
    assert paths.catalogs_path.is_dir()
    assert paths.masks_path.is_dir()

    assert paths.runner_path.is_dir()

def test__configs_exist():
    assert paths.config_path.is_dir()
    survey_config_path = paths.config_path / "survey_config.yaml"
    assert survey_config_path.exists()

def test__get_mosaic_stem():
    mos_stem1 = paths.get_mosaic_stem("AA", 1, "Z", prefix="aa")
    assert mos_stem1 == "aaAA01Z"

    # try with str as tile.
    mos_stem2 = paths.get_mosaic_stem("BB", "a_string", "Y")
    assert mos_stem2 == "BBa_stringY"

def test__get_mosaic_path():
    bp_str = str(paths.base_path)   

    mos_path1 = paths.get_mosaic_path("CC", "fff", "X", extension=".cov.fits")    
    assert str(mos_path1) == bp_str + "/data/mosaics/CCfffX/CCfffX.cov.fits"

    mos_path2 = paths.get_mosaic_path("DD", 8, "W")
    assert str(mos_path2) == bp_str + "/data/mosaics/DD08W/DD08W.fits"


def test__get_catalog_stem():
    cat_stem1 = paths.get_catalog_stem(
        "EE", 4, "U", measurement_band="T", prefix="pr_", suffix="_suf"
    )
    assert cat_stem1 == "pr_EE04UTfp_suf"
    
    cat_stem_2 = paths.get_catalog_stem(
        "FF", "d3d", "S"
    )
    assert cat_stem_2 == "FFd3dS"

def test__get_catalog_path():
    bp_str = str(paths.base_path)

    cat_path1 = paths.get_catalog_path(
        "GG", 99, "R", measurement_band="Q", prefix="pr_", extension=".tiger.fits" 
    ) # tiger is a cat...?!
    assert str(cat_path1) == bp_str + "/data/catalogs/GG99/pr_GG99RQfp.tiger.fits"

    # normal extension.
    cat_path2 = paths.get_catalog_path(
        "HH", 5, "P"
    )
    assert str(cat_path2) == bp_str + "/data/catalogs/HH05/HH05P.cat.fits"






    
