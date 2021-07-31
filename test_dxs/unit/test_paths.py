from dxs import paths

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

    
