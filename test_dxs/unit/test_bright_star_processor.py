import os
import pytest
import yaml

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from regions import CircleSkyRegion, RectangleSkyRegion

from easyquery import Query

from dxs import BrightStarProcessor
from dxs import paths

def test__bsp_init():

    test_table = Table()
    bsp = BrightStarProcessor(test_table)

    assert len(bsp.mag_ranges) == len(bsp.circle_radii) + 1
    # check anything to see that it's initialised properly.

def test__bad_inputs_fail():
    fail_tab = Table({"ra": [1.,2.,3.], "dec": [1., 2., 3.]})
    fail_path = paths.scratch_test_path / "fail_cat.csv"
    fail_tab.write(fail_path, overwrite=True)
    with pytest.raises(ValueError):
        bsp_fail1 = BrightStarProcessor(fail_path)
    os.remove(fail_path)

    bad_config = {
        "diffraction_spike_angles": [0, 90],
        "region_coefficients": {
            "mag_ranges": [0, 2, 3, 4, 5],
            "box_widths": [20, 10],
            "box_heights": [1000, 500],
            "circle_radii": [500, 100],
        }
    }

    bad_config_path = paths.scratch_test_path / "bad_bsp_config.yaml"
    with open(bad_config_path, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(ValueError):
        bsp_fail2 = BrightStarProcessor(Table(), config_path=bad_config_path)
        

def test__bsp_from_file():
    catalog = Table({
        "id": [1,2,3,4,5,6],
        "J": [5., 6., 7., 11., 12., 17.],
        "K": [3.5, 5.1, 5.5, 9.5, 11.5, 16.5],
    })
    catalog_path = paths.scratch_test_path / "bsp_from_file_cat.csv"
    catalog.write(catalog_path, overwrite=True)

    queries = (Query("J-K<1.0") | Query("K<8.0")) & (Query("K<15.0"))
    bsp = BrightStarProcessor.from_file(catalog_path, query=queries)

    assert len(bsp.star_table) == 4
    assert np.allclose(bsp.star_table["id"], [1,2,3,5])
    assert np.allclose(bsp.star_table["J"], [5., 6., 7., 12.])
    assert np.allclose(bsp.star_table["K"], [3.5, 5.1, 5.5, 11.5])

def test__get_regions_for_star():
    bsp = BrightStarProcessor(Table())
    regions = bsp.get_regions_for_star(
        SkyCoord(ra=45., dec=45., unit="deg"), 1000 * u.arcsec, 50 * u.arcsec, 100 * u.arcsec
    )
    assert len(regions)
    assert all([isinstance(x, RectangleSkyRegion) for x in regions[:-1]])
    assert isinstance(regions[-1], CircleSkyRegion)

    

    
    
