

from dxs import BrightStarProcessor

from astropy.table import Table

def test__bsp_init():

    test_table = Table()
    bsp = BrightStarProcessor(test_table)

    assert len(bsp.mag_ranges) == len(bsp.circle_radii) + 1
    # check anything to see that it's initialised properly.

#def test__bsp_regions():

    
    
