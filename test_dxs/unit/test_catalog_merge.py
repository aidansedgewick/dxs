import numpy as np

from astropy.table import Table

from easyquery import Query

from dxs import catalog_merge

from dxs import paths

def test__modify_id_value():
    id_vals = np.array([0,1,2,3,4,5,6,7])
    other_data = np.array([1.0, 2.1, 33.2, 1e5, -10.0, 77.7, 0.0, 4.])
    t = Table({"id": id_vals, "other_data": other_data})
    table_path = paths.temp_test_path / "test_modify_id.cat.fits"
    t.write(table_path, overwrite=True)

    catalog_merge._modify_id_value(table_path, id_modifier=1000000)
    modified = Table.read(table_path)
    assert all(modified["id"] == id_vals + 1000000)
    assert all(modified["other_data"] == other_data)

def test__merge_catalogs():

    # make an array of points, split it with overlap, and see what merges.
    ra_points = np.linspace(-1.0, 1.0, 100) # spacing here should be larger than match error.
    dec_points = np.linspace(-1.0, 1.0, 100) # spacing here should be larger than match error.
    
    ra_vals, dec_vals = np.meshgrid(ra_points, dec_points)
    ra_vals = ra_vals.flatten()
    dec_vals = dec_vals.flatten()

    id_vals = np.arange(len(ra_vals))
    snr_vals = np.full(len(ra_vals), 1.) + 1e-2 * np.random.uniform(0, 1, len(ra_vals))

    main_catalog = Table({"id": id_vals, "ra": ra_vals, "dec": dec_vals, "snr": snr_vals})

    # split catalog: 

    catalog1 = Query("ra > -0.1", "dec > -0.1").filter(main_catalog)
    catalog2 = Query("ra > -0.1", "dec < 0.1").filter(main_catalog)
    catalog3 = Query("ra < 0.1", "dec > -0.1").filter(main_catalog)
    catalog4 = Query("ra < 0.1", "dec < 0.1").filter(main_catalog)

    for cat, tile in zip([catalog1, catalog2, catalog3, catalog4], [1, 2, 3, 4]):
        cat["tile"] = np.full(len(cat), tile)
        cat[ cat["id"] % 4 == tile-1 ]["snr"] = 10.
        catalog_path = paths.temp_test_path / f"catalog_{tile}.cat.fits"
        cat.write(catalog_path, overwrite=True)

    assert all(catalog3["tile"] == 3)
    catalog_list = [paths.temp_test_path / f"catalog_{tile}_merge.cat.fits" for tile in [1, 2, 3, 4]]

    output_path = paths.temp_test_path / "merged_catalog.cat.fits"
    #catalog_merge.merge_catalogs(catalog_list, output_path, "id", "ra", "dec", "snr")

    # TODO URGENT FIX



