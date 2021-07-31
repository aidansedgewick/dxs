import getpass
import logging
import tqdm
import sys
import yaml
from argparse import ArgumentParser
from traceback import print_exc

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import regions

from dxs.utils.image import build_mosaic_wcs

from dxs import paths

logger = logging.getLogger("euc_proc")

try:
    import eagleSqlTools
except Exception:
    print_exc()
    print("-----------\n\n")
    print("\033[33mneed eagleSqlTools\033[0m - not in standard requirements.txt.")
    print("use:\n    python3 -m pip install eagleSqlTools")
    sys.exit()

if __name__ == "__main__":

    survey_config_path = paths.config_path / "survey_config.yaml"
    with open(survey_config_path, "r") as f:
        survey_config = yaml.load(f, Loader=yaml.FullLoader)

    parser = ArgumentParser()
    parser.add_argument("--force-download", action="store_true", default=False)
    args = parser.parse_args()

    euclid_dir = paths.input_data_path / "external/euclidsim"
    euclid_dir.mkdir(exist_ok=True, parents=True)
    euclid_cat_path = euclid_dir / "euclid_k.cat.fits"
    if euclid_cat_path.exists() is False:
        pw_path = paths.config_path / "virgodb.yaml"
        if pw_path.exists():        
            with open(pw_path, "r") as f:
                virgodb_config = yaml.load(f, Loader=yaml.FullLoader)
            usr = virgodb_config["username"]
            pwd = virgodb_config["password"]
        else:
            usr = input("virgo-db Username: ")
            pwd = getpass.getpass()

        ###====================SQL CONNECTION==================###
        con = eagleSqlTools.connect(usr, password=pwd)
        ###====================================================###
        del pwd

        opt_mag_cols = ",\n    ".join(
            f"DES_{band}_obs_app as {band.lower()}_mag_kron" for band in "grizY" 
        )
        nir_mag_cols = ",\n    ".join(
            f"mag_{band}_obs_app as {band}_mag_kron" for band in "JHK" 
        )
        mir_mag_cols = ",\n    ".join(
            f"IRAC_{b}_obs_app as I{ii}_mag_kron" for ii, b in enumerate([36, 45, 58, 80],1)
        )
        other_cols = ",\n    ".join(
            ["ra", "dec", "sfr", "redshift_obs as z_obs", "stellarmass"]
        )

        cols = ",\n    ".join(
            [other_cols, opt_mag_cols, nir_mag_cols, mir_mag_cols]
        )

        #cols = f"ra,\n    dec,\n    {opt_mag_cols},\n    {nir_mag_cols},\n    {mir_mag_cols}"
        query = (
            f"select \n    {cols}\n"
            f"from EUCLID_v1..LC_DEEP_Gonzalez2014a\n"
            f"where mag_K_obs_app < 24.5"
        )
        print(query)
        data = eagleSqlTools.execute_query(con, query)
        print(f"selected {len(data)} rows")

        col_names = [col.split()[-1] for col in cols.split(",")]
        euclid = Table(data, names=col_names)
        euclid.meta["SQL_QUER"] = query
        euclid.write(euclid_cat_path, overwrite=True)
        euclid_query_path = euclid_dir / "euclid_query.sql"
        with open(euclid_query_path, "w+") as f:
            f.writelines(query)
    else:
        euclid = Table.read(euclid_cat_path)

    
    min_ra  = min(euclid["ra"])  - 0.1
    max_ra  = max(euclid["ra"])  + 0.1
    min_dec = min(euclid["dec"]) - 0.1
    max_dec = max(euclid["dec"]) + 0.1

    resolution = 60.

    center = [
        0.5 * (min_ra + max_ra),
        0.5 * (min_dec + max_dec),
    ]
    delta_coord = [max_ra - min_ra, max_dec - min_dec]
    d_res = resolution / 3600.

    cdec_factor = np.cos( np.radians( center[1] ))
    shape_out = [
        int(1.05 * delta_coord[0] / d_res * cdec_factor ), 
        int(1.05 * delta_coord[1] / d_res),
    ]
    wcs_out = build_mosaic_wcs(
        center=center, size=shape_out, pixel_scale=resolution
    )
    header = wcs_out.to_header()

    mask_array = np.zeros(shape_out[::-1])

    arr = np.column_stack((euclid["ra"], euclid["dec"]))

    logger.info(f"start wcs transform, {len(arr)} obj")
    rand_pix = wcs_out.wcs_world2pix(arr, 0).astype(int) # NOT array_index - this gives (col, row)

    for pix in tqdm.tqdm(rand_pix):
        mask_array[tuple(pix[::-1])] += 1    

    fig, ax = plt.subplots()
    plot_array = mask_array.copy()
    plot_array[ plot_array == 0 ] = np.nan
    ax.imshow(plot_array)
    plt.show()


    """

    fig, ax = plt.subplots()
    #ax.scatter(t["ra"], t["dec"], s=1, color="k")

    center = SkyCoord(ra=45., dec=45., unit="deg")
    size = (10000, 10000)

    header = build_mosaic_header(
        (center.ra.value, center.dec.value), size, pixel_scale=2.0
    )
    mosaic_wcs = WCS(header)
    x = np.vstack([t["ra"], t["dec"]]).T
    pix_coords = mosaic_wcs.wcs_world2pix(x, 0)
    ax.scatter(pix_coords[:,0], pix_coords[:,1], s=1, color="k")

    r = np.sqrt(20.5 / np.pi)
    sky_circle = regions.CircleSkyRegion(center, radius=r * u.deg)
    pix_circle = sky_circle.to_pixel(mosaic_wcs)
    pix_circle.plot(ax=ax, color="r", lw=5)

    data = np.zeros(size)
    mask = pix_circle.to_mask()
    slicer = mask.bbox.slices
    
    data[slicer] = mask.data

    mask_path = euclid_dir / "euclid_mask.fits"

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(mask_path, overwrite=True)

    plt.show()
    """

    
    
    
    


