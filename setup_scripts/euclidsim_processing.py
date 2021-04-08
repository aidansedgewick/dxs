import getpass
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

from dxs.mosaic_builder import build_mosaic_header

from dxs import paths

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
        cols = f"ra,\n    dec,\n    {opt_mag_cols},\n    {nir_mag_cols}"
        query = (
            f"select \n    {cols}\n"
            f"from EUCLID_v1..LC_DEEP_Gonzalez2014a\n"
            f"where mag_K_obs_app < 24.0"
        )
        print(query)
        data = eagleSqlTools.execute_query(con, query)
        print(f"selected {len(data)} rows")

        col_names = [col.split()[-1] for col in cols.split(",")]
        t = Table(data, names=col_names)
        t.write(euclid_cat_path, overwrite=True)
        euclid_query_path = euclid_dir / "euclid_query.sql"
        with open(euclid_query_path, "w+") as f:
            f.writelines(query)
    else:
        t = Table.read(euclid_cat_path)

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

    
    
    
    


