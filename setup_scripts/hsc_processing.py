import logging
import subprocess
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_fill_holes

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from easyquery import Query
from regions import DS9Parser

from dxs.utils.misc import calc_range
from dxs.utils.table import fix_column_names

from dxs import paths

logger = logging.getLogger("hsc_proc")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

def download_data(url, output_path):
    try:
        result = subprocess.run(["wget", url, "-O", output_path])
        return 0
    except Exception as e:
        print(f"Download failed; result:")
        print(e)
        return 1

def make_field_mask(region_list_path):

    logger.info("reading...")
    with open(region_list_path) as f:
        l = f.readlines()
    logger.info("read.")
    prefix = "".join(l[:6])
    for ii in range(7, len(l)):
        s = prefix + l[ii]
        region = DS9Parser(s).shapes#.to_regions()[0]
        #print(region)
        #mask = region.to_pixel().to_mask()
        if ii % 1000 == 0:
            print(ii, len(l))
            

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fields", default="SA,LH,EN,XM", required=False)
    parser.add_argument("--force-download", action="store_true", default=False)
    parser.add_argument("--skip-mask", action="store_true", default=False)
    args = parser.parse_args()
    
    fields = args.fields.split(",")
    base_dir = paths.input_data_path / "external/hsc"
    base_dir.mkdir(exist_ok=True, parents=True)
    catalog_dir = base_dir / "catalogs"
    catalog_dir.mkdir(exist_ok=True, parents=True)

    base_url = survey_config["hsc"]["url_base"]

    regions_path = base_dir / "new_S18Amask_i.reg"
    if not regions_path.exists():
        tar_path = base_dir / "S18A_starmask.tar.xz"
        if not tar_path.exists():
            tar_url = survey_config["hsc"]["mask_tar_url"]
            subprocess.run(["wget", tar_url, "-O", tar_path])
        subprocess.run(["tar", "-vxf", tar_path, "-C", base_dir])

    for field in fields:
        catalog_url_code = survey_config["hsc"]["catalog_urls"].get(field, None)
        if catalog_url_code is None:
            continue
        url = base_url + catalog_url_code
        output_path = base_dir / f"{field}_catalog.fits"

        if not output_path.exists() or args.force_download:
            status = download_data(url, output_path)
            if status == 1:
                query_path = base_dir / f"{field}_hsc_query.sql"
                try:
                    print_path = query_path.relative_to(Path.cwd())
                except:
                    print_path = query_path
                print("Try logging onto https://hsc-release.mtk.nao.ac.jp/doc/")
                print("and using the query in:")
                print(f"{print_path}")
            else:
                fix_column_names(
                    output_path, column_lookup={"ra": "ra_hsc", "dec": "dec_hsc"}
                )

        random_url_code = survey_config["hsc"]["random_urls"].get(field, None)
        if random_url_code is None:
            continue
        url = base_url + random_url_code
        random_output_path = base_dir / f"{field}_randoms.fits"
        
        if not random_output_path.exists() or args.force_download:
            status = download_data(url, random_output_path)
            if status == 1:
                query_path = base_dir / f"{field}_hsc_query.sql"
                try:
                    print_path = query_path.relative_to(Path.cwd())
                except:
                    print_path = query_path
                print("Try logging onto https://hsc-release.mtk.nao.ac.jp/doc/")
                print("and using the query in:")
                print(f"{print_path}")


        logger.info("reading catalog")
        cat = Table.read(random_output_path)
        cat = Query(
            "isprimary", 
            "~i_pixelflags_bad", 
            "~i_pixelflags_saturatedcenter", 
            "~i_pixelflags_edge", 
            "~i_pixelflags_isnull"
        ).filter(cat)
        logger.info("read.")        

        pixscale = 20. / 3600.

        min_ra, max_ra = calc_range(cat["ra"])
        min_ra -= 0.1
        max_ra += 0.1

        min_dec, max_dec = calc_range(cat["dec"])
        min_dec -= 0.1
        max_dec += 0.1

        nx_pix = int(np.ceil( (max_ra - min_ra)/pixscale))
        ny_pix = int(np.ceil( (max_dec - min_dec)/pixscale))

        xbins = np.linspace(min_ra, max_ra, nx_pix)
        ybins = np.linspace(min_dec, max_dec, ny_pix)

        print(nx_pix, ny_pix)
        xy = (np.array([1,nx_pix]), np.array([1 , ny_pix]))
        sky_coord = SkyCoord(ra=[min_ra, max_ra], dec=[min_dec, max_dec], unit="degree")
        print(sky_coord)
        wcs = fit_wcs_from_points(xy, sky_coord, projection="TAN", sip_degree=3)
        print(wcs)

        horizontal = np.array([[1,0,1]]) # vertical 'structure' type
        vertical = horizontal.T                # horizontal 'structure'.
        foursquare = np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])


        data, _, _ = np.histogram2d(cat["dec"], cat["ra"], bins=[ybins, xbins])
        binary_data = data.copy()
        binary_data[ binary_data > 1 ] = 1

        binary_data = binary_data.astype(float)
        """
        fig, ax = plt.subplots()
        ax.imshow(data.astype(bool).T)
        fig, ax = plt.subplots()
        ax.imshow(binary_data.T)
        plt.show()"""

        mask_hdu = fits.PrimaryHDU(data=binary_data, header=wcs.to_header())
        mask_hdu.writeto(base_dir / f"{field}_mask.fits", overwrite=True)
        
        



        
