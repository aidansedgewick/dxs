import logging
import re
import subprocess
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_fill_holes, binary_erosion

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from easyquery import Query
from regions import DS9Parser

from dxs.utils.image import build_mosaic_wcs, mask_regions_in_mosaic
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

def inElaisN1(ra, dec):
    x= (239.0 < ra) & (ra < 247.0) & (53.0 < dec) & (dec < 57.0)
    return x

def inSA22(ra, dec):
    return (332.0 < ra) & (ra < 336.0) & (-1.2 < dec) & (dec < 2.0)

def inXMM(ra, dec):
    return (34.0 < ra) & (ra < 38.0) & (-5.5 < dec) & (dec < -2.0)

def process_large_region_file(file_path, band):

    header = []
    SA_reg = []
    EN_reg = []
    XM_reg = []

    circ_re = re.compile("\(([0-9e\.]*),([-0-9e\.]*),([0-9\.]*)d\).*mag:([0-9\.]*)")
    box_re = re.compile("\(([0-9e\.]*),([-0-9e\.]*),([0-9\.]*)d,([0-9\.]*)d\).*mag:([0-9\.]*)")

    with open(file_path) as f:
        #i = 0
        for i, l in enumerate(f):
            #i = i+1
                
            if l.startswith("circle"):
                #print(l)
                res = circ_re.search(l)
                ra, dec, r, mag = (
                    float(res.group(1)), 
                    float(res.group(2)), 
                    float(res.group(3)), 
                    float(res.group(4)),
                )
            elif l.startswith("box"):
                #print(l)
                res = box_re.search(l)
                ra, dec, w, h, mag = (
                    float(res.group(1)), 
                    float(res.group(2)), 
                    float(res.group(3)), 
                    float(res.group(4)), 
                    float(res.group(5)),
                )
            else:
                header.append(l)
                continue

            if inElaisN1(ra, dec):
                EN_reg.append(l)
            if inSA22(ra, dec):
                SA_reg.append(l)
            if inXMM(ra, dec):
                XM_reg.append(l)
            if i % 10000 == 0:
                print(i)

    reg_dir = paths.input_data_path / "external/hsc/regions"
    reg_dir.mkdir(exist_ok=True, parents=True)

    with open(reg_dir / f"SA_{band}.reg", "w+") as f:
        f.writelines(header + SA_reg)
    with open(reg_dir / f"EN_{band}.reg", "w+") as f:
        f.writelines(header + EN_reg)
    with open(reg_dir / f"XM_{band}.reg", "w+") as f:
        f.writelines(header + XM_reg)

def faster_region_parse(file_path):
    regions = []
    header = []
    logger.info("parsing regions...")
    with open(file_path) as f:
        for i, l in enumerate(f):
            
            if not (l.startswith("circle") or l.startswith("box")):
                header.append(l)
            else:
                reg = DS9Parser("".join(header + [l])).shapes.to_regions()
                regions.append(reg[0])
    logger.info("done parsing")
    return regions
    

if __name__ == "__main__":
    default_fields = ["EN", "SA", "XM"]

    hsc_bands = ["g", "r", "i", "z", "y"]

    parser = ArgumentParser()
    parser.add_argument("--fields", nargs="+", choices=default_fields, default=default_fields, required=False)
    parser.add_argument("--force-download", action="store_true", default=False)
    parser.add_argument("--skip-mask", action="store_true", default=False)
    parser.add_argument("--mask-bands", default=hsc_bands, choices=hsc_bands, nargs="+")
    args = parser.parse_args()
    
    fields = args.fields #.split(",")
    base_dir = paths.input_data_path / "external/hsc"
    base_dir.mkdir(exist_ok=True, parents=True)
    catalog_dir = base_dir / "catalogs"
    catalog_dir.mkdir(exist_ok=True, parents=True)
    mask_dir = base_dir / "masks"
    mask_dir.mkdir(exist_ok=True, parents=True)

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
            print(f"No download URL for field {field}")
            continue
        url = base_url + catalog_url_code
        output_path = catalog_dir / f"{field}_catalog.fits"

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

        if args.skip_mask:
            continue

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
        for band in args.mask_bands:
            """band_cat = Query(
                "isprimary", 
                "~i_pixelflags_bad", 
                "~i_pixelflags_saturatedcenter", 
                "~i_pixelflags_edge", 
                "~i_pixelflags_isnull"
            ).filter(cat)"""

            band_cat = cat[ 
                (~cat[f"{band}_pixelflags_bad"])
                & (~cat[f"{band}_pixelflags_saturatedcenter"])
                & (~cat[f"{band}_pixelflags_edge"])
                & (cat["isprimary"])
                & (~cat[f"{band}_pixelflags_isnull"])
            ]
            
            logger.info("loaded")

        

            resolution = 20.
            d_res = (resolution / 3600.)

            min_ra = min(band_cat["ra"]) - 0.1
            max_ra = max(band_cat["ra"]) + 0.1
            min_dec = min(band_cat["dec"]) - 0.1
            max_dec = max(band_cat["dec"]) + 0.1

            print(min_ra, max_ra)
            print(min_dec, max_dec)

            center = [
                0.5 * (min_ra + max_ra),
                0.5 * (min_dec + max_dec),
            ]
            delta_coord = [max_ra - min_ra, max_dec - min_dec]
            shape_out = [
                int(delta_coord[0] / d_res * np.cos(np.radians(center[1])) ),
                int(delta_coord[1] / d_res),
            ]
            wcs_out = build_mosaic_wcs(
                center=center, size=shape_out, pixel_scale=resolution
            )

            xbins = np.linspace(min_ra, max_ra, shape_out[0])
            ybins = np.linspace(min_dec, max_dec, shape_out[1])

            data, _, _ = np.histogram2d(band_cat["dec"], band_cat["ra"], bins=[ybins, xbins])
            binary_data = data.copy()
            binary_data[ binary_data > 1 ] = 1

            binary_data = binary_erosion(binary_data, iterations=2)

            binary_data = binary_data.astype(float)
            binary_data = np.flip(binary_data, axis=1)
            
            #fig, ax = plt.subplots()
            #ax.imshow(data.astype(bool))
            fig, ax = plt.subplots()
            ax.imshow(binary_data)
            fig.suptitle(f"{field} {band}")
            


            mask_path = mask_dir / f"{field}_{band}_mask.fits"
            mask_hdu = fits.PrimaryHDU(data=binary_data, header=wcs_out.to_header())
            mask_hdu.writeto(mask_path, overwrite=True)

            reg_path = paths.input_data_path / f"external/hsc/regions/{field}_{band}.reg"
            if not reg_path.exists():
                large_region_file = (
                    paths.input_data_path / f"external/hsc/new_S18Amask_{band}.reg"
                )
                process_large_region_file(large_region_file, band)
            region_list = faster_region_parse(reg_path)

            mask_regions_in_mosaic(mask_path, region_list)

            print(f"written mask to {mask_path.relative_to(paths.base_path)}")
    plt.show()
        



        
