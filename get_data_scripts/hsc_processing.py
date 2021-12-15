import logging
import re
import subprocess
import tqdm
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import morphology 

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from easyquery import Query
from regions import DS9Parser

from dxs.utils.image import build_mosaic_wcs, mask_regions_in_mosaic
from dxs.utils.misc import calc_range, print_header
from dxs.utils.table import fix_column_names

from dxs import paths

logger = logging.getLogger("hsc_proc")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

def download_data(url, output_path):
    try:
        result = subprocess.run(["wget", url, "-O", output_path, "--no-check-certificate"])
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
        i = 0
        for i,l in enumerate(f):
            i = i+1
                
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

def faster_region_parse(file_path, mag_lim=12.):
    regions = []
    header = []
    logger.info("parsing regions...")

    mag_re = re.compile(".*mag:([0-9\.]*)")
    with open(file_path) as f:
        for l in tqdm.tqdm(f):
            
            if not (l.startswith("circle") or l.startswith("box")):
                header.append(l)
            else:
                #if l.startswith("box"):
                #    continue
                res = mag_re.search(l)
                mag = float(res.group(1))
                if mag > mag_lim:
                    continue
                reg = DS9Parser("".join(header + [l])).shapes.to_regions()
                regions.append(reg[0])
    logger.info(f"parsed {len(regions)}")
    return regions
    

def create_field_mask(
    field, bands, random_catalog_path, output_path, resolution=10., skip_stars=False,
):

    logger.info(f"read {random_catalog_path}")
    #catalog = Table.read(catalog_path)
    randoms = Table.read(random_catalog_path)

    randoms = Query("isprimary").filter(randoms)
    logger.info("done query")

    min_ra  = min(randoms["ra"])  - 0.1
    max_ra  = max(randoms["ra"])  + 0.1
    min_dec = min(randoms["dec"]) - 0.1
    max_dec = max(randoms["dec"]) + 0.1


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


    arr = np.column_stack((randoms["ra"], randoms["dec"]))
    logger.info(f"start wcs transform, {len(arr)} obj")
    rand_pix = wcs_out.wcs_world2pix(arr, 0).astype(int) # NOT array_index - this gives (col, row)

    for pix in tqdm.tqdm(rand_pix):
        mask_array[tuple(pix[::-1])] += 1
    mask_array[ mask_array > 0 ] = 1


    ##mask_array = binary_dilation(mask_array, iterations=10).astype(int)
    #mask_array = binary_erosion(mask_array, iterations=4).astype(int)

    for band in bands:
        print(f"look at {band}")  
        band_array = mask_array.copy()

        rand_primary_q = Query("isprimary")
        rand_bad_q = Query(f"{band}_pixelflags_bad")
        rand_satcenter_q = Query(f"{band}_pixelflags_saturatedcenter")
        rand_edge_q = Query(f"{band}_pixelflags_edge")
        rand_null_q = Query(f"{band}_pixelflags_isnull")

        #good_randoms = (
        #    rand_primary_q & ~rand_bad_q & ~rand_satcenter_q & ~rand_edge_q & ~rand_null_q
        #).filter(randoms)
            
        bad_randoms = (
            rand_bad_q | rand_satcenter_q | rand_edge_q | rand_null_q
        ).filter(randoms)

        arr = np.column_stack([bad_randoms["ra"], bad_randoms["dec"]])
        bad_pix = wcs_out.wcs_world2pix(arr, 0).astype(int)
        for pix in tqdm.tqdm(bad_pix):
            band_array[tuple(pix[::-1])] = 0.

        band_array = morphology.binary_fill_holes(band_array).astype(int)
        band_array = morphology.binary_erosion(band_array).astype(int)



        # Keep and unmodified version, too...
        exp_mask_output_path = mask_dir / f"{field}_{band}_exp_mask.fits"
        hdu = fits.PrimaryHDU(data=band_array, header=header)
        hdu.writeto(exp_mask_output_path, overwrite=True)
        
        if skip_stars:
            continue

        mask_output_path = mask_dir / f"{field}_{band}_mask.fits"
        hdu = fits.PrimaryHDU(data=band_array, header=header)
        hdu.writeto(mask_output_path, overwrite=True)

        reg_path = paths.input_data_path / f"external/hsc/regions/{field}_{band}.reg"
        if not reg_path.exists():
            large_region_file = (
                paths.input_data_path / f"external/hsc/new_S18Amask_{band}.reg"
            )
            process_large_region_file(large_region_file, band)
        region_list = faster_region_parse(reg_path)

        print(region_list[0].meta)

        mask_regions_in_mosaic(mask_output_path, region_list)

        #aux_reg_path = paths.input_data_path / f"external/hsc/regions/{field}_{band}_aux.reg"
        #if aux_reg_path.exists():
            #aux_regions = DS9Parser(str(aux_reg_path))
            
            #region_list = faster_region_parse(reg_path)

            #print(region_list[0].meta)
    
            #mask_regions_in_mosaic(mask_output_path, region_list)




    #plot_array = mask_array.copy()
    #plot_array[ plot_array == 0 ] = np.nan
    #fig, ax = plt.subplots()
    #ax.imshow(plot_array)

#    return fig

if __name__ == "__main__":
    default_fields = ["EN", "SA", "XM"]

    hsc_bands = ["g", "r", "i", "z", "y"]

    parser = ArgumentParser()
    parser.add_argument("--fields", nargs="+", choices=default_fields, default=default_fields, required=False)
    parser.add_argument("--force-download", action="store_true", default=False)
    parser.add_argument("--skip-mask", action="store_true", default=False)
    parser.add_argument("--mask-bands", default=hsc_bands, choices=hsc_bands, nargs="+")
    parser.add_argument("--skip-stars", default=False, action="store_true")
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
        print_header(f"look at {field}")

        catalog_url_code = survey_config["hsc"]["catalog_urls"].get(field, None)
        if catalog_url_code is None:
            print(f"No download URL for field {field}")
            continue
        url = base_url + catalog_url_code
        catalog_output_path = catalog_dir / f"{field}_hsc.cat.fits"

        if not catalog_output_path.exists() or args.force_download:
            logger.info(f"downloading {catalog_output_path}")
            status = download_data(url, catalog_output_path)
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
                    catalog_output_path, column_lookup={"ra": "ra_hsc", "dec": "dec_hsc"}
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

        fig = create_field_mask(
            field, args.mask_bands, random_output_path, mask_dir, skip_stars=args.skip_stars
        )
  
    plt.show()







        
