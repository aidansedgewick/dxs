import logging
import shutil
import subprocess
import yaml
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table

from easyquery import Query

from dxs.utils.table import explode_columns_in_fits
from dxs.utils.phot import vega_to_ab

from stilts_wrapper import Stilts

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--force-download", default=False, action="store_true")
    parser.add_argument("--skip-process", default=False, action="store_true")
    parser.add_argument("--no-plot", default=False, action="store_true")
    args = parser.parse_args()

    base_dir = paths.input_data_path / "external/uds"
    base_dir.mkdir(exist_ok=True, parents=True)

    catalog_dir = base_dir / "catalogs"
    catalog_dir.mkdir(exist_ok=True, parents=True)

    download_cat_path = base_dir / "uds_jhk_raw.cat.fits"
    cat_path = catalog_dir / "uds_jhk.cat.fits"

    downloaded = False
    if not download_cat_path.exists() or args.force_download:
        cat_url = survey_config["uds"]["catalog_url"]
        status = download_data(cat_url, download_cat_path)
        if status > 1:
            raise IOError()
        downloaded = True   

    if downloaded or (not args.skip_process):
        shutil.copy2(download_cat_path, cat_path)

        pix_apertures = np.array(survey_config["uds"]["phot_apertures"])
        apertures = np.round(pix_apertures * 0.4 / 3., decimals=1)

        ap_suffixes = [str(int(ap*10)) for ap in apertures]

        print("APERTURES (0.1\")", ap_suffixes)

        columns_to_explode = [f"MAG{err}_APER_{b}" for b in ["K", "H", "J"] for err in ["", "ERR"] ]

        print(columns_to_explode)
        explode_columns_in_fits(cat_path, columns_to_explode, suffixes=ap_suffixes)
       
        catalog = Table.read(cat_path)
        for b in ["K", "H", "J"]:
            mag_cols = [f"MAG_AUTO_{b}"] + [
                f"MAG_APER_{b}_{ap}" for ap in ap_suffixes
            ]
            for col in mag_cols:
                print(col)
                catalog[col] = vega_to_ab(catalog[col], band=b)

            catalog.rename_column(f"MAG_AUTO_{b}", f"{b}_mag_auto")
            catalog.rename_column(f"MAGERR_AUTO_{b}", f"{b}_magerr_auto")
            for ap in ap_suffixes:
                catalog.rename_column(f"MAG_APER_{b}_{ap}", f"{b}_mag_aper_{ap}")
                catalog.rename_column(f"MAGERR_APER_{b}_{ap}", f"{b}_magerr_aper_{ap}")
        catalog.rename_column("ALPHA_J2000", "ra")
        catalog.rename_column("DELTA_J2000", "dec")

        catalog.write(cat_path, overwrite=True)
    else:
        catalog = Table.read(cat_path)

    if not args.no_plot:
        pl_cat = catalog[ np.arange(0, len(catalog)-1, 25) ]
        #fig,ax = plt.subplots()
        #ax.scatter(
        #    pl_cat["MAG_AUTO_K"], pl_cat["MAG_APER_J_20"] - pl_cat["MAG_APER_K_20"], 
        #    color="k", s=1
        #)
        
        drgs = Query(
            "J_mag_aper_20 - K_mag_aper_20 > 1.34", 
            "K_mag_aper_20 < 50.", "J_mag_aper_20 < 50."
        ).filter(catalog)
        fig2, ax2 = plt.subplots()
        ax2.semilogy()
        bins = np.arange(15.75, 26.25, 0.5)
        mids = 0.5 * (bins[:-1] + bins[1:])
        print(bins, mids)

        drg_hist, _ = np.histogram(drgs["K_mag_auto"], bins=bins)

        survey_area = 0.88
        drg_hist = drg_hist / 0.88


        ax2.plot(mids, drg_hist)


        lit_df = pd.read_csv(paths.input_data_path / "plotting/kajisawa06_number_counts.csv")
        ax2.scatter(lit_df["Kmag"].values, lit_df["drg"].values, marker="x")

        kim_df = pd.read_csv(paths.input_data_path / "plotting/kim11_number_counts.csv")
        ax2.scatter(kim_df["Kmag"].values, kim_df["drg"].values)
        #ax2.set_ylim()



    print("\n\n\n======== match hsc\n")
    hsc_cat_path = paths.input_data_path / "external/hsc/catalogs/XM_hsc.cat.fits"
    output_path = catalog_dir / "uds_jhk_hsc.cat.fits"
    
    st = Stilts.tskymatch2(
        in1=cat_path, in2=hsc_cat_path,
        out=output_path,
        ra1="ra", dec1="dec",
        ra2="ra_hsc", dec2="dec_hsc",
        error=0.8, join="all1", find="best",
        all_formats="fits"
    )
    st.run()
    
    galex_output_path = catalog_dir / "uds_jhk_galex_hsc.cat.fits"
    galex_path = paths.input_data_path / "external/galex/catalogs/XM_combo_catalog.cat.fits"

    print("\n\n\n======== match galex\n")

    st_galex = Stilts.tskymatch2(
        in1=output_path, in2=galex_path,
        out=galex_output_path,
        ra1="ra", dec1="dec",
        ra2="galex_ra", dec2="galex_dec",
        error=1.5, join="all1", find="best",
        all_formats="fits"
    )
    st_galex.run()

    spuds_output_path = catalog_dir / "uds_jhk_galex_hsc_spuds.cat.fits"
    spuds_path = paths.input_data_path / "external/spuds/spuds.cat.fits"

    print("\n\n\n======== match spuds\n")

    st_spuds = Stilts.tskymatch2(
        in1=galex_output_path, in2=spuds_path,
        out=spuds_output_path, 
        ra1="ra", dec1="dec",
        ra2="ra_spuds", dec2="dec_spuds",
        error=2.0, join="all1", find="best",
        all_formats="fits"
    )
    st_spuds.run()

    if not args.no_plot:
        plt.show()
    


