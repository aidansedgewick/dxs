import subprocess
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from astropy.table import Table

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

swire_config = survey_config["swire"]
url_base = swire_config["url_base"]
field_lookup = swire_config["field_lookup"]
swire_columns = ",".join(col for col in swire_config["select_columns"])


def download_catalog(field, output_path):
    swire_field = field_lookup.get(field, None)
    if swire_field is None and field == "test":
        print("WARN: using small test area inside elaisn1")
        swire_field = "elaisn1"

    ra_limits = survey_config["field_limits"].get(field, {}).get("ra", [242.0, 242.5])
    dec_limits = survey_config["field_limits"].get(field, {}).get("dec", [54.8, 55.2])

    constraints = (
        f"({ra_limits[0]} < ra) and (ra < {ra_limits[1]}) and "
        f"({dec_limits[0]} < dec) and (dec < {dec_limits[1]})"
    )
    query = (
        f"{url_base}"
        f"catalog={swire_field}_cat_s05&"
        f"spatial=None&"
        f"outfmt=1&" # 
        f"constraints={constraints}&" # ampersand included if constraints is not empty string.
        f"selcols={swire_columns}&"
    )

    print(f"DOWNLOAD: {output_path.name}")

    query_cmd = ["wget", f"{query}", "-O", output_path]
    status = subprocess.run(query_cmd)

    print(status)


def process_catalog(input_path, output_path):
    
    cat = Table.read(input_path, format="ascii")
    ap_flux_columns = ["flux_ap3_36", "flux_ap3_45", "flux_ap3_58", "flux_ap3_80"]
    kr_flux_columns = ["flux_kr_36", "flux_kr_45", "flux_kr_58", "flux_kr_80"]

    ap_lookup = {"ap3": "aper_30", "kr": "kron"}
    band_lookup = [36, 45, 58, 80]
    
    err_factor = 2.5 / np.log(10) # natural log!
    for old_ap, new_ap in ap_lookup.items():
        for band in band_lookup:
            old_flux_col = f"flux_{old_ap}_{band}"
            if not old_flux_col in cat.columns:
                continue
            new_flux_col = f"I{band}_flux_{new_ap}"
            mag_col = f"I{band}_mag_{new_ap}"
            magerr_col = f"I{band}_mag_{new_ap}"            
            cat[mag_col] = -2.5 * np.log10(cat[old_flux_col] / 1e6)
            old_fluxerr_col = f"uncf_{old_ap}_{band}"
            new_fluxerr_col = f"I{band}_fluxerr_{new_ap}"
            snr_col = cat[old_flux_col] / cat[old_fluxerr_col]
            cat[magerr_col] = err_factor * 1. / snr_col
             

            cat.rename_column(old_flux_col, new_flux_col)
            cat.rename_column(old_fluxerr_col, new_fluxerr_col)

    cat.write(output_path, overwrite=True)

if __name__ == "__main__":
    

    base_dir = paths.input_data_path / "external/swire/"
    base_dir.mkdir(exist_ok=True, parents=True)

    field_choices = ["EN", "LH", "XM"]

    parser = ArgumentParser()
    parser.add_argument(
        "--fields", choices=field_choices, nargs="+", default=field_choices, required=False
    )
    parser.add_argument("--force-download", action="store_true", default=False)
    parser.add_argument("--skip-catalog", action="store_true", default=False)
    parser.add_argument("--skip-mask", action="store_true", default=False)
    parser.add_argument("--resolution", default=2.0, type=float)
    # remember: dashes go to underscores after parse, ie, "--skip-mask" -> args.skip_mask 
    args = parser.parse_args()

    for field in args.fields:
        tbl_path = base_dir / f"{field}_swire.tbl"
        if not tbl_path.exists() or args.force_download:
            download_catalog(field, tbl_path)
            
        if not args.skip_catalog:
            output_path = base_dir / f"{field}_swire.cat.fits"
            process_catalog(tbl_path, output_path)

            







