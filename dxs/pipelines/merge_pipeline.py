import logging
import yaml
from argparse import ArgumentParser
from itertools import product

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from regions import read_ds9

from dxs import merge_catalogs, CatalogMatcher, CatalogPairMatcher
from dxs.mosaic_builder import add_keys
from dxs.utils.misc import get_git_info, print_header
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("merge_pipeline")

measurement_lookup = {
    "J": "K",
    "K": "J",
    #"H": "",
}

external_data = ["panstarrs", "cfhtls", "hsc", "unwise", "swire"]

merge_config = survey_config["merge"]


def merge_pipeline(
    field, 
    tiles, 
    bands, 
    mosaic_extension=".cov.good_cov.fits", 
    use_fp_catalogs=False,
    require_all=False,
    force_merge=False,
    external=None,
    output_code=0,
    prefix="",
):
    output_catalogs = {}
    for band in bands:
        output_dir = paths.get_catalog_dir(field, output_code, band)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_stem = paths.get_catalog_stem(field, output_code, band)
        output_path = output_dir / f"{output_stem}.cat.fits"
        output_catalogs[band] = output_path

        if output_path.exists() and not force_merge:
            logger.info(f"SKIP {field} {tiles} {band} merge; {output_path.name} exists")
            continue

        print_header(f"merge {output_stem}")

        if use_fp_catalogs:
            mband = measurement_lookup.get(band, None)
        else: 
            mband = None
      
        data_list = []
        tile_list = []
        for tile in tiles:
            spec = (field, tile, band)
            catalog_path = paths.get_catalog_path(*spec, measurement_band=mband)
            if not catalog_path.exists():
                if require_all:
                    raise IOError("No catalog {catalog_path}")
                else:
                    logger.warn("No catalog {catalog_path.name}")
            mosaic_path = paths.get_mosaic_path(*spec, extension=mosaic_extension)
            if not mosaic_path.exists():
                raise IOError("No mosaic_path {mosaic_path}")
            with fits.open(mosaic_path) as f:
                fwcs = WCS(f[0].header)
            data_list.append((catalog_path, fwcs))
            tile_list.append(tile)

        id_col, ra_col, dec_col, snr_col = (
            f"{band}_id", f"{band}_ra", f"{band}_dec", f"{band}_snr_auto"
        )
        merge_kwargs = {
            "merge_error": merge_config["tile_merge_error"], 
            "coverage_col": f"{band}_coverage",
            "value_check_column": f"{band}_mag_auto", "vcheck_max": 22.7,
            "atol": 0.2
        }

        ### do the merge!!!!
        print(f"merging: {field} {tile_list} {band}")
        merge_catalogs(
            data_list, output_path, id_col, ra_col, dec_col, snr_col, **merge_kwargs
        )
            
        branch, local_SHA = get_git_info()
        meta_data = {
            "branch": (branch, "pipeline branch"),
            "localSHA": (local_SHA, "commit SHA ID"),
        }
        add_keys(output_path, meta_data, verbose=True)   
        
    J_output = output_catalogs.get("J", None)
    K_output = output_catalogs.get("K", None)

    if J_output is None and K_output is None:
        # this would only happen on H only merge?
        return None

    H_output = output_catalogs.get("H", None)
    external = external or []
    if H_output is not None:
        external = external.insert(0, "H")

    nir_output_dir = paths.get_catalog_dir(field, output_code, "")

    if J_output and K_output:
        print_header("merge J and K")
        logger.info("start")
        nir_output_stem = paths.get_catalog_stem(field, output_code, "", prefix=prefix)
        nir_output_path = nir_output_dir / f"{nir_output_stem}.cat.fits"
        nir_matcher = CatalogPairMatcher(
            J_output, K_output, nir_output_path,
            output_ra="ra", output_dec="dec",
            ra1="J_ra", dec1="J_dec",
            ra2="K_ra", dec2="K_dec",
        )
        nir_matcher.ra = "ra"
        nir_matcher.dec = "dec"

        if nir_output_path.exists() and not force_merge:
            print("skipping J&K merge")
        else:
            nir_matcher.best_pair_match(error=merge_config["nir_match_error"]) # arcsec
            nir_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
            nir_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")

            tab = Table.read(nir_matcher.catalog_path)
            if "id" not in tab.columns:
                tab.add_column(np.arange(len(tab)), name="id")
            tab.write(nir_matcher.catalog_path, overwrite=True)

    elif (J_output and not K_output) or (K_output and not J_output):
        nir_matcher = CatalogMatcher(catalog)
        nir_matcher.ra = f"{band}_ra"
        nir_matcher.dec = f"{band}_dec"

    ext_output_stem = nir_output_stem
    for ext in external:
        print_header(f"add {ext} catalog")
        if ext == "H":
            ext_catalog_path = H_output
            ext_match_error = merge_config["nir_match_error"]
        else:
            ext_name = f"{field}_{ext}"
            ext_catalog = (
                paths.input_data_path / f"external/{ext}/catalogs/{ext_name}.cat.fits"
            )
            ext_match_error = merge_config[f"{ext}_match_error"]

        output_stem = nir_output_stem + f"_{ext}"
        ext_output_path = nir_output_dir / f"{output_stem}.cat.fits"

        logger.info(f"output at {output_path.name}")
        logger.info(f"match {ext} with error={ext_match_error:.2f}")
        nir_matcher.match_catalog(
            ext_catalog, 
            output_path=ext_output_path,
            ra=f"ra_{ext}", dec=f"dec_{ext}", 
            error=ext_match_error,
            find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": f"{ext}_separation"})

    print(nir_matcher.summary_info)
            
    try:
        print_path = ext_output_path.relative_to(paths.base_path)
    except:
        print_path = ext_output_path
    logger.info(f"final output at\n    {print_path}")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("fields")
    parser.add_argument("tiles")
    parser.add_argument("bands")
    parser.add_argument("--output-code", action="store", default=0, type=int, required=False)
    parser.add_argument("--force-merge", action="store_true", default=False, required=False)
    parser.add_argument("--use-fp", action="store_true", default=False, required=False)
    parser.add_argument(
        "--external", action="store", nargs="+", 
        default=["panstarrs"], choices=external_data, required=False
    )
    parser.add_argument("--require-all", action="store_true", default=False, required=False)
    parser.add_argument("--prefix", action="store", default="", required=False)
    args = parser.parse_args()

    fields = [x for x in args.fields.split(",")]
    tile_ranges = [x for x in args.tiles.split(",")]
    tiles = []
    for t in tile_ranges:
        if "-" in t:
            ts = t.split("-")
            tiles.extend([x for x in range(int(ts[0]), int(ts[1])+1)])
        else:
            tiles.append(int(t))
    bands = [x for x in args.bands.split(",")]

    for field in fields:
        logger.info(f"merge {field} {tiles} for {bands}")
        merge_pipeline(
            field, tiles, bands, 
            output_code=args.output_code, 
            force_merge=args.force_merge,
            use_fp_catalogs=args.use_fp,
            external=args.external,
            require_all=args.require_all,
            prefix=args.prefix
        )
            



