import logging
import yaml
from argparse import ArgumentParser
from itertools import product

from astropy.table import Table

from dxs import merge_catalogs, CatalogPairMatcher
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("main")

measurement_lookup = {
    "J": "K",
    "K": "J",
    "H": "",
}

external_data = ["panstarrs", "cfhtls", "hsc", "unwise"]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("fields")
    parser.add_argument("tiles")
    parser.add_argument("bands")
    parser.add_argument("--output-code", action="store", default=0, type=int, required=False)
    parser.add_argument("--skip-merge", action="store_true", default=False, required=False)
    parser.add_argument("--include-fp", action="store_true", default=False, required=False)
    parser.add_argument(
        "--external", action="store", nargs="+", default=["panstarrs"], required=False
    )
    parser.add_argument("--prefix", action="store", default=None, required=False)
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

    combinations = product(fields, bands)
    combinations = [x for x in combinations] # don't want generator here

    output_catalogs = {
        field: {
            f"{band}_output": None for band in "J H K".split()
        } for field in fields
    }
    
    for band in bands:
        measurement_band = measurement_lookup[band]
        if args.include_fp:
            cat_band = f"{band}{measurement_band}"
        else:
            cat_band = band    
        catalog_lists = []
        for field in fields:
            catalog_list = []
            mosaic_list = []
            for tile in tiles:
                catalog_dir = paths.get_catalog_dir(field, tile, band)
                catalog_stem = paths.get_catalog_stem(
                    field, tile, cat_band, prefix=args.prefix
                )            
                catalog_list.append(catalog_dir / f"{catalog_stem}.fits")
                mosaic_path = paths.get_mosaic_path(field, tile, band, extension=".cov.good_cov.fits")
                mosaic_list.append(mosaic_path)
    
            print([x.stem for x in catalog_list])
            
            combined_dir = paths.get_catalog_dir(field, args.output_code, band)
            combined_dir.mkdir(exist_ok=True, parents=True)
            combined_stem = paths.get_catalog_stem(
                field, args.output_code, band, prefix=args.prefix
            )

            combined_output_path = combined_dir / f"{combined_stem}.fits"
            id_col = f"{band}_id"
            ra_col = f"{band}_ra"
            dec_col = f"{band}_dec"
            snr_col = f"{band}_snr_auto"

            if args.skip_merge is False:
                merge_catalogs(
                    catalog_list, mosaic_list, combined_output_path, 
                    id_col=id_col, ra_col=ra_col, dec_col=dec_col, snr_col=snr_col,
                    value_check_column=f"{band}_mag_aper_30", atol=0.1
                )
            catalog_lists.append(catalog_list)

            output_key = f"{band}_output"
            if output_key not in output_catalogs[field]:
                raise ValueError(f"I don't know what to do with {output_key}")
            output_catalogs[field][output_key] = combined_output_path
    

    for field in fields:

        J_output = output_catalogs[field].get("J_output")
        K_output = output_catalogs[field].get("K_output")
        print(f"J: {J_output}\nK: {K_output}")
        if not (J_output and K_output):
            print("continue!")
            continue

        if J_output and K_output:
            pair_output_stem = paths.get_catalog_stem(
                field, args.output_code, "", prefix=args.prefix
            )
            pair_output_stem += "_" + "_".join(args.external)
            pair_output_dir = paths.get_catalog_dir(field, args.output_code, "")
            pair_output_path = pair_output_dir / f"{pair_output_stem}.fits"
            pair_matcher = CatalogPairMatcher(
                J_output, K_output, pair_output_path,
                output_ra="ra", output_dec="dec",
                ra1="J_ra", dec1="J_dec",
                ra2="K_ra", dec2="K_dec",
            )
            pair_matcher.best_pair_match(error=1.0) # arcsec
            pair_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
            pair_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")

            if "panstarrs" in args.external:
                pair_matcher.ra = "K_ra"
                pair_matcher.dec = "K_dec"
                ps_name = f"{field}_panstarrs"
                ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
                pair_matcher.match_catalog(
                    ps_catalog_path, ra="ra_panstarrs", dec="dec_panstarrs", 
                    error=1.0, find="best1"
                )
                pair_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})
            if "cfhtls" in args.external:
                cfhtls_name = f"{field}_i"
                cfhtls_catalog_path = (
                    paths.input_data_path / f"external/cfhtls/{cfhtls_name}.fits"
                )
                pair_matcher.match_catalog(
                    cfhtls_catalog_path, ra="ra_cfhtls", dec="dec_cfhtls", 
                    error=0.5, find="best1"
                )
                pair_matcher.fix_column_names(column_lookup={"Separation": "cfhtls_separation"})
            if "hsc" in args.external:
                hsc_name = f"{field}_catalog"
                hsc_catalog_path = (
                    paths.input_data_path / f"external/hsc/catalogs/{hsc_name}.fits"
                )
                pair_matcher.match_catalog(
                    hsc_catalog_path, ra="ra_hsc", dec="dec_hsc",
                    error=1.0, find="best1"
                )
                pair_matcher.fix_column_names(column_lookup={"Separation": "hsc_separation"})
            if "unwise" in args.external:
                unwise_name = f"{field}_unwise.fits"
                unwise_catalog_path = (
                    paths.input_data_path / f"external/unwise/catalogs/{field}" / unwise_name
                )
                pair_matcher.match_catalog(unwise_catalog_path, ra="ra_unwise", dec="dec_unwise", error=2.0)
                pair_matcher.fix_column_names(column_lookup={"Separation": "unwise_separation"})
            
            
        try:
            print_path = pair_output_path.relative_to(Path.cwd())
        except:
            print_path = pair_output_path        
        print(f"now do:\n    topcat {print_path}")

            



