import logging
import yaml
from argparse import ArgumentParser
from itertools import product

from astropy.table import Table

from regions import read_ds9

from dxs import merge_catalogs, CatalogMatcher, CatalogPairMatcher
from dxs.mosaic_builder import add_keys
from dxs.utils.misc import get_git_info
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("merge_pipeline")

measurement_lookup = {
    "J": "K",
    "K": "J",
    "H": "",
}

external_data = ["panstarrs", "cfhtls", "hsc", "unwise", "swire"]

merge_config = survey_config["merge"]

def merge_pipeline(
    field, tiles, bands, prefix=None, external=None, 
    external_match_ra="ra", external_match_dec="dec",
    output_code=False, skip_merge=False, include_fp=False, require_all=False
):

    output_catalogs = {}
    for band in bands:
        measurement_band = measurement_lookup[band]
        if include_fp:
            cat_band = f"{band}{measurement_band}"
        else:
            cat_band = band    
        catalog_lists = []
        catalog_list = []
        region_list = []
        for tile in tiles:
            catalog_dir = paths.get_catalog_dir(field, tile, band)
            catalog_stem = paths.get_catalog_stem(
                field, tile, cat_band, prefix=prefix
            )
            catalog_list.append(catalog_dir / f"{catalog_stem}.cat.fits")
            #mosaic_path = paths.get_mosaic_path(
            #    field, tile, band, extension=".cov.good_cov.fits"
            #)
            #mosaic_list.append(mosaic_path)
            region_path = paths.get_mosaic_path(
                field, tile, band, extension=".reg"
            )
            region = read_ds9(region_path)
            assert len(region) == 1
            region_list.append(region[0])

        print("merging", [x.stem for x in catalog_list])
        
        if require_all:
            if not all([mp.exists() for mp in region_list]):
                logger.info("skipping merge")
                continue

        combined_dir = paths.get_catalog_dir(field, args.output_code, band)
        combined_dir.mkdir(exist_ok=True, parents=True)
        combined_stem = paths.get_catalog_stem(
            field, args.output_code, band, prefix=args.prefix
        )

        combined_output_path = combined_dir / f"{combined_stem}.cat.fits"
        if skip_merge is False:
            logger.info(f"merging {field} {band}")
            id_col = f"{band}_id"
            ra_col = f"{band}_ra"
            dec_col = f"{band}_dec"
            snr_col = f"{band}_snr_auto"
            merge_error = merge_config["tile_merge_error"]
            merge_catalogs(
                catalog_list, region_list, combined_output_path, 
                error=merge_error,
                id_col=id_col, ra_col=ra_col, dec_col=dec_col, snr_col=snr_col,
                coverage_col = f"{band}_coverage", 
                value_check_column=f"{band}_mag_auto", 
                atol=0.1
            )
            branch, local_SHA = get_git_info()
            data = {
                "branch": (branch, "pipeline branch"),
                "localSHA": (local_SHA, "commit SHA ID"),
            }
            add_keys(combined_output_path, data, verbose=True)                        
        output_catalogs[band] = combined_output_path
        catalog_lists.append(catalog_list)
                
    J_output = output_catalogs.get("J", None)
    K_output = output_catalogs.get("K", None)

    if J_output is None and K_output is None:
        return None

    if J_output and K_output:
        logger.info("merge J and K")
        nir_output_stem = paths.get_catalog_stem(field, output_code, "", prefix=prefix)
        if len(external) > 0:
            nir_output_stem += "_" + "_".join(external)
        nir_output_dir = paths.get_catalog_dir(field, output_code, "")
        nir_output_path = nir_output_dir / f"{nir_output_stem}.cat.fits"
        nir_matcher = CatalogPairMatcher(
            J_output, K_output, nir_output_path,
            output_ra="ra", output_dec="dec",
            ra1="J_ra", dec1="J_dec",
            ra2="K_ra", dec2="K_dec",
        )
        nir_matcher.best_pair_match(error=merge_config["nir_match_error"]) # arcsec
        nir_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
        nir_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")
        nir_matcher.ra = external_match_ra
        nir_matcher.dec = external_match_dec

    elif (J_output and not K_output) or (K_output and not J_output):
        band = "J" if J_output else "K"
        catalog = J_output if J_output else K_output
        logger.info(f"use {catalog.stem}.")

        nir_output_stem = paths.get_catalog_stem(field, output_code, band, prefix=prefix)
        nir_output_dir = paths.get_catalog_dir(field, output_code, "")
        nir_output_path = nir_output_dir / f"{nir_output_stem}.cat.fits"
        nir_matcher = CatalogMatcher(catalog, output_path=nir_output_path)
        if len(external) > 0:
            nir_output_stem += "_" + "_".join(external)
        nir_matcher.ra = f"{band}_ra"
        nir_matcher.dec = f"{band}_dec"
        

    if external is None:
        return None

    if "panstarrs" in external:
        ps_name = f"{field}_panstarrs"
        ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
        ps_error = merge_config["panstarrs_match_error"]
        logger.info(f"match panstarrs with error={ps_error:.2f}")
        nir_matcher.match_catalog(
            ps_catalog_path, ra="ra_panstarrs", dec="dec_panstarrs", 
            error=ps_error, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})
    if "cfhtls" in external:
        cfhtls_name = f"{field}_i"
        cfhtls_catalog_path = (
            paths.input_data_path / f"external/cfhtls/{cfhtls_name}.fits"
        )
        cfhtls_error = merge_config["cfhtls_match_error"]
        nir_matcher.match_catalog(
            cfhtls_catalog_path, ra="ra_cfhtls", dec="dec_cfhtls", 
            error=cfhtls_error, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "cfhtls_separation"})
    if "hsc" in args.external:
        hsc_name = f"{field}_catalog"
        hsc_catalog_path = (
            paths.input_data_path / f"external/hsc/catalogs/{hsc_name}.fits"
        )
        hsc_error=merge_config["hsc_match_error"]
        nir_matcher.match_catalog(
            hsc_catalog_path, ra="ra_hsc", dec="dec_hsc",
            error=hsc_error, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "hsc_separation"})
    if "unwise" in args.external:
        unwise_name = f"{field}_unwise.fits"
        unwise_catalog_path = (
            paths.input_data_path / f"external/unwise/catalogs/{field}" / unwise_name
        )
        unwise_error = merge_config["unwise_match_error"]
        logger.info("match unwise with error={unwise_error:.2f}")
        nir_matcher.match_catalog(
            unwise_catalog_path, ra="ra_unwise", dec="dec_unwise",
            error=unwise_error, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "unwise_separation"})
    if "swire" in args.external:
        swire_name = f"{field}_swire.cat.fits"
        swire_catalog_path = (
            paths.input_data_path / "external/swire/catalogs" / swire_name
        )
        swire_error = merge_config["swire_match_error"]
        logger.info(f"match swire with error={swire_error:.2f}")
        nir_matcher.match_catalog(
            swire_catalog_path, ra="ra_swire", dec="dec_swire",
            error=swire_error, find="best1"
        )        

    else:
        return None

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
    parser.add_argument("--require-all", action="store_true", default=False, required=False)
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

    for field in fields:
        merge_pipeline(
            field, tiles, bands, 
            output_code=args.output_code, 
            skip_merge=args.skip_merge,
            include_fp=args.include_fp,
            external=args.external,
            require_all=args.require_all,            
        )
            



