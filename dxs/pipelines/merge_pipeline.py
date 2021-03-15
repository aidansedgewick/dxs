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

logger = logging.getLogger("merge_pipeline")

measurement_lookup = {
    "J": "K",
    "K": "J",
    "H": "",
}

external_data = ["panstarrs", "cfhtls", "hsc", "unwise"]

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
        mosaic_list = []
        for tile in tiles:
            catalog_dir = paths.get_catalog_dir(field, tile, band)
            catalog_stem = paths.get_catalog_stem(
                field, tile, cat_band, prefix=prefix
            )
            catalog_list.append(catalog_dir / f"{catalog_stem}.fits")
            mosaic_path = paths.get_mosaic_path(
                field, tile, band, extension=".cov.good_cov.fits"
            )
            mosaic_list.append(mosaic_path)

        print([x.stem for x in catalog_list])
        
        combined_dir = paths.get_catalog_dir(field, args.output_code, band)
        combined_dir.mkdir(exist_ok=True, parents=True)
        combined_stem = paths.get_catalog_stem(
            field, args.output_code, band, prefix=args.prefix
        )

        combined_output_path = combined_dir / f"{combined_stem}.fits"
        if skip_merge is False:
            logger.info(f"merging {field} {band}")
            id_col = f"{band}_id"
            ra_col = f"{band}_ra"
            dec_col = f"{band}_dec"
            snr_col = f"{band}_snr_auto"
            merge_catalogs(
                catalog_list, mosaic_list, combined_output_path, 
                id_col=id_col, ra_col=ra_col, dec_col=dec_col, snr_col=snr_col,
                value_check_column=f"{band}_mag_aper_30", atol=0.1
            )
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
        nir_output_dir = paths.get_catalog_dir(field, output_code, "", prefix=prefix)
        nir_output_path = nir_output_dir / f"{nir_output_stem}.fits"
        nir_matcher = CatalogPairMatcher(
            J_output, K_output, nir_output_path,
            output_ra="ra", output_dec="dec",
            ra1="J_ra", dec1="J_dec",
            ra2="K_ra", dec2="K_dec",
        )
        nir_matcher.best_pair_match(error=1.0) # arcsec
        nir_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
        nir_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")
        nir_matcher.ra = external_match_ra
        nir_matcher.dec = external_match_dec

    elif (J_output and not K_output) or (K_output and not J_output):
        band = "J" if J_output else "K"
        catalog = J_output if J_output else K_output
        logger.info(f"use {catalog.stem}.")

        nir_output_stem = paths.get_catalog_stem(field, output_code, band, prefix=prefix)
        nir_matcher = CatalogMatcher(catalog, output_path=nir_output_path)
        if len(external) > 0:
            nir_output_stem += "_" + "_".join(external)
        nir_output_dir = paths.get_catalog_dir(field, output_code, "")
        nir_output_path = nir_output_dir / f"{nir_output_stem}.fits"
        nir_matcher.ra = external_match_ra
        nir_matcher.dec = external_match_dec
        

    if external is None:
        return None

    if "panstarrs" in external:
        ps_name = f"{field}_panstarrs"
        ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
        nir_matcher.match_catalog(
            ps_catalog_path, ra="ra_panstarrs", dec="dec_panstarrs", 
            error=1.0, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})
    if "cfhtls" in external:
        cfhtls_name = f"{field}_i"
        cfhtls_catalog_path = (
            paths.input_data_path / f"external/cfhtls/{cfhtls_name}.fits"
        )
        nir_matcher.match_catalog(
            cfhtls_catalog_path, ra="ra_cfhtls", dec="dec_cfhtls", 
            error=0.5, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "cfhtls_separation"})
    if "hsc" in args.external:
        hsc_name = f"{field}_catalog"
        hsc_catalog_path = (
            paths.input_data_path / f"external/hsc/catalogs/{hsc_name}.fits"
        )
        nir_matcher.match_catalog(
            hsc_catalog_path, ra="ra_hsc", dec="dec_hsc",
            error=1.0, find="best1"
        )
        nir_matcher.fix_column_names(column_lookup={"Separation": "hsc_separation"})
    if "unwise" in args.external:
        unwise_name = f"{field}_unwise.fits"
        unwise_catalog_path = (
            paths.input_data_path / f"external/unwise/catalogs/{field}" / unwise_name
        )
        nir_matcher.match_catalog(unwise_catalog_path, ra="ra_unwise", dec="dec_unwise", error=2.0)
        nir_matcher.fix_column_names(column_lookup={"Separation": "unwise_separation"})
    
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
            



