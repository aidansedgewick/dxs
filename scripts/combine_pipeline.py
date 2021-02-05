import logging
import yaml
from argparse import ArgumentParser
from itertools import product

from dxs import combine_catalogs, CatalogPairMatcher
from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger("main")

measurement_lookup = {
    "J": "K",
    "K": "J",
    "H": None,
}

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("fields")
    parser.add_argument("tiles")
    parser.add_argument("bands")
    parser.add_argument("--output_code", action="store", default=0, type=int, required=False)
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
        combined_band = f"{band}{measurement_band}"
            
        catalog_lists = []
        for field in fields:
            catalog_list = []
            for tile in tiles:
                catalog_dir = paths.get_catalog_dir(field, tile, band)
                catalog_stem = paths.get_catalog_stem(
                    field, tile, combined_band
                )            
                catalog_list.append(catalog_dir / f"{catalog_stem}.fits")
            print([x.stem for x in catalog_list])
            
            combined_dir = paths.get_catalog_dir(field, args.output_code, band)
            combined_dir.mkdir(exist_ok=True, parents=True)
            combined_stem = paths.get_catalog_stem(field, args.output_code, band)

            combined_output_path = combined_dir / f"{combined_stem}.fits"
            id_col = f"{band}_id"
            ra_col = f"{band}_ra"
            dec_col = f"{band}_dec"
            snr_col = f"{band}_snr_auto"
            combine_catalogs(
                catalog_list, combined_output_path, 
                id_col=id_col, ra_col=ra_col, dec_col=dec_col, snr_col=snr_col
            )
            catalog_lists.append(catalog_list)

            output_key = f"{band}_output"
            if output_key not in output_catalogs[field]:
                raise ValueError(f"I don't know what to do with {output_key}")
            output_catalogs[field][output_key] = combined_output_path
    

    for field in fields:

        J_output = output_catalogs[field].get("J_output")
        K_output = output_catalogs[field].get("K_output")
        print(J_output)
        print(K_output)

        if J_output and K_output:
            pair_output_stem = paths.get_catalog_stem(field, args.output_code, "")
            pair_output_dir = paths.get_catalog_dir(field, args.output_code, "")
            pair_output_path = pair_output_dir / f"{pair_output_stem}.fits"
            pair_matcher = CatalogPairMatcher(
                J_output, K_output, pair_output_path,
                output_ra="ra", output_dec="dec",
                ra1="J_ra", dec1="J_dec",
                ra2="K_ra", dec2="K_dec",
            )
            pair_matcher.best_pair_match(error=2.0) # arcsec
            pair_matcher.fix_column_names(column_lookup={"Separation": "JK_separation"})
            pair_matcher.select_best_coords(snr1="J_snr_auto", snr2="K_snr_auto")

            ps_name = f"{field}_panstarrs"
            ps_catalog_path = paths.input_data_path / f"external/panstarrs/{ps_name}.fits"
            pair_matcher.match_catalog(ps_catalog_path, ra="i_ra", dec="i_dec", error=2.0)
            pair_matcher.fix_column_names(column_lookup={"Separation": "ps_separation"})

            

    

            



