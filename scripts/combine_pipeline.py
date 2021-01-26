import logging
import yaml
from argparse import ArgumentParser
from itertools import product

from dxs import combine_catalogs
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

    catalog_lists = []
    for combo in combinations:
        #print(combo)
        field, band = combo
        measurement_band = measurement_lookup[band]

        catalog_list = []
        for tile in tiles:
            catalog_dir = paths.get_catalog_dir(field, tile, band)
            catalog_stem = paths.get_catalog_stem(
                field, tile, band, measurement_band=measurement_band
            )            
            catalog_list.append(catalog_dir / f"{catalog_stem}.fits")
        print([x.stem for x in catalog_list])
        
        combined_dir = paths.get_catalog_dir(field, 0, band)
        combined_dir.mkdir(exist_ok=True, parents=True)
        combined_stem = paths.get_catalog_stem(field, 0, band)

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
        



