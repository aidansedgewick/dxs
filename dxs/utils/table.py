import yaml

import numpy as np

import astropy.io.fits as fits
from astropy.table import Table

from dxs import paths

default_lookup_path = paths.config_path / "sextractor/column_name_lookup.yaml"

def table_to_numpynd(table):
    return np.array([x.data for x in table.values()]).T

def fix_column_names(
    catalog_path, 
    output_path=None, 
    input_columns=None, 
    output_columns=None,
    column_lookup=None,
    band=None,
    prefix=None,
    suffix=None
):
    print(catalog_path)
    with fits.open(catalog_path, mode="update") as catalog:
        header = catalog[1].header
        catalog_columns = {
            f"TTYPE{ii}": header[f"TTYPE{ii}"] for ii in range(1, header["TFIELDS"]+1)
        }
        band = f"{band}_" or ""
        prefix = prefix or ""
        suffix = suffix or ""

        skip_renamed = input_columns is None
        input_columns = input_columns or [x for x in catalog_columns.values()]
        output_columns = output_columns or [x.lower() for x in catalog_columns.values()]
        column_lookup = column_lookup or {}

        if not isinstance(input_columns, list):
            input_columns=[input_columns]
        if not isinstance(output_columns, list):
            output_columns = [output_columns]
        lookup = {i:o for i,o in zip(input_columns, output_columns)}
        lookup.update(column_lookup)
        for column_ttype, column in catalog_columns.items():
            if skip_renamed and column.startswith(band):
                continue # probably already renamed
            if column in lookup:
                new_name = f"{band}{prefix}{lookup[column]}{suffix}"
                catalog[1].header[column_ttype] = new_name
        if output_path is None or output_path == catalog_path:
            catalog.flush()
        else:
            catalog.writeto(output_path, overwrite=True)

def fix_sextractor_column_names(catalog_path, band=None, prefix=None, suffix=None):
    sextractor_lookup = _load_sextractor_column_lookup()
    fix_column_names(
        catalog_path, 
        column_lookup=sextractor_lookup, 
        band=band, 
        prefix=prefix, 
        suffix=suffix
    )

def _load_sextractor_column_lookup(lookup_path=default_lookup_path):
    with open(lookup_path, "r") as f:
        sextractor_lookup = yaml.load(f, Loader=yaml.FullLoader)
    return sextractor_lookup


