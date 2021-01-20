import logging
import yaml

import numpy as np

import astropy.io.fits as fits
from astropy.table import Table

from dxs import paths

logger = logging.getLogger("table_utils")

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
    suffix=None,
    hdu=1
):
    print(catalog_path)
    with fits.open(catalog_path, mode="update") as catalog:
        header = catalog[hdu].header
        catalog_columns = {
            f"TTYPE{ii}": header[f"TTYPE{ii}"] for ii in range(1, header["TFIELDS"]+1)
        }
        
        if band is not None:
            band_fix = f"{band}_"
        else:
            band_fix = ""
        prefix = prefix or ""
        suffix = suffix or ""
        
        if input_columns is None:
            lookup = {}
        elif isinstance(input_columns, list):
            if output_columns is None:
                output_columns = input_columns #[x.lower() for x in input_columns]
            if isinstance(output_columns, list):
                if len(input_columns) != len(output_columns):
                    li, lo = len(input_columns), len(output_columns)
                    raise ValueError(
                        f"len input_columns ({li}) not equal to len output_columns ({lo})"
                    )
            lookup = {i:o for i, o in zip(input_columns, output_columns)}
        elif input_columns == "catalog":
            if output_columns is not None:
                logger.warn("input is \"catalog\" so ignoring output_columns {output_columns}")
            lookup = {x: x.lower() for x in catalog_columns.values()}
        else:
            raise ValueError(
                f"input_columns is list or string \"catalog\", not {input_columns}"
            )
        print(catalog_path.stem, "lookup is", lookup)
        column_lookup = column_lookup or {}
        lookup.update(column_lookup)
        print(catalog_path.stem, "lookup is", lookup)
        for column_ttype, column in catalog_columns.items():
            if column in lookup:
                new_name = f"{band_fix}{prefix}{lookup[column]}{suffix}"
                catalog[hdu].header[column_ttype] = new_name
        new_cols = [
            catalog[hdu].header[f"TTYPE{ii}"] 
            for ii in range(1, catalog[hdu].header["TFIELDS"]+1)
        ]
        counts = {x: new_cols.count(x) for x in set(new_cols)}
        repeats = {k: v for k, v in counts.items() if v > 1}
        if len(repeats) > 0:
            raise ValueError(f"Can't have repeated column names, {repeats}.")

        if output_path is None or output_path == catalog_path:
            catalog.flush()
        else:
            catalog.writeto(output_path, overwrite=True)

def fix_sextractor_column_names(catalog_path, band=None, prefix=None, suffix=None):
    sextractor_lookup = _load_sextractor_column_lookup()
    fix_column_names(
        catalog_path, 
        input_columns="catalog",
        column_lookup=sextractor_lookup, 
        band=band, 
        prefix=prefix, 
        suffix=suffix
    )

def _load_sextractor_column_lookup(lookup_path=default_lookup_path):
    with open(lookup_path, "r") as f:
        sextractor_lookup = yaml.load(f, Loader=yaml.FullLoader)
    return sextractor_lookup


