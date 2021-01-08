import numpy as np

import astropy.io.fits as fits
from astropy.table import Table

def table_to_numpynd(table):
    return np.array([x.data for x in table.values()]).T

def fix_column_names(
    catalog_path, 
    outpath=None, 
    input_columns=None, 
    output_columns=None,
    band=None,
    prefix=None,
    suffix=None
):
    with fits.open(catalog_path, mode="update") as catalog:
        header = catalog[1].header
        catalog_columns = {
            f"TTYPE{ii}": header[f"TTYPE{ii}"] for ii in range(1, header["TFIELDS"]+1)
        }
        band = f"{band}_" or ""
        prefix = prefix or ""
        suffix = suffix or ""

        input_columns = input_columns or catalog_columns.values()
        output_columns = output_columns or catalog_columns.values()
        if input_columns == "default":
            column_lookup = _load_sextractor_column_lookup()
        else:
            column_lookup = {i:o for i,o in zip(input_columns, output_columns)}
        for column_ttype, column in enumerate(catalog_columns):
            if column in column_lookup:
                new_name = "{band}{prefix}{column_lookup[column]}{suffix}
                header[column_ttype] = new_name
        if output_path is None:
            catalog.flush()
        else:
            catalog.writeto(outpath, overwrite=True)
            
def _load_sextractor_column_lookup(lookup_path=default_lookup_path):
    raise NotImplementedError


