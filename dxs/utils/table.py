import logging
import yaml
from pathlib import Path

import numpy as np

import astropy.io.fits as fits
from astropy.table import Table

from easyquery import Query

from dxs import paths

logger = logging.getLogger("table_utils")

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
        column_lookup = column_lookup or {}
        lookup.update(column_lookup)
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
        
def remove_objects_in_bad_coverage(
    catalog_path, 
    coverage_map_path, 
    coverage_column, 
    N_pixels=3500*3500,
    absolute_minimum=3, 
    frac=0.9, 
    hdu=0
):
    catalog_path = Path(catalog_path)
    coverage_map_path = Path(coverage_map_path)
    with fits.open(coverage_map_path) as f:
        data = f[hdu].data.astype(int)
        coverage_hist = np.bincount(data.flatten())        
        coverage_vals = np.arange(len(coverage_hist))
        print({k:v for k,v in zip(coverage_vals, coverage_hist)})
        min_coverage = coverage_vals[ coverage_hist > N_pixels ][0]
        min_coverage = np.ceil(frac*min_coverage)
        minimum_coverage = int(np.max([absolute_minimum, min_coverage, 1]))
        data[data < minimum_coverage] = 0

        new_image = fits.PrimaryHDU(data=data, header=f[hdu].header)
        new_image_path = coverage_map_path.with_suffix(".good_cov.fits")
        new_image.writeto(new_image_path, overwrite=True)
    logger.info(f"{catalog_path.stem}: keep objects with >={minimum_coverage} stack coverage.")

    catalog = Table.read(catalog_path)
    clen = len(catalog)
    new_catalog = Query(f"{coverage_column} >= {minimum_coverage}").filter(catalog)
    nclen = len(new_catalog)
    logger.info(f"remove {clen-nclen} objects")
    new_catalog.write(catalog_path, overwrite=True)

    
        






