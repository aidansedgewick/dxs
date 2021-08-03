import logging
import yaml
from pathlib import Path
from typing import Dict

import numpy as np

from astropy.io import fits
from astropy.table import Table, Column

from easyquery import Query

from dxs import paths

logger = logging.getLogger("table_utils")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

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
    logger.info(f"fix col names: {catalog_path.name}")
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
        #if output_path is None or output_path == catalog_path:
        #    catalog.flush()
        #else:
        catalog.writeto(output_path, overwrite=True)

def explode_columns_in_fits(
    catalog_path: str, column_names: str, new_column_names=None, suffixes=None, remove=True
):
    """
    if column_names is list, then new_names should either be None, or nested list.
    """
    catalog = Table.read(catalog_path)

    if not isinstance(column_names, list):
        column_names = [column_names]
    if new_column_names is None:
        new_column_names = [None] * len(column_names)
    elif isinstance(new_column_names, list):
        if len(column_names) > 1:
            if len(column_names) != len(new_names):
                raise ValueError(
                    "There should be the same number of column_names as"
                    "new_column_names elements (each a list)"
                )
            if not all([isinstance(x, list) or x is None for x in new_column_names]):
                raise ValueError(
                    "if column_names is a list len>1, "
                    + "then each of new_column_names should be a list "
                    + "-- each with the number of columns to be exploded."
                )
    for column_name, new_names in zip(column_names, new_column_names):        
        explode_column(catalog, column_name, new_names=new_names, suffixes=suffixes, remove=remove)
        logger.info(f"explode {column_name}")
    catalog.write(catalog_path, overwrite=True)


def explode_column(
    table: Table, column_name: str, new_names=None, suffixes=None, remove=True
):
    col = table[column_name]
    if len(col.shape) != 2:
        logger.info(f"Can't explode column {column_name}, shape {col.shape}")
        return
    N_cols = col.shape[1]
    if new_names is None:
        if suffixes is None:
            suffixes = ["_{ii}" for ii in range(N_cols)]
        assert len(suffixes) == N_cols
        new_names = [f"{column_name}_{suffix}" for suffix in suffixes]
    assert len(new_names) == N_cols

    for ii, col_name in enumerate(new_names):
        new_col = col[:, ii]
        table.add_column(new_col, name=col_name)
    if remove:
        table.remove_column(column_name)
        
def remove_objects_in_bad_coverage(
    catalog_path, coverage_column, minimum_coverage,
):
    catalog = Table.read(catalog_path)
    c_len = len(catalog)

    queries = (f"{coverage_column} >= {minimum_coverage}", )
    new_catalog = Query(*queries).filter(catalog)
    nc_len = len(new_catalog)
    logger.info(f"remove {c_len-nc_len} objects in bad coverage")
    new_catalog.write(catalog_path, overwrite=True)    

def add_map_value_to_catalog(
    catalog_path, mosaic_path, column_name, ra=None, dec=None, xpix=None, ypix=None, hdu=0
):
    catalog = Table.read(catalog_path)
    with fits.open(mosaic_path) as mosaic:
        logger.info(f"open {mosaic_path}")
        mosaic_data = mosaic[hdu].data
        header = mosaic[hdu].header
    use_coords = all([ra, dec])
    use_pixels = all([xpix, ypix])
    err_msg = "Must provide xpix, ypix column names OR ra, dec column names -- not both."
    if use_coords and use_pixels:
        raise ValueError(err_msg)
    if use_coords:
        mosaic_wcs = WCS(header)
        image_positions = table_to_numpynd( catalog[[ra, dec]] )  # TODO: use SkyCoord?
        pixels = mosaic_wcs.wcs_world2pix(image_positions, 0)
        x_values = pixels[:,0].astype(int)
        y_values = pixels[:,1].astype(int)
    elif xpix is not None and ypix is not None:
        x_values = catalog[xpix].astype(int)
        y_values = catalog[ypix].astype(int)
    else:
        raise ValueError(err_msg)
    map_values = mosaic_data[ y_values, x_values ]
    col = Column(map_values, column_name)
    catalog.add_column(col)
    catalog.write(catalog_path, overwrite=True)
    info = f"added map data {column_name} from {mosaic_path.stem}"
    return info

def add_column_to_catalog(catalog_path, column_data: Dict):
    """
    Opens FITS file, adds some data.
    
    Returns a string with some info.
    """

    catalog = Table.read(catalog_path)
    for column_name, column_values in column_data.items():
        catalog.add_column(column_values, name=column_name)
    catalog.write(catalog_path, overwrite=True)
    return f"add {len(column_data)} columns: " + " ".join(c for c in column_data.keys())




