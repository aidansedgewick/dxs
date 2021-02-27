import json
import logging
import shutil
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column, vstack
from astropy.wcs import WCS

from astromatic_wrapper.api import Astromatic
from easyquery import Query

from dxs.crosstalk_processor import CrosstalkProcessor
#from dxs.mosaic_builder import get_mosaic_dir, get_mosaic_stem
from dxs.pystilts import Stilts
from dxs.utils.misc import check_modules, format_flags, create_file_backups
from dxs.utils.table import fix_column_names, table_to_numpynd
from dxs.utils.region import in_only_one_tile

from dxs import paths

logger = logging.getLogger("catalog_merge")

def merge_catalogs(
    catalog_list, mosaic_list, output_path, id_col, ra_col, dec_col, snr_col, error=0.5, 
    value_check_column=None, rtol=1e-5, atol=1e-8, tiles_to_ignore=None
):   
    output_path = Path(output_path)
    temp_concat_path = paths.temp_sextractor_path / f"{output_path.stem}_concat.fits"
    temp_overlap_path = paths.temp_sextractor_path / f"{output_path.stem}_overlap.fits"
    temp_single_path = paths.temp_sextractor_path / f"{output_path.stem}_single.fits"
    temp_output_path = paths.temp_sextractor_path / f"{output_path.name}"
    # Backup all files
    catalog_list = create_file_backups(catalog_list, paths.temp_sextractor_path)
    output_path = Path(output_path)
    for ii, catalog_path in enumerate(catalog_list):
        id_modifier = int(f"1{ii+1:02d}")*1_000_000
        _modify_id_value(catalog_path, id_modifier, id_col=id_col)
    # Concatenate them...
    stilts = Stilts.tcat_fits(catalog_list, output_path=temp_concat_path)
    stilts.run()
    # Now select the objects which are in the overlapping regions.
    concat = Table.read(temp_concat_path)
    coords = SkyCoord(ra=concat[ra_col], dec=concat[dec_col], unit="degree")
    single_tile_mask = in_only_one_tile(coords, mosaic_list)
    single = concat[ single_tile_mask ]
    overlap = concat[ ~single_tile_mask ]

    single.write(temp_single_path, overwrite=True)
    overlap.write(temp_overlap_path, overwrite=True)
    # Do internal match to select objects which appear in more than one frame.
    stilts = Stilts.tmatch1_sky_fits(
        temp_overlap_path, output_path=temp_overlap_path, 
        ra=ra_col, dec=dec_col, error=error
    )
    stilts.run()
    overlap = Table.read(temp_overlap_path)
    # separate into unique objects in overlap region vs. grouped objects in overlap region.
    overlap["GroupID"] = overlap["GroupID"].filled(-99)
    overlap_groups = overlap[ overlap["GroupID"] >=0 ]
    overlap_unique = overlap[ overlap["GroupID"] == -99 ]
    logger.info(f"merge - {len(overlap_unique)} unique sources in in overlap region")

    grouped = overlap_groups.group_by("GroupID")
    split_at = grouped.groups.indices
    group_lengths = np.diff(split_at)
    n_groups = len(group_lengths)
    # Have we matched the correct objects?!
    val_check_maxima = np.maximum.reduceat(grouped[value_check_column], split_at[:-1])
    val_check_minima = np.minimum.reduceat(grouped[value_check_column], split_at[:-1])
    close = np.isclose(val_check_maxima, val_check_minima, atol=atol, rtol=rtol).sum()
    if close != n_groups:
        logger.warn(f"{n_groups-close} matches have {value_check_column} diff >tolerances")

    snr_maxima = np.maximum.reduceat(grouped[snr_col], split_at[:-1]) # Find maxima split by group.
    argmax_mask = (np.repeat(snr_maxima, group_lengths) == grouped[snr_col])
    argmax_inds = np.arange(len(grouped[snr_col]))[ argmax_mask ]

    grouped_best = grouped[ argmax_inds ]
    assert len(grouped_best["GroupID"]) == len(np.unique(grouped_best["GroupID"]))
    assert len(grouped_best) == n_groups

    #index = np.repeat(np.arange(n_groups), group_lengths)
    #all_argmax = np.flatnonzero(np.repeat(maxima, group_lengths) == arr) # Repeat each maximum the number of times in each group. 
    #result = all_argmax[np.unique(index[all_argmax], return_index=True)[1]]

    single = Table.read(temp_single_path)    
    output = vstack([single, grouped_best, overlap_unique])
    output.remove_columns(["GroupSize", "GroupID"])
    output.write(temp_output_path, overwrite=True)

    shutil.copy2(temp_output_path, output_path)

    logger.info(f"done merging - output {output_path}!")


def _modify_id_value(catalog_path, id_modifier, id_col="id",):
    catalog = Table.read(catalog_path)
    catalog[id_col] = id_modifier + catalog[id_col]
    catalog.write(catalog_path, overwrite=True)

