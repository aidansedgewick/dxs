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
from regions import read_ds9

from dxs.crosstalk_processor import CrosstalkProcessor
from dxs.pystilts import Stilts
from dxs.utils.misc import check_modules, format_flags, create_file_backups
from dxs.utils.table import fix_column_names, table_to_numpynd
from dxs.utils.region import in_only_one_tile, in_only_one_region, guess_wcs_from_region

from dxs import paths

logger = logging.getLogger("catalog_merge")

def merge_catalogs(
    catalog_list, region_list, output_path, id_col, ra_col, dec_col, snr_col,
    error=0.5, coverage_col=None, value_check_column=None, rtol=1e-5, atol=1e-8
):   
    output_path = Path(output_path)
    stem = output_path.stem
    temp_concat_path = paths.temp_stilts_path / f"{stem}_concat.fits"
    temp_overlap_path = paths.temp_stilts_path / f"{stem}_overlap.fits"
    temp_single_path = paths.temp_stilts_path / f"{stem}_single.fits"
    temp_output_path = paths.temp_stilts_path / f"{output_path.name}"
    # Backup all files
    catalog_list = create_file_backups(catalog_list, paths.temp_stilts_path)
    output_path = Path(output_path)
    for ii, catalog_path in enumerate(catalog_list):
        id_modifier = int(f"1{ii+1:02d}")*1_000_000
        _modify_id_value(catalog_path, id_modifier, id_col=id_col)
    # Concatenate them...
    logger.info(f"concatenate all {len(catalog_list)} catalogs")
    concat_list = []
    for catalog_path, region in zip(catalog_list, region_list):
        if isinstance(region, Path) or isinstance(region, str):
            logger.info(f"read {region.name}")
            region = read_ds9(region)
            print(region)
        rwcs = guess_wcs_from_region(region)
        tab = Table.read(catalog_path)
        if coverage_col is not None:
            tab = tab[ tab[coverage_col] > 0 ]

        coord = SkyCoord(ra=tab[ra_col], dec=tab[dec_col], unit="degree")
        in_region_mask = region.contains(coord, rwcs)
        tab = tab[ in_region_mask ]
        concat_list.append(tab)
    concat = vstack(concat_list)
    concat.write(temp_concat_path, overwrite=True)
    #stilts = Stilts.tcat_fits(catalog_list, output_path=temp_concat_path)
    #stilts.run()
    # Now select the objects which are in the overlapping regions.
    
    single, overlap = separate_single_overlap(
        temp_concat_path, ra_col=ra_col, dec_col=dec_col, region_list=region_list
    )
    single.write(temp_single_path, overwrite=True)
    overlap.write(temp_overlap_path, overwrite=True)
    # Do internal match to select objects which appear in more than one frame.
    logger.info("internal match on objects in overlapping regions")
    stilts = Stilts.tmatch1_sky_fits(
        temp_overlap_path, output_path=temp_overlap_path, 
        ra=ra_col, dec=dec_col, error=error
    )
    stilts.run()
    overlap = Table.read(temp_overlap_path)
    # separate into unique objects in overlap region vs. grouped objects in overlap region.
    overlap_unique, overlap_groups = separate_unique_grouped(overlap, "GroupID")
    grouped_best = select_group_argmax(
        overlap_groups, "GroupID", snr_col, value_check_column=value_check_column,
        atol=atol
    )
    single = Table.read(temp_single_path)    
    output = vstack([single, grouped_best, overlap_unique])
    #output.remove_columns(["GroupSize", "GroupID"])
    output.write(temp_output_path, overwrite=True)
    shutil.copy2(temp_output_path, output_path)
    logger.info(f"done merging - output {output_path}!")

def _modify_id_value(catalog_path, id_modifier, id_col="id",):
    catalog = Table.read(catalog_path)
    catalog[id_col] = id_modifier + catalog[id_col]
    catalog.write(catalog_path, overwrite=True)

def separate_single_overlap(table_path, ra_col, dec_col, region_list):
    logger.info(f"separate obj in/not in overlapping regions")
    concat = Table.read(table_path)
    coords = SkyCoord(ra=concat[ra_col], dec=concat[dec_col], unit="degree")
    single_tile_mask = in_only_one_region(coords, region_list)
    single = concat[ single_tile_mask ]
    overlap = concat[ ~single_tile_mask ]
    return single, overlap

def select_group_argmax(
    table, group_id, argmax_col, value_check_column=None, rtol=1e-5, atol=1e-8
):
    grouped = table.group_by(group_id)
    split_at = grouped.groups.indices
    group_lengths = np.diff(split_at)
    n_groups = len(group_lengths)
    # Have we matched the correct objects?!
    if value_check_column is not None:
        val_check_maxima = np.maximum.reduceat(grouped[value_check_column], split_at[:-1])
        val_check_minima = np.minimum.reduceat(grouped[value_check_column], split_at[:-1])
        val_check_diff = val_check_maxima - val_check_minima
        assert len(val_check_diff) == n_groups
        close = np.isclose(val_check_maxima, val_check_minima, atol=atol, rtol=rtol).sum()
        close = (abs(val_check_diff) < atol).sum()
        if close != n_groups:
            logger.warn(
                f"{n_groups-close} of {n_groups} matches have "
                f"{value_check_column} diff > tolerance {atol}"
            )
    else:
        val_check_diff = None
    # Find maxima split by group.
    argmax_maxima = np.maximum.reduceat(grouped[argmax_col], split_at[:-1])
    assert len(argmax_maxima) == n_groups
    argmax_mask = (np.repeat(argmax_maxima, group_lengths) == grouped[argmax_col])
    # in the case where the maximum snr value appears >1 in a group
    # we'd select more than 1 per group!
    gid = grouped[group_id][ argmax_mask ]
    first_in_group = np.full(len(gid), False)
    first_in_group[ np.unique(gid, return_index=True)[1] ] = True
    assert first_in_group.sum() == n_groups

    argmax_inds = np.arange(len(grouped[argmax_col]))[ argmax_mask ]
    if not all(first_in_group):
        assert len(first_in_group) > n_groups
        argmax_inds = argmax_inds[ first_in_group ]
    grouped_best = grouped[ argmax_inds ]
    assert len(grouped_best[group_id]) == len(np.unique(grouped_best[group_id]))
    assert len(grouped_best) == n_groups
    if val_check_diff is not None:
        grouped_best[f"{value_check_column}_range"] = val_check_diff
    return grouped_best

def separate_unique_grouped(table, group_id, fill_value=-99):
    if any(table["GroupID"] < 0):
        raise ValueError("All non-nan entries in group_id={group_id} should be >=0")
    table["GroupID"] = table[group_id].filled(fill_value) #stilts internal match gives NaN for unique object GroupID
    table_groups = table[ table[group_id] >=0 ]
    table_unique = table[ table[group_id] == fill_value ]
    logger.info(f"merge: {len(table_unique)} unique obj in in overlap")
    return table_unique, table_groups


