import json
import logging
import os
import shutil
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn, vstack
from astropy.time import Time
from astropy.wcs import WCS

from astromatic_wrapper.api import Astromatic
from easyquery import Query
from regions import read_ds9

from stilts_wrapper import Stilts

from dxs.crosstalk_processor import CrosstalkProcessor
from dxs.utils.misc import check_modules, format_flags, create_file_backups
from dxs.utils.table import fix_column_names, table_to_numpynd
#from dxs.utils.region import in_only_one_tile, in_only_one_region, guess_wcs_from_region

from dxs import paths

logger = logging.getLogger("catalog_merge")

def merge_catalogs(
    data_list, 
    output_path, 
    id_col, 
    ra_col, 
    dec_col, 
    snr_col,
    merge_error=0.8, 
    coverage_col=None, 
    value_check_column=None, 
    vcheck_min=-np.inf,
    vcheck_max=np.inf,
    rtol=1e-5, 
    atol=1e-8,
):   
    """
    Merge catalogs.

    Arguments
    ---------
    data_list
        list of tuples (astropy_table, wcs) or (catalog_path, wcs)
    output_path 
        where to save the output table.
    id col
        We modify the ID of each table so that object IDs are not mangled.
    ra_col, dec_col
        cols to merge positions 
    snr_col
        select the object in overlapping regions with the highest SNR.
    merge_error
        how far apart in arcsec can objects be and be merged? in ARCSEC. default 0.8"
    coverage_col
        don't merge objects in bad coverage (ie, if coverage_col  value is 0)
    value_check_column
    
    """
    output_path = Path(output_path)
    stem = output_path.name.split(".")[0]



    # Concatenate them...
    concat_list = []
    wcs_list = []
    for ii, (tab, assoc_wcs) in enumerate(data_list):
        if isinstance(tab, str) or isinstance(tab, Path):
            catalog_path = Path(tab)
            logger.info(f"reading {catalog_path.name}")
            tab = Table.read(catalog_path)

        # modify id value.
        id_modifier = int(f"1{ii+1:02d}")*1_000_000
        tab[id_col] = tab[id_col] + id_modifier

        if coverage_col is not None:
            tab = tab[ tab[coverage_col] > 0 ]
        coord = SkyCoord(ra=tab[ra_col], dec=tab[dec_col], unit="degree")       
        in_region_mask = assoc_wcs.footprint_contains(coord)

        ### what if everything is NOT in the region_mask?
        if not sum(in_region_mask) == len(in_region_mask):
            xpix, ypix = SkyCoord(
                    ra=tab[~in_region_mask][ra_col], 
                    dec=tab[~in_region_mask][dec_col],
                    unit="deg"
            ).to_pixel(assoc_wcs)
            xlen, ylen = assoc_wcs.pixel_shape
            x_edge_effect = (abs(xpix) < 1.0) | (abs(xpix-xlen) < 1.0)
            y_edge_effect = (abs(ypix) < 1.0) | (abs(ypix-ylen) < 1.0)
            is_edge_effect = x_edge_effect | y_edge_effect
            if not all(is_edge_effect):
                print("ODD XPIX", xpix)
                print("ODD YPIX", ypix)
                          
        # everything here is in the mosaic...
        tab = tab[ in_region_mask ]
        concat_list.append(tab)
        wcs_list.append(assoc_wcs)

    # Concatenate all the individual tables.
    concat_time = Time.now().fits
    logger.info(f"DATE-HDU={concat_time}")
    for cat in concat_list:
        cat.meta["DATE-HDU"] = concat_time
    concat = vstack(concat_list) # astropy Table vstack.

    # Now select the objects which are in the overlapping regions.
    coords = SkyCoord(ra=concat[ra_col], dec=concat[dec_col], unit="deg")
    overlap_mask = contained_in_multiple_wcs(coords, wcs_list)
    single = concat[ ~overlap_mask ]
    overlap = concat[ overlap_mask ]
    
    # write this out temporarily...
    input_overlap_path = paths.scratch_stilts_path / f"{stem}_overlap.cat.fits"
    if input_overlap_path.exists():
        os.remove(input_overlap_path)
    assert not input_overlap_path.exists()
    overlap.write(input_overlap_path, overwrite=True)

    # Do internal match to select objects which appear in more than one frame.
    logger.info("internal match on objects in overlapping regions")
    output_overlap_path = paths.scratch_stilts_path / f"{stem}_overlap_grouped.cat.fits"
    if output_overlap_path.exists():
        os.remove(output_overlap_path)
    assert not output_overlap_path.exists()
    
    stilts = Stilts.tmatch1(
        in_=input_overlap_path, out=output_overlap_path, 
        params=merge_error, all_formats="fits",
        values=f"\"{ra_col} {dec_col}\"", action="identify", matcher="sky"
    )
    print(stilts.cmd)
    stilts.run()
    overlap = Table.read(output_overlap_path)

    # separate into unique objects in overlap region vs. grouped objects in overlap region.
    overlap_unique, overlap_groups = separate_unique_grouped(overlap, "GroupID")
    grouped_best = select_group_argmax(overlap_groups, "GroupID", snr_col)

    if value_check_column is not None:
        value_check_diff = check_for_merge_errors(
            overlap_groups, "GroupID", value_check_column,
            vcheck_min=vcheck_min,
            vcheck_max=vcheck_max,
            atol=atol, 
            rtol=rtol
        )
        grouped_best[f"{value_check_column}_range"] = value_check_diff

    #single = Table.read(temp_single_path)

    output_time = Time.now().fits
    logger.info(f"DATE-HDU={concat_time}")
    for cat in [single, grouped_best, overlap_unique]:
        cat.meta["DATE-HDU"] = output_time

    output = vstack([single, grouped_best, overlap_unique])
    output["GroupID"] = output["GroupID"].filled(-99)
    output["GroupSize"] = output["GroupSize"].filled(-99)

    output.write(output_path, overwrite=True)
    logger.info(f"done merging - output {output_path}!")

def contained_in_multiple_wcs(coords, wcs_list, at_least_one=True):
    grid = np.zeros( (len(coords), len(wcs_list)) )
    for ii, wcs in enumerate(wcs_list):
        grid[:, ii] = wcs.footprint_contains(coords)
    contained_by_N = grid.sum(axis=1)
    if at_least_one:
        N_zeros = sum(contained_by_N < 1)
        if N_zeros > 0:
            raise ValueError(f"{N_zeros} coords not contained by ANY wcs in wcs_list")
    mask = contained_by_N > 1
    return mask

def select_group_argmax(
    table, 
    group_id, 
    argmax_col, 
):
    grouped = table.group_by(group_id)
    split_at = grouped.groups.indices
    group_lengths = np.diff(split_at)
    n_groups = len(group_lengths)

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
    return grouped_best

def check_for_merge_errors(
    table, group_id, value_check_column, 
    vcheck_min=-np.inf,
    vcheck_max=np.inf,
    rtol=1e-5, 
    atol=1e-8
):
    grouped = table.group_by(group_id)
    split_at = grouped.groups.indices
    group_lengths = np.diff(split_at)
    n_groups = len(group_lengths)

    val_check_maxima = np.maximum.reduceat(grouped[value_check_column], split_at[:-1])
    assert len(val_check_maxima) == n_groups
    val_check_minima = np.minimum.reduceat(grouped[value_check_column], split_at[:-1])
    assert len(val_check_minima) == n_groups
    full_val_check_diff = val_check_maxima - val_check_minima

    min_mask = (val_check_minima > vcheck_min)
    max_mask = (val_check_maxima < vcheck_max)
    
    relevant_val_check_maxima = val_check_maxima[ min_mask & max_mask ]
    relevant_val_check_minima = val_check_minima[ min_mask & max_mask ]
    relevant_val_check_diff   = relevant_val_check_maxima - relevant_val_check_minima

    n_relevant_groups = len(relevant_val_check_diff)
    close = np.isclose(
        relevant_val_check_maxima, relevant_val_check_minima, atol=atol, rtol=rtol
    ).sum()
    #close = (abs(relevant_val_check_diff) < atol).sum()
    
    if close != n_relevant_groups:
        diff = n_relevant_groups - close
        
        msg = (
            f"\n{diff} of {n_relevant_groups} objects with"
            + f" {vcheck_min:.2f} < {value_check_column} < {vcheck_max:.2f}"
            + f" are different > tolerance (atol={atol}, rtol={rtol})"
        )
        logger.warning(msg)
    else:
        logger.info(f"value check of {n_relevant_groups} obj is fine")
    return full_val_check_diff



def separate_unique_grouped(table, group_id, fill_value=-99):
    if any(table["GroupID"] < 0):
        raise ValueError("All non-nan entries in group_id={group_id} should be >=0")
    if isinstance(table["GroupID"], MaskedColumn):
        #stilts internal match gives NaN for unique object GroupID
        table["GroupID"] = table[group_id].filled(fill_value) 
    table_groups = table[ table[group_id] >=0 ]
    table_unique = table[ table[group_id] == fill_value ]
    logger.info(f"merge: {len(table_unique)} unique obj in in overlap")
    return table_unique, table_groups


