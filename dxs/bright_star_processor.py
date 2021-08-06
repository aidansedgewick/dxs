import gc
import logging
import yaml
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS

from easyquery import Query
from regions import CircleSkyRegion, RectangleSkyRegion, write_ds9

from dxs import paths

logger = logging.getLogger("bright_stars")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

pix_scale = survey_config["mosaics"]["pixel_scale"]

class BrightStarProcessor:
    """
    parameters are read from ./configuration/survey_config.yaml
    """

    def __init__(self, star_table: Table, config_path=None):
        if not isinstance(star_table, Table):
            msg1 = "star_table should be an astropy Table.\n"
            msg2 = "Or, you can use bsp = BrightStarProcessor.from_file(<catpath>, **kwargs)"
            msg3 = " where **kwargs are passed to astropy.table.Table.read()"
            raise ValueError(msg1 + msg2 + msg3)
        self.star_table = star_table
        
        if config_path is None:            
            bright_star_config = survey_config["bright_stars"]
        else:
            with open(config_path, "r") as f:
                bright_star_config = yaml.load(f, Loader=yaml.FullLoader)
        self.box_angles = bright_star_config["diffraction_spike_angles"] * u.degree
        coeffs = bright_star_config["region_coefficients"]
        self.mag_ranges = np.array([-20.] + coeffs["mag_ranges"])
        self.box_widths = np.array(coeffs["box_widths"]) * (pix_scale / 3600. * u.degree)
        self.box_heights = np.array(coeffs["box_heights"]) * (pix_scale / 3600. * u.degree)
        self.circle_radii = np.array(coeffs["circle_radii"]) * (pix_scale / 3600. * u.degree)

        array_sizes = [len(x) for x in [self.box_widths, self.box_heights, self.circle_radii]]
        if not all(x == len(self.mag_ranges)-1 for x in array_sizes):
            lmr = len(self.mag_ranges)
            raise ValueError(
                f"each len(box_widths), len(box_heights), len(circle_radii) = {array_sizes} "
                f"should equal len(mag_ranges)-1 = {lmr-1}"
            )

    @classmethod
    def from_file(cls, csv_path, config_path=None, query: Query = None, **kwargs):
        csv_path = Path(csv_path)
        logger.info(f"reading from {csv_path.name}")
        star_table = Table.read(csv_path, **kwargs)
        if query is not None:
            lst = len(star_table)
            star_table = query.filter(star_table)
            logger.info(f"remove {lst-len(star_table)} objects from star_table")
        return cls(star_table, config_path=config_path)

    def process_region_masks(self, mag_col, ra_col="ra", dec_col="dec", ncpus=None):
        """
        which column to look at?
        """
        mag_min, mag_max = self.mag_ranges[0], self.mag_ranges[-1]
        queries = (f"{mag_min} < {mag_col}", f"{mag_col} < {mag_max}")
        table = Query(*queries).filter(self.star_table)
        mag_indices = np.digitize(table[mag_col], bins=self.mag_ranges)
        assert all(mag_indices < len(self.mag_ranges)) # there are no stars outside the min/max
        assert all(mag_indices > 0) # there are none less than the min_mag
        mag_indices = mag_indices - 1 # digitize assumes that [-inf, bins[0]] is the 0th bin.
        coords = SkyCoord(ra=table[ra_col], dec=table[dec_col], unit="degree")
        
        widths = self.box_widths[mag_indices]
        heights = self.box_heights[mag_indices]
        radii = self.circle_radii[mag_indices]
        if ncpus is None:
            region_list = []
            for coord, width, height, radius in zip(coords, widths, heights, radii):
                region_list.extend(
                    self.get_regions_for_star(coord, width, height, radius)
                )
        else:            
            # Pool is overkill - creating the shapes is much faster than expected...
            args = [x for x in zip(coords, widths, heights, radii)]
            with Pool(ncpus) as pool:
                nested_list = list(
                    pool.map(self._get_regions_wrapper, args)
                )
            region_list = [x for l in nested_list for x in l]
        self.region_list = region_list
        self.comment = "region_list from {mag_col}"
        return region_list

    def get_regions_for_star(self, coord, width, height, radius):
        star_regions = []
        for angle in self.box_angles: 
            sky_region = RectangleSkyRegion(
                center=coord, width=width, height=height, angle=angle
            )
            star_regions.append(sky_region)
        sky_region = CircleSkyRegion(center=coord, radius=radius)
        star_regions.append(sky_region)
        return star_regions

    def _get_regions_wrapper(self, arg):
        return self.get_regions_for_star(*arg)

    def write_region_list(self, output_path, region_list=None):
        if region_list is None:
            logger.info(f"using {self.comment}") # comment is so we know what's writing!
            region_list = self.region_list
        write_ds9(output_path, region_list)

if __name__ == "__main__":
    tmass_catalog_path = (
        paths.input_data_path / "external/tmass/tmass_SA22_stars.csv"
    )
    star_table = Table.read(tmass_catalog_path, format="ascii")

    bsp = BrightStarProcessor(star_table)
    region_list = bsp.process_regions("k_m")





