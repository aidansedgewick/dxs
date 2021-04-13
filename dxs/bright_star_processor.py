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
    def from_file(cls, csv_path, queries=None, config_path=None, **kwargs):
        logger.info(f"reading from {csv_path.name}")
        star_table = Table.read(csv_path, **kwargs)
        if queries is not None:
            star_table = Query(*queries).filter(star_table)
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

def stack_bright_stars(
    self, table: Table, stacked_image_path, 
    mag_col, ra_col="ra", dec_col="dec", id_col=None, mosaic_paths=None, 
    cutout_size=(500, 500), save_cutouts=False, 
    imshow_kwargs=None, savefig_kwargs=None
):
    imshow_kwargs = imshow_kwargs or {}
    savefig_kwargs = savefig_kwargs or {}

    consecutives = [sum(1 for _ in group) for _, group in groupby(mosaic_paths)]
    if len(consecutives) > 20:
        print("It's better to sort your inputs such that the objects from the same mosaic are sorted.")

    stacked_image_path = Path(stacked_image_path)

    cutouts_dir = paths.temp_data_path / "cutouts"
    cutouts_dir.mkdir(exist_ok=True, parents=True)

    coords = SkyCoord(ra=table[ra_col], dec=table[dec_col], unit="degree")
    cutout_paths = []
    data_list = []

    average_ra = np.average(coords.ra.degree)
    average_dec = np.average(coords.dec.degree)

    data = 0 
    old_mosaic_path = 0
    for ii, (coord, mag, mosaic_path) in zip(coords, table[mag_col], mosaic_paths):
        if mosaic_path != old_mosaic_path:
            del data
            with fits.open(mosaic_path) as f:
                data = f[0].data.copy()
                wcs = WCS(f[0].header)
        old_mosaic_path = mosaic_path
        cutout = Cutout2D(
            data, position=coord, size=cutout_size, wcs=wcs, mode="partial"
        )
        if save_cutouts or ii == 0: # we need a header anyway.
            header = cutout.wcs.to_header()
            header["CRVAL1"] = average_ra
            header["CRVAL2"] = average_dec
            header["CRPIX1"] = cutout_size[0] / 2.
            header["CRPIX2"] = cutout_size[1] / 2.

        if save_cutouts:
            hdu = fits.PrimaryHDU(data=cutout.data.copy(), header=header)
            cutout_path = cutouts_dir / f"{coord.ra.degree*100:05d}_{abs(coord.ra.degree)*100:04d}_cutout.fits"
            hdu.writeto(cutout_path, overwrite=True)
            cutout_paths.append(cutout_path)
            # astropy garbage collection is not good...
            del hdu
        del cutout
        data_list.append(cutout.data)
    gc.collect()

    cube = np.dstack(data_list)
    output = np.nanmedian(cube, axis=2)

    hdu = fits.PrimaryHDU(data=output, header=header)
    hdu.writeto(stacked_image_path, overwrite=True)

    fig, ax = plt.subplots()
    ax.imshow(output, **imshow_kwargs)
    png_path = stacked_image_path.with_suffix(".png")
    fig.savefig(png_path, **savefig_kwargs)
    plt.close()
    

if __name__ == "__main__":
    tmass_catalog_path = (
        paths.input_data_path / "external/tmass/tmass_SA22_stars.csv"
    )
    star_table = Table.read(tmass_catalog_path, format="ascii")

    bsp = BrightStarProcessor(star_table)
    region_list = bsp.process_regions("k_m")





