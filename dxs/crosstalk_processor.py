import json
import logging
import time
import yaml
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm

import astropy.io.ascii as ascii
import astropy.io.fits as fits
from astropy import wcs
from astropy.table import Table, Column, vstack, join

from dxs.mosaic_builder import get_stack_data
from dxs.pystilts import Stilts
from dxs.utils.table import table_to_numpynd, fix_column_names

from dxs import paths

logger = logging.getLogger("crosstalks")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

import matplotlib.pyplot as plt

class CrosstalkProcessor:
    def __init__(
        self, stack_list, star_catalog: Table, crosstalk_catalog_path=None, 
        max_order=8, crosstalk_separation=256, n_cpus=None
    ):
        # TODO URGENT: does not correctly find all crosstalks...!
        self.stack_list = stack_list
        self.star_catalog = star_catalog
        self.star_catalog.add_column(np.arange(len(star_catalog)), name="parent_id")
        if crosstalk_catalog_path is None:
            crosstalk_dir = paths.temp_data_path / "crosstalks"
            crosstalk_dir.mkdir(exist_ok=True, parents=True)
            crosstalk_catalog_path = crosstalk_dir / f"crosstalks_{int(time.time())}.cat"
        self.crosstalk_catalog_path = Path(crosstalk_catalog_path)
        self.crosstalk_orders = np.concatenate( 
            [np.arange(-max_order, 0), np.arange(1, max_order+1)]
        )
        self.crosstalk_separation = crosstalk_separation
        self.n_cpus = n_cpus

    @classmethod
    def from_dxs_spec(
        cls, field, tile, band, star_catalog: Table, crosstalk_catalog_path=None, n_cpus=None
    ):
        stack_data = get_stack_data(field, tile, band)
        stack_list = [paths.stack_data_path / f"{x}.fit" for x in stack_data["filename"]]
        if crosstalk_catalog_path is None:
            crosstalk_catalog_dir = paths.get_catalog_dir(field, tile, band)
            stem = paths.get_catalog_stem(field, tile, band)
            crosstalk_catalog_path = crosstalk_catalog_dir / f"{stem}_crosstalks.fits"
        return cls(
            stack_list, 
            star_catalog, 
            crosstalk_catalog_path=crosstalk_catalog_path,
            n_cpus=n_cpus,
        )

    def collate_crosstalks(
        self, mag_column, mag_limit=15.0, ra="ra", dec="dec", save_path=None
    ):
        crosstalk_table_list = []
        logger.info(f"collate crosstalks from {len(self.stack_list)} stacks")
        if self.n_cpus is None:
            for ii, stack_path in tqdm.tqdm(enumerate(self.stack_list)):
                stack_crosstalks = self.get_crosstalks_in_stack(
                    stack_path, mag_column=mag_column, mag_limit=mag_limit, ra="ra", dec="dec"
                )
                crosstalk_table_list.append(stack_crosstalks)
        else:
            kwargs = {
                "mag_column": mag_column, "mag_limit": mag_limit, "ra": ra, "dec": dec,
            }
            arg_list = [(stack_path, kwargs) for stack_path in self.stack_list]
            with Pool(self.n_cpus) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.map(self._crosstalks_in_stack_wrapper, arg_list),
                        total=len(self.stack_list)
                    )    
                )
                crosstalk_table_list = [t for result in results for t in result]
        all_crosstalks = vstack(crosstalk_table_list, join_type="exact")
        grouped = all_crosstalks.group_by(
            ["parent_id", "crosstalk_direction", "crosstalk_order"]
        )
        crosstalk_locations = grouped.groups.aggregate(np.mean)
        star_data_copy = self.star_catalog[["parent_id", ra, dec, mag_column]].copy()
        crosstalks = join(
            crosstalk_locations, star_data_copy, keys="parent_id", join_type="left"
        )
        crosstalks.rename_column(ra, f"parent_ra")
        crosstalks.rename_column(dec, f"parent_dec")
        crosstalks.rename_column(mag_column, "parent_mag")
        crosstalks.write(self.crosstalk_catalog_path, overwrite=True)
        return crosstalks

    def _crosstalks_in_stack_wrapper(self, arg):
        args, kwargs = arg
        return self.get_crosstalks_in_stack(args, **kwargs)    
            
    def get_crosstalks_in_stack(
        self, stack_path, mag_column=None, mag_limit=15.0, ra="ra", dec="dec"
    ):
        crosstalk_table_list = []
        with fits.open(stack_path) as f:
            for ii, ccd in enumerate(survey_config["ccds"]):
                header = f[ccd].header
                fwcs = wcs.WCS(header)
                xlen = header["NAXIS1"]
                ylen = header["NAXIS2"]
                stars_in_frame = self.stars_in_frame_from_wcs(
                    fwcs, star_table=self.star_catalog.copy(), xlen=xlen, ylen=ylen,
                    mag_column=mag_column, mag_limit=mag_limit, ra=ra, dec=dec
                )
                frame_crosstalks = self.get_crosstalk_pixels(
                    stars_in_frame, xlen=xlen, ylen=ylen
                )
                crosstalk_pixels = table_to_numpynd(frame_crosstalks[ ["xpix", "ypix"] ])
                crosstalk_locations = fwcs.wcs_pix2world(crosstalk_pixels, 1)

                frame_crosstalks["crosstalk_ra"] = crosstalk_locations[:,0]
                frame_crosstalks["crosstalk_dec"] = crosstalk_locations[:,1]
                frame_crosstalks.remove_columns(["xpix", "ypix"])
                crosstalk_table_list.append(frame_crosstalks)
        stack_crosstalks = vstack(crosstalk_table_list, join_type="exact")
        return stack_crosstalks

    def stars_in_frame_from_wcs(
        self, fwcs, star_table, xlen=4096, ylen=4096, 
        mag_column=None, mag_limit=15.0, ra="ra", dec="dec"
    ):
        if mag_column is not None:
            star_table = star_table[ star_table[mag_column] < mag_limit ]
        star_locations = table_to_numpynd(star_table[[ra, dec]])
        star_pix = fwcs.wcs_world2pix(star_locations, 1)

        star_table["xpix"] = star_pix[:,0]
        star_table["ypix"] = star_pix[:,1]
        xmask = (0 < star_pix[:,0]) & (star_pix[:,0] < xlen)
        ymask = (0 < star_pix[:,1]) & (star_pix[:,1] < ylen)
        star_table = star_table[ xmask & ymask ]

        return star_table

    def get_crosstalk_pixels(self, stars_in_frame, xlen=4096, ylen=4096):
        xmid = xlen // 2
        ymid = ylen // 2
        bottom = (stars_in_frame["ypix"] < ymid)
        top = (stars_in_frame["ypix"] >= ymid)
        left = (stars_in_frame["xpix"] < xmid)
        right = (stars_in_frame["xpix"] >= xmid)
       
        table_list = []
        column_selection = ["parent_id", "xpix", "ypix"]
        for ii, order in enumerate(self.crosstalk_orders):
            # Quadrants whose crosstalks scatter in x; start with top_right
            top_right = stars_in_frame[ top & right ][ column_selection ]
            xpix = top_right["xpix"] + order*self.crosstalk_separation
            top_right["xpix"] = xpix
            top_right = top_right[ (xmid < xpix) & (xpix < xlen) ]
            # Bottom left
            bottom_left = stars_in_frame[ bottom & left ][ column_selection ]
            xpix = bottom_left["xpix"] + order*self.crosstalk_separation
            bottom_left["xpix"] = xpix
            bottom_left = bottom_left[ (0 < xpix) & (xpix < xmid) ]
            
            # Quadrants whose crosstalks scatter in y direction; start top left
            top_left = stars_in_frame[ top & left ][ column_selection ]
            ypix = top_left["ypix"] + order*self.crosstalk_separation
            top_left["ypix"] = ypix
            top_left = top_left[ (ymid < ypix) & (ypix < ylen) ]
            # finally bottom right
            bottom_right = stars_in_frame[ bottom & right ][ column_selection ]
            ypix = bottom_right["ypix"] + order*self.crosstalk_separation
            bottom_right["ypix"] = ypix
            bottom_right = bottom_right[ (0 < ypix) & (ypix < ymid) ]

            quadrants = [top_right, bottom_left, top_left, bottom_right]
            directions = ["x", "x", "y", "y"]
            for quad, direction in zip(quadrants, directions):
                if len(quad) > 0:
                    quad.add_column(order, name="crosstalk_order")
                    quad.add_column(direction, name="crosstalk_direction")
                    table_list.append(quad)
        crosstalk_locations = vstack(table_list, join_type="exact")
        return crosstalk_locations

    def match_crosstalks_to_catalog(
        self, catalog_path, crosstalk_catalog_path=None, output_path=None,
        ra=None, dec=None, error=1.0, band=None
    ):
        catalog_path = Path(catalog_path)
        if crosstalk_catalog_path is None:
            crosstalk_catalog_path = self.crosstalk_catalog_path
        crosstalk_catalog_path = Path(crosstalk_catalog_path)
        if output_path is None:
            output_path = catalog_path
        logger.info("match crosstalks {crosstalk_catalog_path.name} to {catalog_path.name}")
        stilts = Stilts.tskymatch2_fits(
            catalog_path,
            crosstalk_catalog_path,
            output_path=output_path,
            ra1=ra,
            dec1=dec,
            join="all1",
            find="best1",
            ra2="crosstalk_ra",
            dec2="crosstalk_dec",
            error=error
        )
        stilts.run()
        self.flag_crosstalks_in_catalog(output_path)

    def flag_crosstalks_in_catalog(self, catalog_path, coeffs=None):
        catalog = Table.read(catalog_path)
        crosstalk_flag = np.zeros(len(catalog))
        print(catalog.colnames)
        crosstalk_mask = abs(catalog["crosstalk_order"]) > 0
        crosstalk_flag[ crosstalk_mask ] = 1
        catalog.add_column(crosstalk_flag, name="crosstalk_flag")
        catalog.write(catalog_path, overwrite=True)



if __name__ == "__main__":


    star_table_path = (
        paths.input_data_path / "external/tmass/tmass_ElaisN1_stars.csv"
    )
    star_catalog = Table.read(star_table_path, format="ascii")
    star_catalog = star_catalog[ star_catalog["k_m"] < 12.0 ]
    processor = CrosstalkProcessor.from_dxs_spec("EN", 4, "K", star_catalog=star_catalog)
    crosstalks = processor.collate_crosstalks(mag_column="k_m", mag_limit=12.0)
    print(crosstalks)

    plt.scatter(star_catalog["ra"], star_catalog["dec"], color="k", marker="x", s=8)
    plt.scatter(crosstalks["ra"], crosstalks["dec"], c=abs(crosstalks["order"]), s=4)
    plt.show()













