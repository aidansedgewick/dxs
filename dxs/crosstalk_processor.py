import json
import time
import yaml
from pathlib import Path

import numpy as np

import astropy.io.ascii as ascii
import astropy.io.fits as fits
from astropy import wcs
from astropy.table import Table, Column, vstack, join

from dxs.mosaic_builder import get_stack_data
from dxs.catalog_builder import get_catalog_dir, get_catalog_stem
from dxs.pystilts import Stilts
from dxs.utils.table import table_to_numpynd, fix_column_names

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

import matplotlib.pyplot as plt

class CrosstalkProcessor:
    def __init__(
        self, stack_list, star_catalog: Table, 
        max_order=8, crosstalk_separation=256
    ):
        self.stack_list = stack_list
        self.star_catalog = star_catalog
        self.star_catalog.add_column(np.arange(len(star_catalog)), name="Xid")
        self.crosstalk_orders = np.concatenate( 
            [np.arange(-max_order, 0), np.arange(1, max_order+1)]
        )
        self.crosstalk_separation = crosstalk_separation

    @classmethod
    def from_dxs_spec(cls, field, tile, band, star_catalog: Table):
        stack_data = get_stack_data(field, tile, band)
        stack_list = [paths.stack_data_path / f"{x}.fit" for x in stack_data["filename"]]
        catalog_dir = get_catalog_dir(field, tile, band)
        return cls(stack_list, star_catalog)

    def collate_crosstalks(
        self, mag_column=None, mag_limit=15.0, ra="ra", dec="dec", save_path=None
    ):
        crosstalk_table_list = []
        for ii, stack_path in enumerate(self.stack_list):
            print(ii, len(self.stack_list))
            stack_crosstalks = self.get_crosstalks_in_stack(
                stack_path, mag_column=mag_column, mag_limit=mag_limit, ra="ra", dec="dec"
            )
            crosstalk_table_list.append(stack_crosstalks)
        all_crosstalks = vstack(crosstalk_table_list, join_type="exact")
        grouped = all_crosstalks.group_by(["Xid", "crosstalk_direction", "crosstalk_order"])
        crosstalks = grouped.groups.aggregate(np.mean)
        crosstalks
        if save_path is not None:
            self.crosstalk_catalog_path = save_path
            crosstalks.write(save_path, format="fits", overwrite=True)
        else:
            return crosstalks
            
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
        for ii, order in enumerate(self.crosstalk_orders):
            # Quadrants whose crosstalks scatter in x; start with top_right
            top_right = stars_in_frame[ top & right ][ ["Xid", "xpix", "ypix"] ]
            xpix = top_right["xpix"] + order*self.crosstalk_separation
            top_right["xpix"] = xpix
            top_right = top_right[ (xmid < xpix) & (xpix < xlen) ]
            # Bottom left
            bottom_left = stars_in_frame[ bottom & left ][ ["id", "xpix", "ypix"] ]
            xpix = bottom_left["xpix"] + order*self.crosstalk_separation
            bottom_left["xpix"] = xpix
            bottom_left = bottom_left[ (0 < xpix) & (xpix < xmid) ]
            
            # Quadrants whose crosstalks scatter in y direction; start top left
            top_left = stars_in_frame[ top & left ][ ["id", "xpix", "ypix"] ]
            ypix = top_left["ypix"] + order*self.crosstalk_separation
            top_left["ypix"] = ypix
            top_left = top_left[ (ymid < ypix) & (ypix < ylen) ]
            # finally bottom right
            bottom_right = stars_in_frame[ bottom & right ][ ["id", "xpix", "ypix"] ]
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
        self, catalog_path,  ra=None, dec=None, crosstalk_catalog_path=None
    ):
        if crosstalk_catalog_path is None:
            crosstalk_catalog_path = self.crosstalk_catalog_path
        temporary_catalog = paths.temp_data_path / "temp_stilts_{int(time.time())}.cat"
        stilts = Stilts.tskymatch2_fits(
            catalog_path, 
            crosstalk_catalog_path,
            temporary_catalog,
            ra1=ra, 
            dec1=dec, 
            ra2="crosstalk_ra", 
            dec2="crosstalk_dec"
        )
        stilts.run()
        input_columns = ["GroupID", "GroupSize", "Separation"]
        output_columns = ["crosstalk_group_id", "crosstalk_group_size", "crosstalk_separation"]
        fix_column_names(
            temporary_catalog_path, outpath=catalog_path, 
            input_columns=input_columns, output_columns=output_columns
        )

    def flag_crosstalks_in_catalog(self, catalog_path, coeffs=None):
        raise NotImplementedError
        

if __name__ == "__main__":


    star_table_path = (
        paths.input_data_path / "external_catalogs/tmass/tmass_ElaisN1_stars.csv"
    )
    star_catalog = Table.read(star_table_path, format="ascii")
    star_catalog = star_catalog[ star_catalog["k_m"] < 12.0 ]
    processor = CrosstalkProcessor.from_dxs_spec("EN", 4, "K", star_catalog=star_catalog)
    crosstalks = processor.collate_crosstalks(mag_column="k_m", mag_limit=12.0)
    print(crosstalks)

    plt.scatter(star_catalog["ra"], star_catalog["dec"], color="k", marker="x", s=8)
    plt.scatter(crosstalks["ra"], crosstalks["dec"], c=abs(crosstalks["order"]), s=4)
    plt.show()













