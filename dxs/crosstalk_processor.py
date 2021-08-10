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

from stilts_wrapper import Stilts

from dxs import CatalogExtractor
from dxs.mosaic_builder import get_stack_data
from dxs.utils.image import scale_mosaic
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
        max_order=8, crosstalk_separation=256
    ):
        self.stack_list = stack_list
        self.star_catalog = star_catalog
        self.star_catalog.add_column(np.arange(len(star_catalog)), name="parent_id")
        if crosstalk_catalog_path is None:
            scratch_crosstalk_dir = paths.scratch_data_path / "crosstalks"
            scratch_crosstalk_dir.mkdir(exist_ok=True, parents=True)
            crosstalk_catalog_path = scratch_crosstalk_dir / f"crosstalks_{int(time.time())}.cat"
        self.crosstalk_catalog_path = Path(crosstalk_catalog_path)
        self.crosstalk_orders = np.concatenate( 
            [np.arange(-max_order, 0), np.arange(1, max_order+1)] # remember arange endpoints...
        )
        self.crosstalk_separation = crosstalk_separation

    @classmethod
    def from_dxs_spec(
        cls, field, tile, band, star_catalog: Table, crosstalk_catalog_path=None
    ):
        stack_data = get_stack_data(field, tile, band)
        stack_list = [paths.stack_data_path / f"{x}.fit" for x in stack_data["filename"]]
        if crosstalk_catalog_path is None:
            crosstalk_catalog_dir = paths.get_catalog_dir(field, tile, band) / "aux"
            crosstalk_catalog_dir.mkdir(exist_ok=True, parents=True)
            stem = paths.get_catalog_stem(field, tile, band)
            crosstalk_catalog_path = crosstalk_catalog_dir / f"{stem}_crosstalks.fits"
        return cls(
            stack_list, 
            star_catalog, 
            crosstalk_catalog_path=crosstalk_catalog_path,
        )

    def collate_crosstalks(
        self, mag_column, mag_limit=15.0, ra="ra", dec="dec", 
        save_path=None, n_cpus=None, ccds=None
    ):
        crosstalk_table_list = []
        logger.info(f"collate crosstalks from {len(self.stack_list)} stacks")
        if n_cpus is None:
            for ii, stack_path in tqdm.tqdm(enumerate(self.stack_list)):
                stack_crosstalks = self.get_crosstalks_in_stack(
                    stack_path, mag_column=mag_column, mag_limit=mag_limit, 
                    ra="ra", dec="dec", ccds=ccds,
                )
                crosstalk_table_list.append(stack_crosstalks)
        else:
            kwargs = {
                "mag_column": mag_column, 
                "mag_limit": mag_limit, 
                "ra": ra, 
                "dec": dec,
                "ccds": ccds,
            }
            arg_list = [(stack_path, kwargs) for stack_path in self.stack_list]
            with Pool(n_cpus) as pool:
                crosstalk_table_list = list(
                    tqdm.tqdm(
                        pool.map(self._crosstalks_in_stack_wrapper, arg_list),
                        total=len(self.stack_list)
                    )    
                )
                #crosstalk_table_list = [t for t in results]
        all_crosstalks = vstack(crosstalk_table_list, join_type="exact")
        grouped = all_crosstalks.group_by(
            ["crosstalk_direction", "parent_id", "crosstalk_order", "crosstalk_ccd", "crosstalk_pointing"]
        )
        crosstalk_locations = grouped.groups.aggregate(np.mean)
        star_data_copy = self.star_catalog[["parent_id", ra, dec, mag_column]].copy()
        crosstalks = join(
            crosstalk_locations, star_data_copy, keys="parent_id", join_type="left"
        )
        crosstalks.rename_column(ra, f"parent_ra")
        crosstalks.rename_column(dec, f"parent_dec")
        crosstalks.rename_column(mag_column, "parent_mag")
        crosstalks.remove_columns(["crosstalk_ccd", "crosstalk_pointing"])
        crosstalks.write(self.crosstalk_catalog_path, overwrite=True)
        return crosstalks

    def _crosstalks_in_stack_wrapper(self, arg):
        args, kwargs = arg
        return self.get_crosstalks_in_stack(args, **kwargs)    
            
    def get_crosstalks_in_stack(
        self, stack_path, mag_column=None, mag_limit=15.0, ra="ra", dec="dec", ccds=None
    ):
        if ccds is None:
            ccds = survey_config["ccds"]
        crosstalk_table_list = []
        with fits.open(stack_path) as f:
            pointing = f[0].header["OBJECT"].split()[3]
            for ii, ccd in enumerate(ccds):
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
                frame_crosstalks["crosstalk_ccd"] = np.full(len(crosstalk_pixels), ccd)
                frame_crosstalks["crosstalk_pointing"] = np.full(len(crosstalk_pixels), pointing)
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
        ra=None, dec=None, error=1.0, flag_value=1.0
    ):
        catalog_path = Path(catalog_path)
        if crosstalk_catalog_path is None:
            crosstalk_catalog_path = self.crosstalk_catalog_path
        crosstalk_catalog_path = Path(crosstalk_catalog_path)
        if output_path is None:
            output_path = catalog_path
        logger.info(f"match crosstalks {crosstalk_catalog_path.name} to {catalog_path.name}")
        stilts = Stilts.tskymatch2_fits(
            in1=catalog_path,
            in2=crosstalk_catalog_path,
            out=output_path,
            ra1=ra,
            dec1=dec,
            join="all1",
            find="best1",
            ra2="crosstalk_ra",
            dec2="crosstalk_dec",
            error=error,
            all_formats="fits"
        )
        stilts.run()
        self.flag_crosstalks_in_catalog(output_path, flag_value)

    """
    def extract_from_inverse_mosaic(self, mosaic_path, weight_path=None):
        mosaic_path = Path(mosaic_path)
        scratch_crosstalks_dir = paths.scratch_data_path / "crosstalks"
        scratch_crosstalks_dir.mkdir(exist_ok=True, parents=True)
        inv_mosaic_path = scratch_crosstalks_dir / f"inverse_{mosaic_path.stem}.fits"
        xtalk_catalog_path = scratch_crosstalks_dir / f"{mosaic_path.stem}_detections.fits"
        scale_mosaic(mosaic_path, value=-1.0, save_path=inv_mosaic_path)
        if weight_path is None:
            weight_path = mosaic_path.with_suffix(".weight.fits")
        extractor = CatalogExtractor(
            detection_mosaic_path=inv_mosaic_path, 
            weight_path=weight_path,
            catalog_path=xtalk_catalog_path,
            sextractor_config_file=paths.config_path / "sextractor/inv_xtalks.sex",
            sextractor_parameter_file=paths.config_path / "sextractor/inv_xtalks.param",
        )
        extractor.extract()
        os.remove(extractor.detection_mosaic_path)
        os.remove(extractor.segmentation_mosaic_path)
        return extractor.catalog_path
    """

    def flag_crosstalks_in_catalog(self, catalog_path, flag_value=1, coeffs=None):
        catalog = Table.read(catalog_path)
        crosstalk_flag = np.zeros(len(catalog))
        crosstalk_mask = abs(catalog["crosstalk_order"]) > 0
        crosstalk_flag[ crosstalk_mask ] = flag_value
        catalog.add_column(crosstalk_flag, name="crosstalk_flag")
        catalog.write(catalog_path, overwrite=True)



def calc_crosstalk_magnitude_coeffs(cat, band):
    """
    """
    raise NotImplementedError
    """
    fig,axes = plt.subplots(2,4, figsize=(10,6))
    axes = axes.flatten()

    crosstalks = cat[ cat[f"{band}_crosstalk_flag"] > 0 ]

    cat = cat[ cat[f"{band}_mag_auto"] < 50. ]

    for ii, order in enumerate(range(1,9)):
        co = cat[ abs(cat[f"{band}_crosstalk_order"]) == order ]
        col = np.sqrt(
            (co[f"{band}_crosstalk_ra"]-co[f"{band}_ra"])**2
            + (co[f"{band}_crosstalk_dec"]-co[f"{band}_dec"])**2
        )
        axes[ii].scatter(
            co[f"{band}_parent_mag"], co[f"{band}_mag_auto"], s=1, c=col
        )

        for yoff in [6,7,8,9,10,11,12]:
            axes[ii].plot((0,20), (yoff, 20+yoff), color="k", ls="--", alpha=0.2)
        axes[ii].set_xlim(6,13)
        axes[ii].set_ylim(14,22)
    return fig
    """











