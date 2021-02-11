import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table

from easyquery import Query

from dxs.utils.misc import calc_mids
from dxs import paths

logger = logging.getLogger("quick_plotter")

class QPlot:
    def __init__(self, plot_layout=(1,1), figsize=None):
        figure, axes = plt.subplots(*plot_layout, figsize=None)
        self.figure = figure
        self.axes = axes

    def plot_number_density(
        self, column, selection=None, ax=None,
        survey_area=1.0, bins=None, step=0.5, **kwargs
    ):
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
    
        if bins is None:
            min_value = np.min(selection[column])
            max_value = np.max(selection[column])
            bins = np.arange( np.floor(min_value), np.ceil(max_value) + step, step)

        number_density, bin_edges = np.histogram(selection[column], bins=bins)
        number_density = number_density / survey_area
        bin_mids = calc_mids(bin_edges)
        ax.plot(bin_mids, number_density, **kwargs)

    def scatter_from_csv(self, csv_path, x_data, y_data, ax=None, **kwargs):
        data = pd.read_csv(csv_path)
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
        ax.scatter(data[xcol], data[ycol], **kwargs)

    def color_magnitude(self, c1, c2, mag, selection=None, ax=None, **kwargs):
        """ EN-US spelling for mpl consistency"""
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
            
        ydata = selection[c1] - selection[c2]
        xdata = selection[mag]
        ax.scatter(xdata, ydata, **kwargs)

    def color_color(self, c_x1, c_x2, c_y1, c_y2, selection=None, ax=None, **kwargs):
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
        ydata = selection[c_x1] - selection[c_x2]
        xdata = selection[c_y1] - selection[c_y2]

        self.axes.scatter(xdata, ydata, **kwargs)

    def plot_from_csv(self, csv_path, x_data, y_data, ax=None, **kwargs):
        data = pd.read_csv(csv_path)
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
        ax.scatter(data[xcol], data[ycol], **kwargs)

    def plot_positions(
        self, xpos, ypos, ax=None, xlim=None, ylim=None, **kwargs
    ):
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
        ax.scatter(xpos, ypos, **kwargs)

    def plot_coordinates(
        self, ra, dec, selection=None, ax=None, xlim=None, ylim=None, **kwargs
    ):
        self.plot_positions(
            selection[ra], 
            selection[dec], 
            ax=ax, xlim=None, ylim=None, **kwargs
        )
        if selection is None:
            raise ValueError("must provide selection")
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
        ax.set_xlim(ax.get_xlim()[::-1])

class QuickPlotter:

    def __init__(self):
        self.plot_list = []
        pass

    @classmethod
    def from_fits(cls, catalog_path, *queries):
        qp = cls()
        qp.selection_from_fits(catalog_path, "full_catalog", *queries)
        return qp

    def selection_from_fits(self, catalog_path, name: str, *queries):
        catalog_path = Path(catalog_path)
        logger.info(f"read from {catalog_path.stem}")
        catalog = Table.read(catalog_path)
        self.create_selection(name, *queries, catalog=catalog)
        #setattr(self, name, catalog)

    def create_selection(self, name: str, *queries: Tuple[str], catalog: Table = None):
        
        if catalog is None:
            catalog = self.catalog
        elif isinstance(catalog, str):
            catalog = self.get_selection(catalog)
        selection = Query(*queries).filter(catalog)
        if len(selection) == 0:
            logger.warn(f"create_selection - len selection {name} is zero")
        else:
            logger.info(f"{name}: {len(selection)} objects")
        setattr(self, name, selection)

    def get_selection(self, name):
        return getattr(self, name)

    def del_selection(self, name):
        delattr(self, name)

    def selection_in_coverage(self, ra, dec, image_list, selection=None):
        selection

    def remove_crosstalks(self, crosstalk_query=None, name="catalog", catalog=None):
        if crosstalk_query is None:
            q_J = Query("J_crosstalk_flag > 0")
            q_K = Query("K_crosstalk_flag > 0")
            crosstalk_query = (~q_J | ~q_K)
        if catalog is None:
            catalog = self.full_catalog
        selection = crosstalk_query.filter(catalog)
        logger.info(f"Discard {len(catalog)-len(selection)} crosstalks")
        setattr(self, name, selection)

    def remove_bad_magnitudes(
        self, bands, mag_limit=30.0, catalog=None, name="catalog", ap="auto"
    ):
        if catalog is None:
            catalog = self.catalog
        elif isinstance(catalog, str):
            catalog = self.get_selection(catalog)
        queries = []
        if not isinstance(bands, list):
            bands = [bands]
        for band in bands:
            if f"{band}_mag_{ap}" in catalog.colnames:
                queries.append(f"(J_mag_{ap} < {mag_limit})")
        queries = tuple(queries)
        
        selection = Query(*queries).filter(catalog)
        logger.info(f"Discard {len(catalog)-len(selection)} objects mag>{mag_limit:.2f}")
        setattr(self, name, selection)


    def create_plot(self, name: str, plot_layout=(1,1), figsize=None):
        plot = QPlot(plot_layout=plot_layout, figsize=figsize)
        self.plot_list.append(name)
        setattr(self, name, plot)

    def save_all_plots(self, save_dir=None, extension=".png", prefix="", **kwargs):

        save_dir = save_dir or Path.cwd()
        save_dir = Path(save_dir)
        plot_paths = []
        for plot_name in self.plot_list:
            plot = getattr(self, plot_name)
            plot_path = save_dir / f"{prefix}{plot_name}{extension}"
            plot.fig.savefig(plot_path, **kwargs)
        try:
            print_dir = save_dir.relative_to(Path.cwd())
        except:
            print_dir = save_dir

        print_path = print_dir / f"*{extension}"
        plot_viewers = {".png": "eog", ".pdf": "evince"}
        viewer = plot_viewers[extension]
        command = "view plots with \n" + f"   {viewer} {print_path}"
        print(command)






