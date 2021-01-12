from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table

from easyquery import Query

from dxs.utils.misc import calc_mids
from dxs import paths

class Plot:
    def __init__(plot_layout=(1,1), figsize=None):
        fig, axes = plt.subplots(layout, figsize=None)
        self.fig = fig
        self.axes = axes

    def plot_number_density(
        self, column, selection=None, 
        survey_area=1.0, bins=None, step=0.5, **kwargs
    ):
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
    
        if bins is None:
            min_val = np.min(selection[column])
            max_val = np.max(selection[column])
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

    def plot_from_csv(self, csv_path, x_data, y_data, ax=None, **kwargs):
        data = pd.read_csv(csv_path)
        if ax is not None:
            ax = self.axes[ax]
        else:
            ax = self.axes
        ax.scatter(data[xcol], data[ycol], **kwargs)

    def plot_positions(
        self, column, xpos, ypos, selection=None,
    ):
        pass

class QuickPlotter:

    def __init__(self, catalog: Table):
        self.catalog = catalog

    @classmethod
    def from_fits(cls, catalog_path):
        catalog = Table.read(catalog_path)
        return cls(catalog)

    def catalog_from_fits(catalog_path, name: str):
        catalog_path = Path(catalog_path)
        catalog = Table.read(catalog_path)
        setattr(self, name, catalog)

    def create_selection(self, name: str, queries: Tuple[str], catalog: Table = None):
        if catalog is None:
            catalog = self.catalog
        selection = Query(*queries).filter(catalog)
        setattr(self, name, selection)

    def create_plot(self, name: str, plot_layout=(1,1), figsize=None):
        plot = Plot(plot_layout=plot_layout, figsize=figsize)
        settattr(self, name, plot)







