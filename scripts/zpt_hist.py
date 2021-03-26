import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dxs.utils.misc import calc_mids

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

data = pd.read_csv(paths.header_data_path)

magzpt_bins = np.arange(23.7, 24.3, 0.02)
magzpt_mids = calc_mids(magzpt_bins)

fig, axes = plt.subplots(12,2, figsize=(6, 8))

field = "SA"
band = "K"

N_tiles = survey_config["tiles_per_field"][field]
tiles = [x for x in range(1, N_tiles+1)]

magzpt_cols = [f"magzpt_{ccd}" for ccd in survey_config["ccds"] ]

dat_subset = data.query(f"field=='{field}' & band=='{band}'")
full_magzpt_data = dat_subset[ magzpt_cols ].values.flatten()
full_magzpt_hist, _ = np.histogram(full_magzpt_data, bins=magzpt_bins)
full_magzpt_hist = full_magzpt_hist / len(full_magzpt_data)

fig2,ax2 = plt.subplots()
ax2.plot(magzpt_mids, full_magzpt_hist)

for ii, tile in enumerate(tiles):
    tile_dat = data.query(f"field=='{field}' & band=='{band}' & tile=='{tile}'")
    
    magzpt_data = tile_dat[ magzpt_cols ].values.flatten()
    magzpt_median = np.median(magzpt_data)

    scaling = 10**(-0.4*(magzpt_data - magzpt_median))
    print(scaling)

    hist, _  = np.histogram(magzpt_data, bins=magzpt_bins)
    hist = hist / len(magzpt_data)
    axes[ii, 0].plot(magzpt_mids, hist, color="k", ls="--")
    axes[ii, 0].plot(magzpt_mids, full_magzpt_hist)
    ax2.plot(magzpt_mids, hist, color="k", alpha=0.3)
    if ii != len(axes)-1:
        axes[ii, 0].set_xticks([])
        axes[ii, 1].set_xticks([])
    axes[ii, 0].set_yticks([])
    axes[ii, 1].set_yticks([])

    

fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
