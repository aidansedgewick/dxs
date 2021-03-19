import os
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

from astropy.table import Table

from easyquery import Query

from dxs.utils.poly import Poly

from dxs import paths

suffix = "panstarrs"

field = "SA"

catalog_path = paths.get_catalog_path(field, 00, "", prefix="sm", suffix=f"_{suffix}")
catalog = Table.read(catalog_path)

queries = (
    #f"i_mag_kron < 18.0", 
    #f"i_mag_kron > 16.0", 
    f"K_stell > 0.9",
    f"i_mag_aper_30 < 50", 
    f"r_mag_aper_30 < 50",
    f"J_mag_aper_30 < 50",
    f"K_mag_aper_30 < 50",
    f"i_mag_aper_30 - K_mag_aper_30 < 4.0",
    f"J_mag_aper_30 - K_mag_aper_30 < 1.0", 
)
stars = Query(*queries).filter(catalog)

def kernel_isocontour_integral(kernel, contour_level, low_bounds, high_bounds):
    kernel_copy = deepcopy(kernel)
    
    def _evaluate_patch(self, points):
        print("using patch")
        x = kernel(points)
        x[ x < contour_level ] = 0.
        return x

    kernel_copy.evaluate = _evaluate_patch.__get__(kernel_copy, gaussian_kde)
    kernel_copy.__call__ = _evaluate_patch.__get__(kernel_copy, gaussian_kde)
    kernel_copy.pdf = _evaluate_patch.__get__(kernel_copy, gaussian_kde)

    #print(kernel_copy.__call__)
    #print(kernel_copy.evaluate)
    N = 10000

    points = np.vstack([np.random.uniform(l, h, N) for l, h in zip(low_bounds, high_bounds)])
    
    samples = kernel_copy(points)
    norm = np.prod([h-l for l, h in zip(low_bounds, high_bounds)])
    integral = np.sum(samples) / (N* norm)
    
    print(integral)
    return integral
    


limits = np.arange(16.0, 22.5, 0.5) #[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]

dx = 0.01
xmin, xmax = -0.25, 2.25
ymin, ymax = -0.25, 1.25

x, y = np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dx, dx)
xpoints = [xmin, xmax, xmax, xmin, xmin]
ypoints = [ymin, ymin, ymax, ymax, ymin]
xx, yy = np.meshgrid(x, y)
ravel_grid = np.vstack([yy.ravel(), xx.ravel()])

fig1, axes1 = plt.subplots(2, len(limits)-1, figsize=(12, 6))

fig2, ax2 = plt.subplots()

zmaxes_x = np.zeros(len(limits) - 1)
zmaxes_y = np.zeros(len(limits) - 1)

for ii, (low, high) in enumerate(zip(limits[:-1], limits[1:])):
    stars_ii = Query(f"{low} < i_mag_kron", f"i_mag_kron < {high}").filter(stars)

    ydat = stars_ii["J_mag_aper_30"] - stars_ii["K_mag_aper_30"]
    xdat = stars_ii["r_mag_aper_30"] - stars_ii["i_mag_aper_30"]
    mask = (ymin < ydat) & (ydat < ymax) & (xmin < xdat) & (xdat < xmax)
    ydat = ydat[ mask ]
    xdat = xdat[ mask ]

    data = np.vstack([ydat, xdat])

    kernel = gaussian_kde(data, bw_method=0.15)
    print(kernel.integrate_box([ymin, xmin], [ymax, xmax]))
    z = np.reshape(kernel(ravel_grid).T, xx.shape)
    zmax = z.max()
    max_index = np.unravel_index(np.argmax(z), z.shape)
    print(max_index)
    zmax_x, zmax_y = x[max_index[1]], y[max_index[0]] # np array is row indexed
    zmaxes_x[ii] = zmax_x
    zmaxes_y[ii] = zmax_y

    #im = axes[0, ii].pcolormesh(x, y, z, vmin=0, vmax=25)
    axes1[0, ii].imshow(z.T)
    axes1[1, ii].scatter(xdat, ydat, s=1, color="k", alpha=0.1)
    axes1[1, ii].plot(xpoints, ypoints, color="r", lw=3)
    low_bounds, high_bounds = [ymin, xmin], [ymax, xmax]
    N_points = 250_000

    points = np.vstack([
        np.random.uniform(l, h, N_points) for l, h in zip(low_bounds, high_bounds)
    ])
    norm = np.prod([h-l for l, h in zip(low_bounds, high_bounds)])
    samples = kernel(points)
    print(f"generated {N_points} samples")

    integral_target = 0.90


    lmin, lmax = 0, zmax
    for jj in range(2):
        prev_integral = np.inf
        prev_level = 0.0
        samples_jj = samples.copy()
        print(f"check integral between {lmin:.2f}, {lmax:.2f}")
        for level in np.linspace(lmin, lmax, 50):
            samples_jj = samples_jj[ samples_jj > level ]
            integral = norm * np.sum(samples_jj) / (N_points)
            if integral < integral_target and prev_integral > integral_target:
                lmin, lmax = prev_level, level
                print(integral, level, "BREAK")
                break

            prev_integral = integral
            prev_level = level

    print(level, integral)

    ax2.scatter(zmax_x, zmax_y, color=f"C{ii%10}", marker="x", s=40, zorder=5)
    ax2.contour(xx, yy, z, levels=[level], colors=f"C{ii%10}")
ax2.set_xlabel("r-i")
ax2.set_ylabel("J-K")        
ax2.plot(zmaxes_x, zmaxes_y, color="k")
plt.show()



















