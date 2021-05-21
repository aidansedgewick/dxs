import sys

import numpy as np
import matplotlib.pyplot as plt

import emcee

from astropy import cosmology
from astropy import units as u
from astropy.table import Table

from scipy.interpolate import interp1d

from easyquery import Query

from dxs.utils.image import objects_in_coverage, calc_survey_area
from dxs.utils.phot import vega_to_ab
from dxs.utils.misc import calc_mids
from dxs import paths

cosmol = cosmology.FlatLambdaCDM(H0=70.0, Om0=0.3, ) #Planck15

field = "LH"
suffix = "_panstarrs_zphot"

if "panstarrs" in suffix and "hsc" not in suffix:
    opt = "panstarrs"
elif "hsc" in suffix and "panstarrs" not in suffix:
    opt = "hsc"
elif "panstarrs" in suffix and "hsc" in suffix:
    raise ValueError(f"suffix {suffix} contains hsc AND panstarrs?!")
else:
    raise ValueError(f"suffix {suffix} has no optical data?")
optical_mask_path = paths.input_data_path / f"external/{opt}/masks/{field}_mask.fits"
print_path = optical_mask_path.relative_to(paths.base_path)
print(f"optical mask at\n    {print_path}")

J_mask_path = paths.masks_path / f"{field}_J_good_cov_mask.fits"
K_mask_path = paths.masks_path / f"{field}_K_good_cov_mask.fits"
opt_mask_list = [J_mask_path, K_mask_path, optical_mask_path]
nir_mask_list = [J_mask_path, K_mask_path]

cat_path = paths.get_catalog_path(field, 00, "", suffix=suffix)

full_catalog = Table.read(cat_path)

full_catalog["J_mag_aper_30"] = full_catalog["J_mag_aper_30"].filled(99) + 0.938
full_catalog["K_mag_aper_30"] = full_catalog["K_mag_aper_30"].filled(99) + 1.900
full_catalog["K_mag_auto"] = full_catalog["K_mag_auto"].filled(99) + 1.900

J_mask_path = paths.masks_path / f"{field}_J_good_cov_mask.fits"
K_mask_path = paths.masks_path / f"{field}_K_good_cov_mask.fits"
opt_mask_list = [J_mask_path, K_mask_path, optical_mask_path]
nir_mask_list = [J_mask_path, K_mask_path]

data_catalog = Query(
    "J_crosstalk_flag < 1", "K_crosstalk_flag < 1", 
    "J_coverage > 0", "K_coverage > 0"
).filter(full_catalog)

opt_catalog_mask = objects_in_coverage(
    opt_mask_list, data_catalog["ra_1"], data_catalog["dec_1"]
)
data_catalog = data_catalog[ opt_catalog_mask ]



opt_area = calc_survey_area(opt_mask_list, density=1e4)
print(f"optical area: {opt_area}")
"""
Kbins = np.arange(16, 25, 0.5)
Kmids = calc_mids(Kbins)
fig, ax = plt.subplots()
Khist, _ = np.histogram(full_cat["K_mag_auto"], bins=Kbins)
Khist = Khist / opt_area
ax.scatter(Kmids, Khist)
ax.semilogy()
ax.set_ylim(1, 1e4)"""

K_faint = 22.0# * u.mag
K_bright = 17.0

survey_frac = (opt_area * (np.pi / 180.) **2 ) / (4. * np.pi)

cat = Query(
    f"{K_bright} < K_mag_auto", f"K_mag_auto < {K_faint}", 
    f"0. < z_phot", f"z_phot < 2.25", 
    f"z_phot_chi2/nusefilt < 30", f"0.1 < z_phot_chi2/nusefilt"
).filter(data_catalog)

k_corr = 2.5 * np.log10(1. + cat["z_phot"])

distmod = cosmol.distmod(cat["z_phot"]).value

absM = cat["K_mag_auto"] - distmod + k_corr # cat["DISTMOD"] #
cat.add_column(absM, name="absM")

print(Table([cat["DISTMOD"], distmod, cat["DISTMOD"] - distmod, k_corr], names = [1,2,3,4]))

print(K_faint - cat["K_mag_auto"])

#z_grid = np.linspace(1e-5, 30, 1000)
z_grid = np.logspace(0, 1.5, 1001)[1:] - 1.
print(z_grid[0])

dL_grid = cosmol.luminosity_distance(z_grid).to(u.pc)
print(dL_grid)

zmax_func_vals = K_faint - (5. * np.log10(dL_grid / (10. * u.pc))) + 2.5 * np.log10(1. + z_grid)
zmax_diff_vals = abs(zmax_func_vals[:, None] - cat["absM"].data) # Each row is a z_max point -- cols are gals.
zmax_idxs = np.argmin(zmax_diff_vals, axis=0)
z_max = z_grid[zmax_idxs]

zmin_func_vals = K_bright - (5. * np.log10(dL_grid / (10. * u.pc))) + 2.5 * np.log10(1. + z_grid)
zmin_diff_vals = abs(zmin_func_vals[:, None] - cat["absM"].data) # Each row is a z_max point -- cols are gals.
zmin_idxs = np.argmin(zmin_diff_vals, axis=0)
z_min = z_grid[zmin_idxs]

print(zmax_diff_vals)
print(zmax_diff_vals[:,0])

print(zmax_diff_vals.shape)

#sys.exit()



d_max = (10**((K_faint - cat["absM"] + 5.) / 5.)) * u.pc
#cat.add_column(d_max.to(u.Mpc), name="d_max")

#d_min = (10**((K_bright - cat["absM"] + 5.) / 5.)) * u.pc
#cat.add_column(d_min.to(u.Mpc), name="d_min")

#print( Table([cat["dL"], cat["d_max"], cat["d_min"]]) )


def cosmol_spline(cosmol_function_of_z, zmin=0.0, zmax=100.0, N=10000, plot=False):
    z_vals = np.linspace(zmin, zmax, N)
    f_vals = cosmol_function_of_z(z_vals)
    print("spline max", f_vals.max())
    if not all(np.diff(f_vals) > 0):
        raise ValueError("function outputs are not monotonic!")
    if plot:
        fig, ax = plt.subplots()
        ax.plot(f_vals, 1. + z_vals)
        ax.loglog()
        plt.show()
    
    return interp1d(f_vals, z_vals, kind="cubic")

spline = cosmol_spline(cosmol.luminosity_distance)

#print("input_max", cat["d_max"].to(u.Mpc).max())
z_max_naive = spline(d_max.to(u.Mpc))
#z_min = spline(cat["d_min"].to(u.Mpc))


"""
dL_app = cosmol.luminosity_distance(cat["z_phot"])

print("dL_app dL diff")

print( Table([cat["dL"], dL_app, cat["dL"] - dL_app], names=["dL", "app", "diff"]) )"""

cat.add_columns([z_max, z_min, z_max_naive], names=["z_max", "z_min", "z_max_naive"])
print(Table([cat["z_phot"], cat["z_max"], cat["z_max_naive"], cat["z_min"]]))

print("max of zmax", max(cat["z_max"]))

#cat["z_min"] = np.zeros(len(cat)) + 0.1
#print("zmax max", z_max.max())

"""
numpy_path = paths.base_path / "zmax.npy"
if not numpy_path.exists():
    z_max = np.array(
        [
            cosmology.z_at_value(cosmol.luminosity_distance, d_max_i, zmax=6.0) 
            for d_max_i in d_max
        ]
    )
    np.save(numpy_path, z_max)
else:
    z_max = np.load(numpy_path)"""

#print(zmax_spline - z_max)

def calc_vol(z_max, z_min, omega, cosmol=cosmol):
    rmax = cosmol.comoving_distance(z_max)
    rmin = cosmol.comoving_distance(z_min)

    return omega / 3 * (rmax ** 3 - rmin ** 3)

#V_max = cosmol.comoving_volume(cat["z_max"]) * survey_frac
#V_max = calc_vol(cat["z_max"], cat["z_min"], opt_area * (np.pi / 180.)**2)

#V_i = calc_vol(cat["z_phot"], np.zeros(len(cat)), opt_area * (np.pi / 180.)**2)

"""
fig, ax = plt.subplots()
bins = np.linspace(0, 2, 200)
V_hist, V_edges = np.histogram(V_i / V_max, bins=bins)
V_mids = calc_mids(V_edges)

ax.plot(V_mids, V_hist)
plt.show()

cat["v_max"] = V_max"""



def schechter(M, phi_star, M_star, alpha):
    delta_M = M - M_star
    x = -0.4 * delta_M * (alpha + 1.)
    return 0.4 * np.log(10.) * phi_star * (10 ** x) * np.exp(-10**(-0.4 * delta_M))

def log_schechter(M, phi_star, M_star, alpha):
    delta_M = M - M_star
    y = 0.4 * np.log(10.)
    return np.log(phi_star * y) - y * (delta_M) * (alpha + 1.) - 10 ** (-0.4 * delta_M)

def log_likelihood(theta, x, y, yerr):
    #m, b, log_f = theta
    #model = log_schechter(x, *theta)
    model = schechter(x, *theta)
    #yerr = yerr / y
    #y = np.log(y)
    sigma2 = yerr ** 2 #+ model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    if not all(np.isfinite(theta)):
        return -np.inf
    return 0.0

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, yerr)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

bin_width = 0.25
absM_bins = np.arange(-30., -18., bin_width)
absM_mids = calc_mids(absM_bins)
absM_hist, _ = np.histogram(cat["absM"], bins=absM_bins)
fig,ax = plt.subplots()
ax.scatter(absM_mids, absM_hist)
ax.semilogy()
ax.set_ylim(0.1, 1e5)

m_grid = np.linspace(-26., -20., 1000)
guess = (2.2e-4, -22.43, -0.94)
s_vals = schechter(m_grid, *guess)

#fig, ax1 = plt.subplots()
#ax1.plot(m_grid, s_vals, color="k")
"""
z_bins   = [0.25, 0.75, 1.00, 1.25, 1.5, 1.75]#, 2.25]
M_faint  = [-20.75, -21.25, -21.75, -22.25, -22.50]
M_bright = [-30.00, -30.00, -30.00, -30.00, -30.00] #,[-24.75, -24.5, -24.5, -25.25, -25.25]

uds_params = [
    (2.9e-3, -22.86, -0.99), 
    (3.4e-3, -22.86, -1.00), 
    (2.8e-3, -22.93, -0.94), 
    (2.6e-3, -23.03, -0.92),
    (1.7e-3, -23.23, -1.00),
]"""


z_bins  =  [0.2,    0.4,    0.6,   0.8,     1.0,   1.25,   1.50,   1.75,   2.00]
M_faint =  [     -20.25, -20.75, -21.00, -21.50, -22.00, -22.25, -22.75, -23.00]
M_bright = [     -30.00, -30.00, -30.00, -30.00, -30.00, -30.00, -30.00, -30.00]


def Mk_z(z):
    return -22.26 - (z / 1.78) ** 0.47

def phi0_z(z):
    return 3.5e-3 * np.exp( -(z / 1.7) ** 1.47 )

def alpha_z(z):
    return -1.07

z_mids = 0.5 * (np.array(z_bins[:-1]) + np.array(z_bins[1:]))
uds_params = [
    (phi0_z(z), Mk_z(z), alpha_z(z)) for z in z_mids
]


z_grid = np.arange(0.0, 2.25, 0.1)
Mz_fig, (z_ax, Mz_ax) = plt.subplots(
    2, 1, gridspec_kw={"height_ratios": [1, 2]}, sharex="col"
)
z_ax.hist(cat["z_phot"], bins=z_grid, histtype="step")
Mz_ax.scatter(cat["z_phot"], cat["absM"], s=1, color="k", alpha=0.1)
for ii, (z_low, z_high) in enumerate(zip(z_bins[:-1], z_bins[1:])):
    Mz_ax.fill_between(
        [z_low, z_high], [M_faint[ii], M_faint[ii]], [M_bright[ii], M_bright[ii]],
        alpha=0.3, color=f"C{ii}"
    )
#plt.show()
phi_fig, phi_axes = plt.subplots(3,3, figsize=(9,9))
phi_axes = phi_axes.flatten()
for ii, (z_low, z_high) in enumerate(zip(z_bins[:-1], z_bins[1:])):
    phi_ax = phi_axes[ii]
    phi_ax.text(0.05, 0.05, f"{z_low:.2f} < z < {z_high:.2f}", transform=phi_ax.transAxes)
    phi_ax.plot(m_grid, schechter(m_grid, *uds_params[ii]), color="k")
    z_cat = Query(
        f"{z_low} < z_phot", f"z_phot < {z_high}",
        f"{M_bright[ii]} < absM", f"absM < {M_faint[ii]}",
    ).filter(cat)

    z_cat["z_max"] = np.minimum(z_cat["z_max"], z_high)
    z_cat["z_min"] = np.maximum(z_cat["z_min"], z_low)

    z_cat["v_max"] = calc_vol(
        z_cat["z_max"], z_cat["z_min"], opt_area * (np.pi / 180.)**2
    )

    phi, _ = np.histogram(
        z_cat["absM"], weights=1. / z_cat["v_max"], bins=absM_bins
    )

    phi = phi / bin_width
    phi_err, _ = np.histogram(
        z_cat["absM"], weights=1. / np.sqrt(z_cat["v_max"]), bins=absM_bins
    )

    Nhist, _ = np.histogram(z_cat["absM"], bins=absM_bins)
    phi_err = phi / 4. #np.sqrt(Nhist)

    mask = phi > 10e-10

    m_good = absM_mids[ mask ]#[2:-1]
    phi_err_good = phi_err[ mask ]#[2:-1]
    phi_good = phi[ mask ]#[2:-1]

    nwalkers = 32
    ndim = 3
    pos = guess + 1e-4 * np.random.randn(nwalkers, ndim)

    params=uds_params[ii]
    """
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(m_good, phi_good, yerr_good)
    )
    sampler.run_mcmc(pos, 5000, progress=True)
    samples = sampler.get_chain()
    params = np.median(samples[:, :, :], axis=(0,1))
    
    #params[0] = np.exp(params[0])

    fig, axes = plt.subplots(
        3, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [2, 1]}, sharex="col", sharey="row"
    )

    labels = ["phi_star", "m_star", "alpha"]
    for jj in range(ndim):
        ax0 = axes[jj, 0]
        p_samples = samples[:, :, jj]
        ax0.plot(p_samples, "k", alpha=0.3)
        ax0.set_xlim(0, len(samples))
        ax0.set_ylabel(labels[jj])
        ax0.yaxis.set_label_coords(-0.1, 0.5)
        ax1 = axes[jj, 1]
        p_hist, p_edges = np.histogram(p_samples, bins=50)
        p_mids = calc_mids(p_edges)
        ax1.plot(p_hist, p_mids)"""

    
    print(params)
    phi_ax.errorbar(m_good, phi_good, yerr=phi_err_good, color=f"C{ii}")


    phi_ax.plot(m_grid, schechter(m_grid, *params), color=f"C{ii}")
    phi_axes[-1].plot(m_grid, schechter(m_grid, *params), color=f"C{ii}")
    phi_ax.semilogy()
    phi_ax.set_ylim(1e-7, 1e-2)
    phi_ax.set_xlim(-20, -26)
    
    if ii < 6:
        phi_ax.set_xticks([])
    if ii % 3 != 0:
        ax.set_yticks([])

phi_axes[-1].semilogy()
phi_axes[-1].set_ylim(1e-7, 1e-2)
phi_axes[-1].set_xlim(-20, -26)

plt.subplots_adjust(hspace=0., wspace=0.)
plt.show()

