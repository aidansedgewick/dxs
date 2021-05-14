import sys

import numpy as np
import matplotlib.pyplot as plt

import emcee

from astropy import cosmology
from astropy import units as u
from astropy.table import Table

from scipy.interpolate import interp1d

from easyquery import Query

from dxs.utils.phot import vega_to_ab
from dxs.utils.misc import calc_mids
from dxs import paths

cosmol = cosmology.Planck15

cat_path = paths.base_path / "./en_photoz_test/en_zphot_join.cat.fits"

full_cat = Table.read(cat_path)


opt_area = 6.93 
"""
Kbins = np.arange(16, 25, 0.5)
Kmids = calc_mids(Kbins)
fig, ax = plt.subplots()
Khist, _ = np.histogram(full_cat["K_mag_auto"], bins=Kbins)
Khist = Khist / opt_area
ax.scatter(Kmids, Khist)
ax.semilogy()
ax.set_ylim(1, 1e4)"""

K_cut = 21.5# * u.mag


survey_frac = (opt_area * (np.pi / 180.) **2 ) / (4. * np.pi)

cat = Query(
    f"K_mag_auto < {K_cut}", 
    f"0. < z_phot", f"z_phot < 2.25", 
    #f"z_phot_chi2 < 30", f"0.1 < z_phot_chi2"
).filter(full_cat)

k_corr = 2.5 * np.log10(1. + cat["z_phot"])

distmod = cosmol.distmod(cat["z_phot"]).value

cat["absM"] = cat["K_mag_auto"] - distmod + k_corr #cat["DISTMOD"]

print(Table([cat["DISTMOD"], distmod, cat["DISTMOD"] - distmod,k_corr], names = [1,2,3,4]))

print(K_cut - cat["K_mag_auto"])

cat["d_max"] = (10**((K_cut - cat["absM"] + 5.) / 5.)) * u.pc

print( Table([cat["dL"], cat["d_max"].to(u.Mpc)]) )


def cosmol_spline(cosmol_function_of_z, zmin=0.0, zmax=16.0, N=1000, plot=False):
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

print("input_max", cat["d_max"].to(u.Mpc).max())
cat["z_max"] = spline(cat["d_max"].to(u.Mpc))
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




V_max = cosmol.comoving_volume(cat["z_max"]) * survey_frac

cat["v_max"] = V_max



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

z_bins   = [0.4, 0.75, 1.00, 1.25, 1.5, 1.75]#, 2.25]
M_faint  = [-21.25, -21.75, -22.25, -22.75, -23.0]
M_bright = [-25.25, -24.5, -24.5, -24.25, -24.25]

uds_params = [
    (2.9e-3, -22.86, -0.99), 
    (3.4e-3, -22.86, -1.00), 
    (2.8e-3, -22.93, -0.94), 
    (2.6e-3, -23.03, -0.92),
    (1.7e-3, -23.23, -1.00),
]

Mz_fig, Mz_ax = plt.subplots()
Mz_ax.scatter(cat["z_phot"], cat["absM"], s=1, color="k")
for ii, (z_low, z_high) in enumerate(zip(z_bins[:-1], z_bins[1:])):
    Mz_ax.fill_between(
        [z_low, z_high], [M_faint[ii], M_faint[ii]], [M_bright[ii], M_bright[ii]],
        alpha=0.3, color=f"C{ii}"
    )
#plt.show()

for ii, (z_low, z_high) in enumerate(zip(z_bins[:-1], z_bins[1:])):
    phi_fig, phi_ax = plt.subplots()
    phi_ax.plot(m_grid, schechter(m_grid, *uds_params[ii]), color="k")
    z_cat = Query(
        f"{z_low} < z_phot", f"z_phot < {z_high}",
        f"{M_bright[ii]} < absM", f"absM < {M_faint[ii]}",
    ).filter(cat)

    phi, _ = np.histogram(
        z_cat["absM"], weights=1. / z_cat["v_max"], bins=absM_bins
    )

    phi = phi / bin_width

    #yerr, _ = np.histogram(
    #    z_cat["absM"], weights=1. / np.sqrt(z_cat["v_max"]), bins=absM_bins
    #)

    Nhist, _ = np.histogram(z_cat["absM"], bins=absM_bins)
    yerr = phi / 4. #np.sqrt(Nhist)

    mask = phi > 10e-10

    m_good = absM_mids[ mask ]#[2:-1]
    yerr_good = yerr[ mask ]#[2:-1]
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
    phi_ax.errorbar(m_good, phi_good, yerr=yerr_good, color=f"C{ii}")


    phi_ax.plot(m_grid, schechter(m_grid, *params), color=f"C{ii}")
    phi_ax.semilogy()
    phi_ax.set_ylim(1e-7, 1e-2)
    phi_ax.set_xlim(-21, -26)
plt.show()

