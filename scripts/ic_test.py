import inspect
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

import emcee
import corner

from dxs.utils.misc import calc_mids
from dxs import paths

fit_log = True

### ===================functional forms=================== ###

def power_law(x, A, d):
    return A * x ** (-d)

def exp_law(x, alpha, beta):
    return alpha * np.exp(-beta * x)

def double_power_law(x, A1, d1, A2, d2):
    return A1 * x ** (-d1) + A2 * x ** (-d2)

def double_power_law_C(x, A1, d1, A2, d2, C):
    return A1 * x ** (-d1) + A2 * x ** (-d2) - C

def broken_power_law(x, A, d1, d2, x_b, delta):
    p1 = (x / x_b) ** (-d1)
    p2 = (0.5 * (1. + x / x_b) ** (1./delta)) 
    return A * p1 * p2 ** (delta * (d1 - d2))

def fixed_double_power_law(x, A1, A2):
    return A1 * x ** (-1.0) + A2 * x ** (-0.4)

def power_exp_law(x, A, d, alpha, beta):
    return A * x ** (-d) + alpha * np.exp(-beta * x)

p0_lookup = {
    "power_law": [1e-3, 0.8],
    "exp_law": [1e-1, 10.],
    "double_power_law": [1e-3, 1.0, 1e-2, 0.4],
    "double_power_law_C": [1e-3, 1.0, 1e-2, 0.4, 0.008],
    "broken_power_law": [0.02, 1.0, 0.4, 0.02, 2.0],
    "fixed_double_power_law": [1e-4, 1e-3],
    "power_exp_law": [1e-4, 1.0, 5e-3, 10.],
}

composite_lookup = {
    "power_exp_law": [power_law, exp_law],
    "double_power_law": [power_law, power_law],
}

### ===================emcee functions===================== ###

def IC_roche(params, theta, func, rr):
    w_theta = func(theta, *params)
    return np.sum(rr * w_theta) / np.sum(rr)

def log_likelihood(params, x, y, yerr, func, rr):
    model = func(x, *params)
    #if rr is not None:
    #    model = model - IC_roche(params, x, func, rr)
    #yerr = yerr / model
    sigma2 = yerr * yerr
    #model = np.log10(model)
    #y = np.log(y)
    resid2 = (y - model) * (y - model)

    return -0.5 * np.sum( resid2 / sigma2 + np.log(2 * np.pi * sigma2) ) # 2*pi in log not important.

def log_prior(params, func):
    if any([p < 0 for p in params]):
        return -np.inf

    if func.__name__ == "power_exp_law":
        pass
    elif func.__name__ == "double_power_law":
        if params[1] < params[3]:
            return -np.inf
    elif func.__name__ == "broken_power_law":
        if params[4] < 1.1:
            return -np.inf
    return 0.

def log_probability(params, x, y, yerr, func, rr):
    lp = log_prior(params, func)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, x, y, yerr, func, rr)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def fit_parameters(
    x, y, yerr, func, rr=None, p0=None, routine="emcee", 
    return_sampler=True,
    xmin=-np.inf, xmax=np.inf, **kwargs
):
    mask = (xmin < x) & (x < xmax)
    x = x[ mask ]
    y = y[ mask ]
    yerr = yerr[ mask ]
    if rr is not None:
        rr = rr[ mask ]
    if p0 is None:
        len_args = func.__code.__.co_argcount - len(func.__defaults__ or [])
        p0 = np.ones(len_args)

    if routine == "emcee":
        ndim = len(p0)
        nwalkers = kwargs.get("nwalkers", 32)
        nsteps = kwargs.get("nsteps", 10000)
        
        pos0 = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(x, y, yerr, func, rr)
        )
        sampler.run_mcmc(pos0, nsteps, progress=True)
        samples = sampler.get_chain()
        burn_in = kwargs.get("burn_in", 1000)
        params = np.median(samples[burn_in:, :, :], axis=(0,1))
        pcov = np.var(samples[burn_in:, :, :], axis=(0,1))

        if return_sampler:
            return params, pcov, sampler
        else:
            return params, pcov

def parameters_from_samples(
    samples, selection="mode", log_bins=False, Nbins=100, burn_in=None
):
    if burn_in is not None:
        samples = samples[burn_in:, :, :]
    if selection == "median":
        params = np.median(samples[:, :, :], axis=(0,1))
    elif selection == "mean" or selection == "average":
        params = np.average(samples[:, :, :], axis=(0,1))
    elif selection == "mode":
        params = np.zeros(samples.shape[2])
        for ii in range(samples.shape[2]):
            data = samples[:, :, ii].flatten()
            #if isinstance(bins, int):
            if log_bins:
                bins = np.logspace(np.log10(data.min()), np.log10(data.max()), Nbins+1)
            else:
                bins = np.linspace(data.min(), data.max(), Nbins+1)
            hist, _ = np.histogram(data, bins=bins)
            bin_mids = 0.5 * (bins[:-1] + bins[1:])
            max_ind = np.argmax(hist)
            params[ii] = bin_mids[max_ind]
    else:
        raise ValueError("use selection from 'median', 'mean', 'mode'")
    return params

def calc_ic(
    x, y, yerr, func, rr, p0=None, routine="emcee", xmin=None, xmax=None, iterations=10, **kwargs
):
    ic_est = 0.
    for ii in range(iterations):
        y_est = y + ic_est
        params, pcov, sampler = fit_parameters(
            x, y_est, yerr, func, rr=rr, p0=p0, routine=routine, xmin=xmin, xmax=xmax
        )
        ic_est = IC_roche(params, x, func, rr)
        print(f"params are: {pprint(params)}")
        print(f"IC est is {ic_est:.4f}")
        p0 = params
    return ic, params

def pprint(l):
    pl = [f"{x:.3e}" if x > 0.001 and x < 1000.0 else f"{x:3e}" for x in l ]
    return pl

def plot_sampler(sampler, params=None, burn_in=1000, log_scale=True):
    samples = sampler.get_chain()
    ndim = samples.shape[2] # third dimension.
    #gs = plt.GridSpec(ndim, 3)
    #fig = plt.figure(figsize=(9, ndim*1.5)) #, sharex=True)

    fig, axes = plt.subplots(
        ndim, 2, sharey="row", gridspec_kw={"width_ratios": [2, 1]}, 
        figsize=(9, ndim*1.5)
    )

    median_params = parameters_from_samples(
        samples, selection="median", burn_in=burn_in
    )
    mode_params = parameters_from_samples(
        samples, selection="mode", burn_in=burn_in, log_bins=log_scale, Nbins=50
    )
    print("mode params", mode_params)
    mean_params = parameters_from_samples(
        samples, selection="mean", burn_in=burn_in
    )

    for jj in range(ndim):
        ax1 = axes[jj, 0] #plt.subplot(gs[jj, :-1])
        ax1.plot(samples[:, :, jj], "k", alpha=0.3)
        ax1.set_xlim(0, len(samples))

        ax2 = axes[jj, 1] #plt.subplot(gs[jj, -1:])
        data = samples[burn_in:, :, jj].flatten()
        full_data = samples[:, :, jj].flatten()
        if log_scale:
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)
        else:
            bins = np.linspace(data.min(), data.max(), 50)
        mids = calc_mids(bins)
        full_hist, _ = np.histogram(full_data, bins=bins)
        full_hist = full_hist / np.sum(full_hist)
        hist, edges = np.histogram(data, bins=bins)
        hist = hist / np.sum(hist)
        ax2.plot(full_hist, mids, color="k", ls=":") # y direction.
        ax2.plot(hist, mids, color="k") # y direction.
        if params is not None:
            ax1.axhline(params[jj], color="r")
            ax2.axhline(params[jj], color="r")
        ax1.axhline(median_params[jj], color="C0")
        ax2.axhline(median_params[jj], color="C0")
        ax1.axhline(mode_params[jj], color="C1")
        ax2.axhline(mode_params[jj], color="C1")
        ax1.axhline(mean_params[jj], color="C2")
        ax2.axhline(mean_params[jj], color="C2")
        if log_scale:
            ax1.semilogy()
            ax2.semilogy()

    fig.subplots_adjust(hspace=0, wspace=0)
    return fig

if __name__ == "__main__":
    ic_guess = 0.008

    ic = 0.

    burn_in = 1000

    pkl_path = paths.data_path / "EN_eros_245_K207_Ncorr_data.pkl"
    with open(pkl_path, "rb") as f:
        corr_data = pickle.load(f)

    jwk_path = paths.input_data_path / "plotting/jwk_2pcf_ps_eros_245.csv"
    jwk = pd.read_csv(jwk_path, names=["x", "w"])

    theta = corr_data["x"]
    w_init = corr_data["w_ls"]
    rr = corr_data["rr"]
    dd = corr_data["dd"]
    w_err = np.sqrt(corr_data["w_ls_var"])

    include_f = False

    func = double_power_law # power_exp_law # 
    p0 = p0_lookup[func.__name__]
    print(p0)

    xmin, xmax = 1e-3, 0.5
    params, pcov, sampler = fit_parameters(
        theta, w_init, w_err, func, rr=rr, p0=p0, xmin=xmin, xmax=xmax,
        nsteps=10000, nwalkers=128,
    )
    plot_sampler(sampler, params=params, log_scale=True, burn_in=burn_in)

    samples = sampler.get_chain()[burn_in:, :, :]
    ndim = samples.shape[2]
    flat_samples = samples.transpose(2, 0, 1).reshape(ndim, -1).T
    #flat_samples = np.log10(flat_samples)
    fig = corner.corner(flat_samples)

    mode_params = parameters_from_samples(
        samples, selection="mode", Nbins=500, 
    )
    
    print(mode_params)

    ic_est = IC_roche(params, theta, func, rr)
    ic_mode = IC_roche(mode_params, theta, func, rr)
    print(f"params are: {pprint(params)}")
    print(f"first IC est is {ic_est:.4f}")
    print(f"first IC mode is {ic_mode:.4f}")
    #ic_est, params = calc_ic(
    #    theta, w_init, w_err, func, rr, p0=p0, xmin=xmin, xmax=xmax
    #)

    xvals = np.logspace(
        np.log10(theta.min()), 
        np.log10(theta.max()), 
        1000
    )
    yvals = func(xvals, *params)
    mode_yvals = func(xvals, *mode_params)

    fig, ax = plt.subplots()
    ax.errorbar(theta, w_init, yerr=w_err, color="C0", ls="-", marker="^", label="LS est.")
    ax.scatter(theta, w_init + ic_guess, color="C1", ls="--", marker="^", label="LS + ic guess")
    ax.scatter(theta, w_init + ic_est, color="C2", ls="--", marker="^", label="LS + ic est")
    ax.scatter(theta, w_init + ic_mode, color="C3", ls="--", marker="^", label="LS + ic mode")


    ax.plot(xvals, yvals, color="k", label="fit")
    ax.plot(xvals, mode_yvals, color="r", label="fit")
    ax.scatter(jwk["x"], jwk["w"], color="k", s=10, zorder=5, label="Kim+ 2014")
    
    composite_list = composite_lookup.get(func.__name__, [])
    for ii, composite_func in enumerate(composite_list):
        composite_params = params[2*ii:2*ii+2]
        ax.plot(xvals, composite_func(xvals, *composite_params), color="k", ls=":")

    ax.loglog()
    ax.legend()
    plt.show()





