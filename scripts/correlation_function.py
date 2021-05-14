import logging
import pickle
import yaml
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.table import Table

import emcee
from dustmaps import sfd
from easyquery import Query
from treecorr import NNCorrelation, Catalog

from dxs.utils.misc import print_header, calc_mids
from dxs.utils.image import uniform_sphere, calc_survey_area, objects_in_coverage
from dxs.utils.phot import apply_extinction, vega_to_ab

from dxs import paths

logger = logging.getLogger("corr_func")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

object_choices = ["ero_245", "ero_295", "pe_gzK", "sf_gzK", "drg", "gals"]
default_treecorr_config_path = paths.config_path / "treecorr/treecorr_default.yaml"


ic_guess = 0.006

gmag = "g_mag_aper_30"
imag = "i_mag_aper_30"
zmag = "z_mag_aper_30"
Jmag = "J_mag_aper_30"
Kmag = "K_mag_aper_30"
Ktot_mag = "K_mag_auto"

queries_lookup = {
    "ero_245": (f"{imag} - {Kmag} > 2.45", ),
    "ero_295": (f"{imag} - {Kmag} > 2.95", ),
    "sf_gzK": (f"({zmag} - {Kmag}) - 1.27 * ({gmag}-{zmag}) >= -0.022", ),
    "pe_gzK": (f"({zmag}-{Kmag}) - 1.27 * ({gmag}-{zmag}) < -0.022", f"{zmag}-{Kmag} > 2.55", ),
    "drg": (f"{Jmag} - {Kmag} > 1.3", ),
}

###====================== selecting ====================###

def get_selection(catalog, obj_type, K_cut):
    obj_query = (*queries_lookup[obj_type], f"{Ktot_mag} < {K_cut}")
    print("use query", obj_query)
    selection = Query(*obj_query).filter(catalog)
    print(f"select {len(selection)} from {len(catalog)}")
    return selection
    
def get_randoms(ra_limits, dec_limits, mask_list, randoms_density):
    full_randoms = SkyCoord(
        uniform_sphere(ra_limits, dec_limits, density=randoms_density), 
        unit="degree"
    )
    random_mask = objects_in_coverage(
        mask_list, full_randoms.ra, full_randoms.dec
    )
    randoms = full_randoms[ random_mask ]
    return randoms

###====================== correlation stuff =======================###

def prepare_components(catalog, randoms, treecorr_config, catalog2=None, randoms2=None):

    if "num_threads" not in treecorr_config:
        treecorr_config["num_threads"] = 3
   
    ###============== do DD stuff =============###
    data_catalog = Catalog(
        ra=catalog["ra"], 
        dec=catalog["dec"], 
        ra_units="deg", dec_units="deg",
        npatch=treecorr_config.get("npatch", 1),
    )
    if catalog2 is not None:
        data_catalog2 = Catalog(
            ra=catalog2["ra"], 
            dec=catalog2["dec"], 
            ra_units="deg", dec_units="deg",
            npatch=data_catalog.patch_centers,
        )
    else:
        data_catalog2 = None

    dd = NNCorrelation(treecorr_config)
    if data_catalog2 is None:
        print("##============ process DD")
        dd.process(data_catalog)
    else:
        print("##============ process cross DD")
        dd.process(data_catalog, data_catalog2)

    ###============== do RR stuff ===============###
    random_catalog = Catalog(
        ra=randoms.ra, 
        dec=randoms.dec, 
        ra_units="deg", dec_units="deg",
        patch_centers=data_catalog.patch_centers,
    )
    use_randoms_catalog2 = False
    if randoms2 is not None:
        if data_catalog2 is not None:
            random_catalog2 = Catalog(
                ra=randoms2.ra, 
                dec=randoms2.dec, 
                ra_units="deg", dec_units="deg",
                patch_centers=data_catalog2.patch_centers,
            )
            rr.process(random_catalog, random_catalog2)
            use_randoms_catalog2 = True
        else:
            print("ignoring randoms2; data_catalog2 is None")
    else:
        random_catalog2 = None

    rr = NNCorrelation(treecorr_config)
    if use_randoms_catalog2:
        print("##============ process cross RR")
        rr.process(random_catalog, random_catalog2)
    else:
        print("##============ process RR")
        rr.process(random_catalog)

    ###============ do DR/RD stuff =============###
    dr = NNCorrelation(treecorr_config)
    print("##============ process DR")
    dr.process(data_catalog, random_catalog)
    if data_catalog2 is not None:
        rd = NNCorrelation(treecorr_config)
        if random_catalog2 is not None:
            print("##============ process RD with randoms2")
            rd.process(data_catalog2, random_catalog2)
        else:
            print("##============ process RD with same DR randoms")
            rd.process(data_catalog2, random_catalog)
    else:
        rd = None

    w_ls, w_lserr = dd.calculateXi(rr=rr, dr=dr, rd=rd)

    return dd, dr, rd, rr


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
    if rr is not None:
        IC = IC_roche(params, x, func, rr)
        y = y + IC

    sigma2 = yerr * yerr
    resid2 = (y - model) * (y - model)
    #if fit_log:
        #lin_model = model.copy()
        #model = np.log(model)
        #yerr = yerr / lin_model
        #y = np.log(y)
    return -0.5 * np.sum( resid2 / sigma2 + np.log(2 * np.pi * sigma2) ) # 2*pi in log not important.

def log_prior(params, func):
    if any([p < 0 for p in params]):
        return -np.inf

    if func.__name__ == "power_exp_law":
        pass
    elif func.__name__ == "double_power_law":
        if params[1] < params[3]: # ie, if ss-powerlaw is not steeper than ls-powerlaw
            return -np.inf
    elif func.__name__ == "broken_power_law":
        if params[4] < 1.1: # smoothness parameter must be greater than 1???
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


def run_sampler(
    x, y, yerr, func, rr=None, p0=None, n_walkers=32, n_steps=5000, routine="emcee", xmin=1e-10, xmax=180.
):
    mask = (xmin < x) & (x < xmax)
    x = x[ mask ]
    y = y[ mask ]
    yerr = yerr[ mask ]
    if rr is not None:
        rr = rr[ mask ]

    if p0 is None:
        len_args = func.__code__.co_argcount - len(func.__defaults__ or []) - 1 # -1 is for "x"
        p0 = np.ones(len_args)
    n_dim = len(p0)
    pos0 = p0 + 1e-4 * np.random.randn(n_walkers, n_dim)
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability, args=(x, y, yerr, func, rr)
    )
    sampler.run_mcmc(pos0, n_steps, progress=True)
    return sampler

def parameters_from_sampler(sampler, burn_in=1000, estimate="median", log_bins=True, Nbins=50):
    samples = sampler.get_chain()
    samples = samples[burn_in:, :, :]
    if estimate == "median":
        params = np.median(samples, axis=(0,1))
    elif estimate == "mean":
        params = np.average(samples, axis=(0,1))
    elif estimate == "mode":
        params = np.zeros(samples.shape[2])
        for ii in range(samples.shape[2]):
            data = samples[:, :, ii].flatten()
            if log_bins:
                bins = np.logspace(np.log10(data.min()), np.log10(data.max()), Nbins+1)
            else:
                bins = np.linspace(data.min(), data.max(), Nbins+1)
            hist, _ = np.histogram(data, bins=bins)
            bin_mids = 0.5 * (bins[:-1] + bins[1:])
            max_ind = np.argmax(hist)
            params[ii] = bin_mids[max_ind]
    pcov = np.var(samples, axis=(0,1))

    return params, pcov

def plot_sampler(sampler, burn_in=1000, log_scale=True, Nbins=50):
    samples = sampler.get_chain()
    n_dim = samples.shape[2]
    fig, axes = plt.subplots(
        n_dim, 2, sharey="row", gridspec_kw={"width_ratios": [2, 1]}, 
        figsize=(7, n_dim*1.5)
    )
    median_params,_ = parameters_from_sampler(sampler, burn_in=burn_in)
    mean_params,_ = parameters_from_sampler(sampler, burn_in=burn_in, estimate="mean")
    mode_params,_ = parameters_from_sampler(sampler, burn_in=burn_in, estimate="mode", log_bins=log_scale, Nbins=50)

    for jj in range(n_dim):
        ax1 = axes[jj, 0] #plt.subplot(gs[jj, :-1])
        ax1.plot(samples[:, :, jj], "k", alpha=0.3)
        ax1.set_xlim(0, len(samples))

        ax2 = axes[jj, 1] #plt.subplot(gs[jj, -1:])
        data = samples[burn_in:, :, jj].flatten()
        full_data = samples[:, :, jj].flatten()
        if log_scale:
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), Nbins)
        else:
            bins = np.linspace(data.min(), data.max(), Nbins)
        mids = calc_mids(bins)
        full_hist, _ = np.histogram(full_data, bins=bins)
        full_hist = full_hist / np.sum(full_hist)
        hist, edges = np.histogram(data, bins=bins)
        hist = hist / np.sum(hist)
        ax2.plot(full_hist, mids, color="k", ls=":") # y direction.
        ax2.plot(hist, mids, color="k") # y direction.
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
    return fig, axes
    

        
    

if __name__ == "__main__":
    default_fields = ["EN", "SA", "LH", "XM"]
    field_choices = ["EN", "SA", "LH", "XM", "Euclid"]

    parser = ArgumentParser()
    parser.add_argument("--fields", choices=field_choices, default=default_fields, nargs="+")
    parser.add_argument("--obj", choices=object_choices, default="eros")
    parser.add_argument("--obj2", choices=object_choices, default=None)
    parser.add_argument("--suffix", default="_panstarrs")
    parser.add_argument("--config-path", default=default_treecorr_config_path)
    args = parser.parse_args()

    suffix = args.suffix

    with open(args.config_path, "r") as f:
        print(f"reading {args.config_path}")
        treecorr_config = yaml.load(f, Loader=yaml.FullLoader)

    if "num_threads" not in treecorr_config:
        treecorr_config["num_threads"] = 3
    
    randoms_density = 10000 #treecorr_config.pop("randoms_density", int(3.6e4))
    print(f"use randoms density {randoms_density} per sq deg.")

    sfdq = sfd.SFDQuery()


    data_path = paths.base_path / f"ero_corr_rd{randoms_density:d}{args.suffix}.pkl"
    if data_path.exists():
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
        print("READ EXISTING DATA")

    else:
        data_dict = {}
        data_catalogs = []
        random_catalogs = []

        print(args.fields)

        for ii, field in enumerate(args.fields):
            print_header(f"looking at {field}")
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

            catalog_path = paths.get_catalog_path(field, 0, "", suffix=args.suffix)
            if not catalog_path.exists():
                print(f"no catalog {catalog_path}")
                continue
            full_catalog = Table.read(catalog_path)

            #logger.info("apply reddening")
            coords = SkyCoord(ra=full_catalog["ra"], dec=full_catalog["dec"], unit="deg")
            ebv = sfdq(coords)

            full_catalog[gmag] = apply_extinction(full_catalog[gmag], ebv, band="g")
            full_catalog[imag] = apply_extinction(full_catalog[imag], ebv, band="i")
            full_catalog[zmag] = apply_extinction(full_catalog[zmag], ebv, band="z")

            logger.info("do ab transform")
            
            full_catalog[Jmag] = vega_to_ab(full_catalog[Jmag], band="J")
            full_catalog[Kmag] = vega_to_ab(full_catalog[Kmag], band="K")
            full_catalog[Ktot_mag] = vega_to_ab(full_catalog[Ktot_mag], band="K")

            catalog = Query(
                "J_crosstalk_flag < 1", "K_crosstalk_flag < 1", 
                "J_coverage > 0", "K_coverage > 0"
            ).filter(full_catalog)

            opt_coverage_mask = objects_in_coverage(
                opt_mask_list, ra=catalog["ra"], dec=catalog["dec"]
            )
            catalog = catalog[opt_coverage_mask]

            eros = get_selection(catalog, obj_type="ero_245", K_cut=20.7)
            drgs = get_selection(catalog, obj_type="drg", K_cut=20.7)
            
            optical_randoms = get_randoms(
                (catalog["ra"].min(), catalog["ra"].max()), 
                (catalog["dec"].min(), catalog["dec"].max()),
                opt_mask_list,
                randoms_density,
            )

            dd, dr, rd, rr = prepare_components(eros, optical_randoms, treecorr_config)
            x = dd.rnom / 3600.
            w, w_err = dd.calculateXi(dr=dr, rd=rd, rr=rr)
            ddp = dd.npairs
            drp = dr.npairs
            rrp = rr.npairs
            rdp = rd.npairs if rd is not None else None

            Nr = len(optical_randoms)
            Nd = len(eros)

            field_data = {
                "x": x, "dd": ddp, "dr": drp, "rd": rdp, "rr": rrp, "w": w, "w_err": w_err,
                "Nd": Nd, "Nr": Nr
            }

            data_dict[field] = field_data

        with open(data_path, "wb+") as f:
            pickle.dump(data_dict, f)
    
    kim2014_ero_corr_path = paths.input_data_path / "plotting/jwk_2pcf_ps_eros_245.csv"

    kim2014_data = pd.read_csv(kim2014_ero_corr_path, names="x w".split())

    func = double_power_law

    x_list = [ d["x"] for field, d in data_dict.items() ]
    for x in x_list:
        assert np.allclose(x, x_list[0])
    x = x_list[0]
        

    total_dd = sum([ d["dd"] for field, d in data_dict.items() ])
    total_dr = sum([ d["dr"] for field, d in data_dict.items() ])
    total_rr = sum([ d["rr"] for field, d in data_dict.items() ])
    #total_dd = sum([d["dd"] in field, d in data_dict.items()])

    Nd = sum([ d["Nd"] for field, d in data_dict.items() ])
    Nr = sum([ d["Nr"] for field, d in data_dict.items() ])

    DD_tot = total_dd / (0.5 * Nd * (Nd - 1))
    DR_tot = total_dr / (Nd * Nr)
    RR_tot = total_rr / (0.5 * Nr * (Nr - 1))

    w_tot = (DD_tot - 2 * DR_tot + RR_tot) / (RR_tot)
    w_tot_err = (1. + w_tot) / np.sqrt(total_dd)

    #fig, ax = plt.subplots()
    #ax.scatter(kim2014_data["x"], kim2014_data["w"], s=8, color="k", label="Kim+ 2014")
    #ax.plot(x, w_tot, color="k", label="comb.")
    xvals = np.logspace(-4, 1, 1000)
    for ii, (field, data) in enumerate(data_dict.items()):
        fig, ax = plt.subplots()
        ax.scatter(kim2014_data["x"], kim2014_data["w"], s=8, color="k", label="Kim+ 2014")
        x = data["x"]
        w = data["w"]
        w_err = (1. + w) / np.sqrt(data["dd"]) #5. #data["w_err"]
        rr = data["rr"]

        ax.errorbar(x, w, yerr=w_err, color=f"C{ii}", label=field)
        ax.plot(x, w + ic_guess, color=f"C{ii}", ls="--")

        
        p0 = p0_lookup[func.__name__]
        print(f"{func.__name__} p0: {p0}")
        
        sampler = run_sampler(
            x, w, w_err, func, rr=rr, p0=p0, xmin=1.2e-3, xmax=0.5, n_walkers=64, n_steps=10000
        )
        params, pcov = parameters_from_sampler(sampler)
        IC = IC_roche(params, x, func, rr)
        print(params, IC)
        samples_fig, samples_ax = plot_sampler(sampler, log_scale=True)

        ax.plot(x, w + IC, color=f"C{ii}", ls=":")
        
        model_y = func(xvals, *params)
        ax.plot(xvals, model_y)
        

        ax.loglog()
        ax.set_xlim(4e-4, 4e0)
        ax.set_ylim(1e-3, 1e1)

        ax.legend()

    plt.show()



