import inspect
import pickle
import yaml

import matplotlib
matplotlib.use("Qt5agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model

from treecorr import estimate_multi_cov

from dxs import analysis_tools as tools

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

analysis_config_path = paths.config_path / "analysis_config.yaml"
with open(analysis_config_path, "r") as f:
    analysis_config = yaml.load(f, Loader=yaml.FullLoader)

def power_law(x, A, d):
    return A * x ** (-d)

def exp_law(x, alpha, beta):
    return alpha * np.exp(-beta * x)

def trunc_power_law(x, A, d, kappa):
    return A * x ** (-d) * np.exp(-kappa * x)

def double_power_law(x, A1, d1, A2, d2):
    #return A1 * x ** (-d1) + A2 * x ** (-d2)
    return power_law(x, A1, d1) + power_law(x, A2, d2)

def double_power_law_C(x, A1, d1, A2, d2, C):
    #return A1 * x ** (-d1) + A2 * x ** (-d2) - C
    return power_law(x, A1, d1) + power_law(x, A2, d2) - C

#def broken_power_law(x, A, d1, d2, x_b, delta):
#    p1 = (x / x_b) ** (-d1)
#    p2 = (0.5 * (1. + x / x_b) ** (1./delta)) 
#    return A * p1 * p2 ** (delta * (d1 - d2))

def fixed_double_power_law(x, A1, A2):
    return A1 * x ** (-1.0) + A2 * x ** (-0.4)

def power_exp_law(x, A, d, alpha, beta):
    #return A * x ** (-d) + alpha * np.exp(-beta * x)
    return power_law(x, A, d) + exp_law(x, alpha, beta)

def double_exp_law(x, alpha1, beta1, alpha2, beta2):
    #return alpha1 * np.exp(-beta1 * x) + alpha2 * np.exp(-beta2 * x)
    return exp_law(x, alpha1, beta1) + exp_law(x, alpha2, beta2)

def trunc_power_exp_law(x, A, d, kappa, alpha, beta):
    #return A * x ** (-d) * np.exp(-kappa * x) + alpha * np.exp(-beta * x)
    return trunc_power_law(x, A, d, kappa) + exp_law(x, alpha, beta)

def double_trunc_power_law(x, A1, d1, kappa1, A2, d2, kappa2):
    return trunc_power_law(x, A1, d1, kappa1) + trunc_power_law(x, A2, d2, kappa2)

def fix_parameters(func, names, values, p0=None, bounds=None, tol=1e-6):
    sig = inspect.signature(func)
    param_list = list(sig.parameters)

    if p0 is None:
        p0 = np.ones(len(param_list))
    if bounds is None:
        bounds = ([-np.inf for _ in param_list], [np.inf for _ in param_list])
    
    if not isinstance(names, list):
        names = [names]
    if not isinstance(values, list):
        values = [values]

    for name, value in zip(names, values):
        idx = param_list.index(name) - 1
        bounds[0][idx] = value - tol
        bounds[1][idx] = value + tol
        p0[idx] = value

    return p0, bounds


p0_lookup = {
    "power_law": [1e-3, 0.8],
    "exp_law": [1e-1, 10.],
    "double_power_law": [1e-3, 1.0, 1e-2, 0.4],
    "double_power_law_C": [1e-3, 1.0, 1e-2, 0.4, 0.008],
    "broken_power_law": [0.02, 1.0, 0.4, 0.02, 2.0],
    "fixed_double_power_law": [1e-4, 1e-3],
    "power_exp_law": [1., 1., 1., 1.], #[1e-4, 1.0, 5e-3, 10.],
    "trunc_power_exp_law": [1e-2, 1.0, 5e1, 1e-2, 4e0],
    "double_trunc_power_law": [1., 1., 1., 1., 1., 1.,],
    "double_exp_law": [1e-1, 10., 1e-3, 100.],
}

bounds_lookup = {
    "double_power_law_C": (
        [1e-8, 1e-1, 1e-6, 1e-2, -1e+1],
        [1e+2, 3e+0, 1e+2, 3e+0, 1e+1]
    ),
    "trunc_power_exp_law": ( #(-np.inf, np.inf),
        [1e-5, 1e-2, 1e-2, 1e-6, 1e-2],
        [1e+2, 5e+0, 1e+2, 1e+0, 1e+3],
    )
}



composite_lookup = {
    "power_exp_law": [power_law, exp_law],
    "double_power_law": [power_law, power_law],
    "double_power_law_C": [power_law, power_law],
    "double_exp_law": [exp_law, exp_law],
    "trunc_power_exp_law": [trunc_power_law, exp_law],
    "double_trunc_power_law": [trunc_power_law, trunc_power_law]
}

composite_n_params = {
    "power_exp_law": (2, 2),
    "trunc_power_exp_law": (3, 2),
    "double_trunc_power_law": (3, 3),
}

object_names = {
    "galaxies": "galaxies", 
    "eros_245": r"EROs, $(i-K) > 2.45$", 
    "eros_295": r"EROs, $(i-K)_{AB} > 2.95$",
    "drgs": r"DRGs, $(J-K)_{AB}$ > 1.34",
    "sf_gzKs": "SF-gzKs",
    "pe_gzKs": "PE-gzKs",
}

def chi_sq(func, params, x, y, err):
    y_model = func(x, *params)
    delta = y_model - y
    chiSq = (delta * delta) / (err * err)
    return chiSq

def IC_Roche(theta, RR, func, *params):    
    if len(RR) != len(theta):
        raise ValueError("len RR should equal len theta")
    #print("in IC Roche")
    #print(params)
    w_theta = func(theta, *params)
    
    ic = np.sum(w_theta * RR) / np.sum(RR)
    return ic

def func_IC(func, theta, RR):
    def mod(x, *params):
        return func(x, *params) + IC_Roche(theta, RR, func, *params)
    return mod

gs = plt.GridSpec(2,2)

#params_fig, params_axes = plt.subplots(2,2)
#params_axes = params_axes.flatten()

func = trunc_power_exp_law
#func = double_trunc_power_law
#func = double_power_law_C # 

params_fig = plt.figure(figsize=(8.5, 6.4))
if func.__name__ in ["power_exp_law", "trunc_power_exp_law", "double_trunc_power_law"]:
    params_axes = {
        "amplitude": params_fig.add_subplot(gs[:, :1]), 
        "slope": params_fig.add_subplot(gs[:1, 1:]), 
        "cutoff": params_fig.add_subplot(gs[1:, 1:]),
    }
elif func.__name__ in ["double_power_law_C", "double_power_law"]:
    params_axes = {
        "amplitude": params_fig.add_subplot(gs[:, :1]), 
        "slope": params_fig.add_subplot(gs[:, 1:]),
    }

elif func.__name__ in ["double_exp_law"]:
    params_axes = {
        "amplitude": params_fig.add_subplot(gs[:, :1]), 
        "cutoff": params_fig.add_subplot(gs[:, 1:]),
    }

params_axes["amplitude"].semilogy()
if "cutoff" in params_axes:
    params_axes["cutoff"].semilogy()

optical_list = ["panstarrs"] #"hsc", 
    
obj = "drgs"

ls_list = ["-", "--"]

for opt_ii, optical in enumerate(optical_list):

    corr_data_path = paths.data_path / f"analysis/corr_{obj}_{optical}_d02.pkl"
    treecorr_data_path = paths.data_path / f"analysis/treecorr_{obj}_{optical}.pkl"

    with open(corr_data_path, "rb") as f:
        full_corr_data = pickle.load(f)

    K_limits = analysis_config["correlation_function"][obj]["K_limits"]


    if optical == "hsc":    
        xmin, xmax = 10**-3.3, 10**0.5
    else:
        xmin, xmax = 10**-3.1, 10**0.3
    x_grid = np.logspace(np.log10(xmin), np.log10(xmax), 1000)

    #fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    fields = ["SA", "EN",  "LH", "XM"]
    #fields = ["EN"]

    params_lookup = {field: [] for field in fields}
    params_lookup["combined"] = []

    theta_cross = {field: [] for field in fields}
    theta_cross["combined"] = []

    K_lims = []

    for ii, corr_data in full_corr_data.items():

        if isinstance(ii, str):
            continue

        K_lim = K_limits[ii]

        K_lims.append(K_lim)

        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8.5, 6.4))
        fig.suptitle(f"{obj}, K<{K_lim} ({optical})")

        axes[0].loglog()
        xref = np.logspace(-4.0, 1.0, 1000)
        pl_ref = power_law(xref, 1e-3, 0.8)
        axes[0].plot(xref, pl_ref, color="k", alpha=0.5)
        axes[0].set_ylim(1e-4, 5e1)
        axes[0].set_xlim(2e-4, 3e0)
        
        """
        tw = axes[0].twiny()
        tw.loglog()
        
        vals = [1, 2, 4, 8, 15, 30]
        arcsec = [v / 3600. for v in vals]
        arcmin = [v / 60. for v in vals]
        deg = vals
        arcsec_labels = [f"{v}"+u"\u2033" for v in vals]
        arcmin_labels = [f"{v}"+u"\u2032" for v in vals]
        deg_labels = [f"{v}"+u"\u00B0" for v in vals]
        tw.set_xticks(arcsec+arcmin+deg)
        tw.set_xticklabels(arcsec_labels + arcmin_labels + deg_labels)

        tw.set_xlim(axes[0].get_xlim())  
        tw.tick_params(axis="x", which="minor", width=0, length=0)"""

        axes[1].set_xlim(axes[0].get_xlim())
        axes[1].set_ylabel("y-f(x)/err")
        axes[1].set_ylim(-3, 3)
        axes[1].fill_between(axes[1].get_xlim(), [-2., -2.], [2., 2.], color="k", alpha=0.3)
        axes[1].fill_between(axes[1].get_xlim(), [-1., -1.], [1., 1.], color="k", alpha=0.3)
        axes[1].axhline(0., color="k")
        axes[1].semilogx()    

        x = corr_data["combined"]["x"]

        missing_fields = [f for f in fields if f not in corr_data]
        if len(missing_fields) > 0:
            print(f"missing {missing_fields}")

        NN_dd_list = [corr_data[f]["NN_dd"] for f in fields if f not in missing_fields]

        w, DD, DR, RD, RR = tools.w_mean_from_NN_list(NN_dd_list, return_components=True)
        cov, w_err = tools.jackknife_cov_from_NN_list(NN_dd_list)

        #x = comb["x"]
        mask = (xmin < x) & (x < xmax)
        x_fit = x[ mask ]
        w_fit = w[ mask ]
        w_err_fit = w_err[ mask ]
        RR_fit = RR[ mask ]

        axes[0].errorbar(
            x, w, yerr=w_err, color=f"k", ls="none", zorder=10, marker="o", mfc="none"
        )
        #axes[0].errorbar(
        #    comb["x"], comb["w"], yerr=comb["w_err"], color=f"k"
        #)

        p0 = p0_lookup[func.__name__]
        bounds = bounds_lookup.get(func.__name__, (-np.inf, np.inf))

        p0, bounds = fix_parameters(func, "d", 1.35, p0=p0, bounds=bounds)
        print(bounds)

        try:
            comb_params, comb_pcov = curve_fit(
                func, x_fit, w_fit, sigma=w_err_fit,
                p0=p0, bounds=bounds, 
                maxfev=10000000
            )
        except Exception as e:
            print("failed to fit")
            print(e)
            continue
        #model = Model(dpl)
        with np.printoptions(precision=4):
            print("comb", obj, K_lim)
            print(comb_params)

        #params_lookup["combined"].append(comb_params)
        

        fit_parameters = comb_params.copy()

        #ic_estimates = [ic_roche]
        for jj in range(20):
            IC = IC_Roche(x_fit, RR_fit, func, *fit_parameters)
            ic_roche = IC
            if func.__name__ == "double_power_law_C":
                IC = IC + fit_parameters[-1]

            w_ic = w_fit + IC
            fit_parameters, fit_pcov = curve_fit(
                func, x_fit, w_ic, sigma=w_err_fit, 
                p0=fit_parameters, bounds=bounds, 
                maxfev=10000000
            )

            IC = IC_Roche(x_fit, RR_fit, func, *fit_parameters)

            if func.__name__ == "double_power_law_C":
                IC = IC + fit_parameters[-1]
                
            with np.printoptions(precision=2):
                print(f"{jj} {obj} {K_lim} {fit_parameters} \n      {ic_roche:.5f} {IC:.5f}")

        #w_ic = w
        #fit_parameters = comb_params

        w_ic = w + IC

        axes[0].errorbar(
            x, w_ic, yerr=w_err, color=f"k", ls="none", zorder=10, marker="o",
        )
        

        params_lookup["combined"].append(fit_parameters)

        w_eval = func(x_grid, *fit_parameters)
        axes[0].plot(x_grid, w_eval, color=f"k", ls="--")

        component_functions = composite_lookup[func.__name__]
        n_params = composite_n_params.get(func.__name__, (2, 2))
        if func.__name__ in ["double_power_law", "double_power_law_C"]:
            A_ratio = comb_params[2] / comb_params[0]
            d_diff = comb_params[3] - comb_params[1]
            theta_c = A_ratio ** (1. / d_diff)
            theta_cross["combined"].append(theta_c)

        stop1 = n_params[0]
        stop2 = n_params[0] + n_params[1]

        if len(component_functions) > 1:
        
            cf1 = component_functions[0]
            print(stop1, fit_parameters[:stop1])
            component1 = cf1(x_grid, *fit_parameters[:stop1])
            axes[0].plot(x_grid, component1, color=f"k", ls=":")

            cf1_init = cf1(x_grid, *comb_params[:stop1])
            axes[0].plot(x_grid, cf1_init, color=f"k", ls=":", alpha=0.5)

            cf2 = component_functions[1]
            component2 = cf2(x_grid, *fit_parameters[stop1:stop2])
            axes[0].plot(x_grid, component2, color=f"k", ls=":")

            cf2_init = cf2(x_grid, *comb_params[stop1:stop2])
            axes[0].plot(x_grid, cf2_init, color=f"k", ls=":", alpha=0.5)


        x_eval = func(x, *fit_parameters)
        resid = (w_ic - x_eval) / w_err
        ### now plot residuals.
        axes[1].scatter(x, resid, color=f"k", s=4)
        """
        for jj, field in enumerate(fields):

            if field not in corr_data:
                print(f"No data for {field}")
                continue
            corr_data_jj = corr_data[field]
            assert np.allclose(corr_data_jj["x"], x)

            w_jj = corr_data_jj["w"]
            w_err_jj = corr_data_jj["w_err"]
            RR_jj = corr_data_jj["rr"]

            mask_jj = mask.copy()    
            x_fit = x[ mask_jj ]
            w_jj_fit = w_jj[ mask_jj ]
            w_err_jj_fit = w_err_jj[ mask_jj ]
            RR_jj_fit = RR_jj[ mask_jj ]

            axes[0].errorbar(
                x, w_jj, yerr=w_err_jj, color=f"C{jj}", ls="none", alpha=0.3, marker="x"
            )

            try:
                params_jj, pcov_jj = curve_fit(
                    func, x_fit, w_jj_fit, sigma=w_err_jj_fit,
                    p0=p0, bounds=bounds, 
                    maxfev=10000000
                )
            except Exception as e:
                print("failed to fit")
                print(e)
                continue
            params_lookup[field].append(params_jj)

            w_eval_jj = func(x_grid, *params_jj)
            axes[0].plot(x_grid, w_eval_jj, color=f"C{jj}", ls="--", alpha=0.3)
            
            #fit_parameters = params_jj.copy()
            #ic_roche = IC_Roche(x_fit, RR_fit, func, *fit_parameters)
            #ic_estimates = [ic_roche]
            #for jj in range(100):        
            #    w_ic = w_fit + ic_roche
            #
            #    fit_parameters, fit_pcov = curve_fit(
            #        func, x_fit, w_ic, sigma=w_err_fit, 
            #        p0=p0, bounds=bounds, 
            #        maxfev=10000000
            #    )
            #    ic_roche = IC_Roche(x_fit, RR_fit, func, *fit_parameters)
            #    with np.printoptions(precision=2):
            #        print(f"{jj} {obj} {K_lim} {fit_parameters} {ic_roche:.5f}")

            if func.__name__ in ["double_power_law", "double_power_law_C"]:
                A_ratio = params_jj[2] / params_jj[0]
                d_diff = params_jj[3] - params_jj[1]
                theta_c = A_ratio ** (1. / d_diff)
                theta_cross[field].append(theta_c)

            component_functions = composite_lookup[func.__name__]
            #if len(component_functions) > 1:
            #    cf1 = component_functions[0]
            #    component1 = cf1(x_grid, *params_jj[:2])
            #    axes[0].plot(x_grid, component1, color=f"C{jj}", ls=":")

            #    cf2 = component_functions[1]
            #    component2 = cf2(x_grid, *params_jj[2:4])
            #    axes[0].plot(x_grid, component2, color=f"C{jj}", ls=":")


            x_eval_jj = func(x, *params_jj)
            resid = (w_jj - x_eval_jj) / w_err_jj
            ### now plot residuals.
            axes[1].scatter(x, resid, color=f"C{jj}", s=4, alpha=0.3)
        """    
        fig.subplots_adjust(hspace=0, wspace=0)

        #fig.savefig(paths.base_path / f"{obj}_{optical}_{ii}.png")
    
        print("\n")





    param_arr = np.vstack(params_lookup["combined"])



    component_functions = composite_lookup[func.__name__]

    cf1 = component_functions[0]
    cf2 = component_functions[1]
    n_params = composite_n_params.get(func.__name__, (2, 2))

    stop1 = n_params[0]
    stop2 = n_params[0] + n_params[1]

    c_params = (param_arr[:, :stop1], param_arr[:, stop1:stop2])

    markers = ["^", "x"]
    labels = [f"SS, {optical}", f"LS, {optical}"]

    for c_ii, (cf, cp) in enumerate(zip(component_functions, c_params)):

        kwargs = {"marker": markers[c_ii], "label": labels[c_ii], "ls": ls_list[opt_ii]}
        params_axes["amplitude"].errorbar(
            K_lims, cp[:,0], color="k", **kwargs
        )
        if cf.__name__ == "power_law":
            params_axes["slope"].errorbar(
               K_lims, cp[:,1], color="k", **kwargs
            )
        elif cf.__name__ == "exp_law":
            params_axes["cutoff"].errorbar(
               K_lims, cp[:,1], color="k", **kwargs
            )
        elif cf.__name__ == "trunc_power_law":
            params_axes["slope"].errorbar(
               K_lims, cp[:,1], color="k", **kwargs
            )
            params_axes["cutoff"].errorbar(
               K_lims, cp[:,2], color="k", **kwargs
            )

for name, ax in params_axes.items():
    ax.set_ylabel(name)
    ax.set_xlabel(r"Limiting $K_{AB}$")
    ax.legend()


params_fig.subplots_adjust(wspace=0.1)



#params_fig.savefig(paths.base_path / "drgs_params.png")

"""
for jj, field in enumerate(fields):
    if field in missing_fields:
        continue
    if len(params_lookup[field]) != len(K_lims):
        continue
    param_arr_jj = np.vstack(params_lookup[field])
    params_axes[0].scatter(K_lims, param_arr_jj[:,0], color=f"C{jj}", marker="x", alpha=0.3)
    params_axes[0].scatter(K_lims, param_arr_jj[:,2], color=f"C{jj}", marker="^", alpha=0.3)

    params_axes[1].scatter(K_lims, param_arr_jj[:,1], color=f"C{jj}", marker="x", alpha=0.3)
    params_axes[1].scatter(K_lims, param_arr_jj[:,3], color=f"C{jj}", marker="^", alpha=0.3)
"""


"""
crossing_fig, crossing_ax = plt.subplots()
th_c = theta_cross["combined"]
crossing_ax.errorbar(K_lims, th_c, color="k", marker="o", zorder=10)

crossing_ax.semilogy()
crossing_ax.set_ylim(1e-3, 1e0)

crossing_ax.set_ylabel(r"$\theta_{\mathrm{cross}}$")
crossing_ax.set_xlabel("Limiting $K_{AB}$")

for jj, field in enumerate(fields):
    th_c = theta_cross[field]

    crossing_ax.scatter(K_lims, th_c, color=f"C{jj}", alpha=0.3)
"""
plt.show()




