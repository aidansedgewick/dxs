import time
from typing import List

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from treecorr import NNCorrelation, Catalog

from dxs.utils.image import uniform_sphere, calc_survey_area, objects_in_coverage

###======================= LUMINOSITY FUNCTIONS ===========================###

def calc_absM(app_mag, distmod=None, z_phot=None, cosmol=None):
    if distmod is not None:
        absM = app_mag - distmod
    else:
        if z_phot is not None:
            if cosmol is None:
                raise ValueError("provide z_phot AND astropy.cosmol object")
            k_corr = 2.5 * np.log10(1. + z_phot)
            distmod = cosmol.distmod(z_phot).value
            absM = app_mag - distmod + k_corr
        else:
            raise ValueError("provide distmod, or z_phot and astropy.cosmol object")
    return absM

def r_at_ref_mag(ref_mag, absM, cosmol):
    z_grid = np.logspace(0, 1.5, 1001)[1:] - 1.
    dL_grid = cosmol.luminosity_distance(z_grid).to(u.pc)

    z_func_vals = ref_mag - (5. * np.log10(dL_grid / (10. * u.pc))) + 2.5 * np.log10(1. + z_grid)
    
    diff_vals = abs(z_func_vals[:, None] - absM) # Each row is a z_max point -- cols are gals.
    z_idxs = np.argmin(diff_vals, axis=0)
    z_vals = z_grid[z_idxs]

    return z_vals

def calc_vmax(z_min, z_max, omega, cosmol):
    rmax = cosmol.comoving_distance(z_max)
    rmin = cosmol.comoving_distance(z_min)

    return omega / 3 * (rmax ** 3 - rmin ** 3)

def calc_luminosity_function(absM, vmax, absM_bins, bin_width=None):
    if bin_width is None:
        width_vals = np.diff(absM_bins)
        bin_width = width_vals[0]
        if not np.allclose(width_vals, bin_width):
            raise ValueError("bins should be constant width")
    weights = 1. / vmax
    phi,_ = np.histogram(absM, weights=weights, bins=absM_bins)

    phi_err = phi / 4.
    
    return phi, phi_err
    

def schechter(M, phi_star, M_star, alpha):
    delta_M = M - M_star
    x = -0.4 * delta_M * (alpha + 1.)
    return 0.4 * np.log(10.) * phi_star * (10 ** x) * np.exp(-10**(-0.4 * delta_M))

def Mk_z_Cirasulo(z):
    return -22.26 - (z / 1.78) ** 0.47

def phi0_z_Cirasulo(z):
    return 3.5e-3 * np.exp( -(z / 1.7) ** 1.47 )

def alpha_z_Cirasulo(z):
    return -1.07

def phi_Cirasulo(z, M_grid):
    cirasulo_params = (phi0_z_Cirasulo(z), Mk_z_Cirasulo(z), alpha_z_Cirasulo(z))
    phi = schechter(M_grid, *cirasulo_params)

    return phi

### ======================== CORRELATION FUNCTIONS ======================== ###

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


def prepare_auto_component(
    treecorr_config, cat=None, ra=None, dec=None, npatch=None, patch_centers=None, cat_only=False
):
    t1 = time.time()
    if npatch is not None and patch_centers is not None:
        raise ValueError(f"can't provide both 'npatch' and 'patch_centers'")
    patch_kwargs = {}
    if npatch is not None:
        print(f"create {npatch} patches")
        patch_kwargs["npatch"] = npatch
    if patch_centers is not None:
        print(f"use existing patch_centers")
        patch_kwargs["patch_centers"] = patch_centers
    if npatch is not None:
        patch_kwargs["npatch"] = npatch

    if cat is None:
        if ra is None or dec is None:
            raise ValueError("if cat is None, must provide ra & dec!")
        cat = Catalog(
            ra=ra, dec=dec, ra_units="deg", dec_units="deg", **patch_kwargs
        )
    if cat_only:
        print("skipping process...")
        return cat, None
    nn = NNCorrelation(treecorr_config)
    nn.process(cat)
    t2 = time.time()
    print(f"component done in {(t2-t1):.2f}s")
    return cat, nn

def prepare_cross_component(
    treecorr_config, cat1=None, cat2=None, 
    ra1=None, ra2=None, dec1=None, dec2=None, 
    npatch=None, patch_centers=None
):
    t1 = time.time()
    if npatch is not None and patch_centers is not None:
        raise ValueError(f"can't provide both 'npatch' and 'patch_centers'")
    patch_kwargs = {}
    if npatch is not None:
        print(f"create {npatch} patches")
        patch_kwargs["npatch"] = npatch
    if patch_centers is not None:
        print(f"use existing patch_centers")
        patch_kwargs["patch_centers"] = patch_centers

    if cat1 is None:
        if ra1 is None or dec1 is None:
            raise ValueError("if cat1 is None, must provide ra1 & dec1!")
        cat1 = Catalog(
            ra=ra1, dec=dec1, patch_centers=patch_centers, npatch=npatch, 
            ra_units="deg", dec_units="deg",
        )

    if cat2 is None:
        if ra2 is None or dec2 is None:
            raise ValueError("if cat2 is None, must provide ra2 & dec2!")   
        cat2 = Catalog(
            ra=ra2, dec=dec2, patch_centers=cat1.patch_centers, npatch=npatch,
            ra_units="deg", dec_units="deg",
        )

    nn = NNCorrelation(treecorr_config)
    nn.process(cat1, cat2)
    t2 = time.time()
    print(f"component done in {(t2-t1):.2f}s")
    return cat1, cat2, nn

def get_jackknife_component(NN: NNCorrelation, ii_jackknife):
    nn_pairs = sum(x.npairs for key, x in NN.results.items() if ii_jackknife not in key)
    nn_tot = sum(x.tot for key, x in NN.results.items() if ii_jackknife not in key)
    NN_jk = nn_pairs / nn_tot

    return nn_pairs, nn_tot

def w_mean_from_NN_list(NN_dd_list: List[NNCorrelation], return_components=False):
    if not isinstance(NN_dd_list, list):
        NN_dd_list = [NN_dd_list]

    ### datas
    dd_pairs = sum([NN_dd.npairs for NN_dd in NN_dd_list])
    dd_tot = sum([NN_dd.tot for NN_dd in NN_dd_list])
    DD = dd_pairs / dd_tot
    ### data-randoms
    dr_pairs = sum([NN_dd._dr.npairs for NN_dd in NN_dd_list])
    dr_tot = sum([NN_dd._dr.tot for NN_dd in NN_dd_list])
    DR = dr_pairs / dr_tot
    ### random-datas
    RD = None
    ### randoms
    rr_pairs = sum([NN_dd._rr.npairs for NN_dd in NN_dd_list])
    rr_tot = sum([NN_dd._rr.tot for NN_dd in NN_dd_list])
    RR = rr_pairs / rr_tot

    w_mean = (DD - 2. * DR + RR) / RR    

    if return_components:
        return w_mean, DD, DR, RD, RR
    else:
        return w_mean

def jackknife_cov_from_NN_list(NN_dd_list: List[NNCorrelation]):
    if not isinstance(NN_dd_list, list):
        NN_dd_list = [NN_dd_list]
    w_mean, DD, DR, RD, RR = w_mean_from_NN_list(NN_dd_list, return_components=True)
    
    """
    w_jk_estimates = []
    DD_jk_list, DR_jk_list, RR_jk_list = [], [], []
    for NN_dd in NN_dd_list:
        for ii_jackknife in range(NN_dd.npatch1):
            DD_jk = get_jackknife_component(NN_dd, ii_jackknife)
            RR_jk = get_jackknife_component(NN_dd._rr, ii_jackknife)
            if NN_dd._dr is not None:
                DR_jk = get_jackknife_component(NN_dd._dr, ii_jackknife)
            else:
                DR_jk = None
            w_jk = (DD_jk - 2. * DR_jk + RR_jk) / RR_jk          
            w_jk_estimates.append(w_jk)

            DR_jk_list.append(DR_jk)
    """

    ### datas
    dd_pairs = [NN_dd.npairs for NN_dd in NN_dd_list]
    dd_tot = [NN_dd.tot for NN_dd in NN_dd_list]
    DD = sum(dd_pairs) / sum(dd_tot)
    ### data-randoms
    dr_pairs = [NN_dd._dr.npairs for NN_dd in NN_dd_list]
    dr_tot = [NN_dd._dr.tot for NN_dd in NN_dd_list]
    DR = sum(dr_pairs) / sum(dr_tot)
    ### random-datas
    RD = None
    ### randoms
    rr_pairs = [NN_dd._rr.npairs for NN_dd in NN_dd_list]
    rr_tot = [NN_dd._rr.tot for NN_dd in NN_dd_list]
    RR = sum(rr_pairs) / sum(rr_tot)

    w_jk_estimates = []
    DD_jk_list, DR_jk_list, RR_jk_list = [], [], []

    for jj, NN_dd in enumerate(NN_dd_list):
        other_dd_pairs = sum([x for ii, x in enumerate(dd_pairs) if ii != jj])
        other_dd_tot = sum([x for ii, x in enumerate(dd_tot) if ii != jj])
        other_dr_pairs = sum([x for ii, x in enumerate(dr_pairs) if ii != jj])
        other_dr_tot = sum([x for ii, x in enumerate(dr_tot) if ii != jj])
        other_rr_pairs = sum([x for ii, x in enumerate(rr_pairs) if ii != jj])
        other_rr_tot = sum([x for ii, x in enumerate(rr_tot) if ii != jj])

        for ii_jackknife in range(NN_dd.npatch1):
            dd_jk_pairs, dd_tot_jk = get_jackknife_component(NN_dd, ii_jackknife)
            DD_jk = (dd_jk_pairs + other_dd_pairs) / (dd_tot_jk + other_dd_tot)
            dr_jk_pairs, dr_tot_jk = get_jackknife_component(NN_dd._dr, ii_jackknife)
            DR_jk = (dr_jk_pairs + other_dr_pairs) / (dr_tot_jk + other_dr_tot)
            rr_jk_pairs, rr_tot_jk = get_jackknife_component(NN_dd._rr, ii_jackknife)
            RR_jk = (rr_jk_pairs + other_rr_pairs) / (rr_tot_jk + other_rr_tot)

            w_jk = (DD_jk - 2. * DR_jk + RR_jk) / RR_jk
            w_jk_estimates.append(w_jk)
            DR_jk_list.append(DR_jk)

    N = len(w_jk_estimates)

    delta_w = [w_jk - w_mean for w_jk in w_jk_estimates]
    #for ii, dw in enumerate(delta_w):
    #    print(ii, dw)
    cov = (N - 1.) / N * sum([np.outer(dw, dw) for dw in delta_w])

    cov_w_err = np.sqrt(cov.diagonal())

    DR_jk_ratios = [DR_jk / DR for DR_jk in DR_jk_list]
    DR_w_err = np.sqrt(
        sum([r_DR * dw * dw for r_DR, dw in zip(DR_jk_ratios, delta_w)])
    )

    print(cov_w_err)
    print(DR_w_err)

    return cov, DR_w_err


"""
def prepare_components(
    catalog, randoms, treecorr_config, catalog2=None, randoms2=None, rr=None
):

    if "num_threads" not in treecorr_config:
        treecorr_config["num_threads"] = 3

    ###============== do RR stuff ===============###    
    t1 = time.time()
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

    if rr is None:        
        rr = NNCorrelation(treecorr_config)
        if use_randoms_catalog2:
            print("##============ process cross RR")
            rr.process(random_catalog, random_catalog2)
        else:
            print("##============ process RR")
            rr.process(random_catalog)
        t2 = time.time()
        print(f"RR in {(t2-t1):.2f}s")
    else:
        if not isinstance(rr, NNCorrelation):
            raise ValueError(f"{rr} should be of type NNCorrelation")
        print("Skip rr, using already processed input catalog...")


    ###============== do DD stuff =============###
    t1 = time.time()
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
    t2 = time.time()
    print(f"DD in {(t2-t1):.2f}s")


    ###============ do DR/RD stuff =============###
    t1 = time.time()
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
    t2 = time.time()
    print(f"DR/RD in {(t2-t1):.2f}s")


    w_ls, w_lserr = dd.calculateXi(rr=rr, dr=dr, rd=rd)

    return dd, dr, rd, rr
"""

def ic_Roche(theta, func, params, RR):    
    if len(RR) != len(theta):
        raise ValueError("len RR should equal len theta")
    w_theta = func(theta, *params)
    
    ic = np.sum(w_theta * RR) / np.sum(RR)
    return ic
    








