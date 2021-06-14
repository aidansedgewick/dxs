import time

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

    return phi_Cirasulo, 

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

def prepare_components(catalog, randoms, treecorr_config, catalog2=None, randoms2=None):

    if "num_threads" not in treecorr_config:
        treecorr_config["num_threads"] = 3
    
    t1 = time.time()
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
    t2 = time.time()
    print(f"DD in {(t2-t1):.2f}s")

    t1 = time.time()
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
    t2 = time.time()
    print(f"RR in {(t2-t1):.2f}s")

    t1 = time.time()
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
    t2 = time.time()
    print(f"DR/RD in {(t2-t1):.2f}s")


    w_ls, w_lserr = dd.calculateXi(rr=rr, dr=dr, rd=rd)

    return dd, dr, rd, rr

    

    
