import yaml

from easyquery import Query

from dxs import paths

survey_config_path = paths.config_paths / "surve_config.yaml"

def get_dM(band):
    """dM is defined st. m_AB = m_Vega + offset"""
    return dM_lookup[band]

def get_f0(band):
    return f0_lookup[band]

def ab_to_vega(mag_ab, dM=None, band=None):
    """dM is defined st. m_AB = m_Vega + offset"""
    if band is not None:
        dM = get_dM(band)
    if dM is None:
        raise ValueError("dM cannot be None - provide dM [float] or band [str]")
    return mag_ab - dM

def vega_to_ab(mag_v, dM=None, band=None):
    """dM is defined st. m_AB = m_Vega + offset"""
    if band is not None:
        dM = get_dM(band)
    if dM is None:
        raise ValueError("dM cannot be None - provide dM [float] or band [str]")
    return mag_v + dM

def vega_to_flux(mag, f0=None, band=None):
    if band is not None:
        get_f0(band)
    if f0 is None:
        raise ValueError("f0 cannot be None - provide dM [float] or band [str]")
    return f0*10**(-mag/2.5)

def ab_to_flux(mag, f0):
    raise NotImplementedError


def select_starforming(
    catalog, g: str, z: str, K: str, use_vega=True, g_lim=99., z_lim=99., K_lim=99., zK_offset=0.0
):
    pass



