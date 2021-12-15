import logging
import yaml

from easyquery import Query

from dxs import paths

logger = logging.getLogger("utils_phot")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

dM_lookup = survey_config["ab_vega_offset"]
f0_lookup = survey_config["zeropoint_flux"]
rv_lookup = survey_config["extinction_coeffs"]

def get_dM(band):
    """dM is defined st. m_AB = m_Vega + offset"""
    return dM_lookup[band]

def get_f0(band):
    return f0_lookup[band]

def get_rv(band):
    return rv_lookup.get(band, None)

def ab_to_vega(mag_ab, dM=None, band=None):
    """dM is defined st. m_AB = m_Vega + offset"""
    if isinstance(dM, str):
        raise ValueError(f"\033[31;1mdM must be float.\033[0m did you mean 'band=\"{dM}\"'?")
    if band is not None:
        dM = get_dM(band)
    if dM is None:
        raise ValueError(f"\033[31;1mdM cannot be None.\033[0m provide dM=[float] or band=[str]")
    return mag_ab - dM

def vega_to_ab(mag_v, dM=None, band=None):
    """dM is defined st. m_AB = m_Vega + offset"""
    if isinstance(dM, str):
        raise ValueError(f"\033[31;1mdM must be float.\033[0m did you mean 'band=\"{dM}\"'?")
    if band is not None:
        dM = get_dM(band)
    if dM is None:
        raise ValueError(f"\033[31;1mdM cannot be None.\033[0m provide dM=[float] or band=[str]")
    logger.info(f"add {dM}")
    return mag_v + dM

def vega_to_flux(mag, f0=None, band=None):
    if isinstance(f0, str):
        raise ValueError(f"\033[31;1mf0 must be float.\033[0m did you mean 'band=\"{f0}\"'?")
    if band is not None:
        get_f0(band)
    if f0 is None:
        raise ValueError("\033[31;1mf0 cannot be None.\033[0m provide dM [float] or band [str]")
    return f0*10**(-mag/2.5)

def ab_to_flux(mag, band=None):
    #if band is not None:
    #    logger.info(f"ignore 'band={band}': fv0=3631 for AB")
    return 3631. * 10**(-mag / 2.5)
    #raise NotImplementedError

def apply_extinction(mag, ebv, band=None, rv=None):
    if rv is None:
        if band is None:
            raise ValueError("Must pass band, eg. band='i'")
        rv = get_rv(band)
    if rv is None:
        raise ValueError("no rv {band} value in survey_config")
    return mag - rv * ebv # make magnitude BRIGHTER.

"""
def select_starforming(
    catalog, gmag: str, zmag: str, Kmag: str, transform_to_AB=True, g_lim=99., z_lim=99., K_lim=99., zK_offset=0.0
):
    if transform_to_AB:
        catalog[gmag] = vega_to_ab(catalog[gmag], band="g")
        catalog[zmag] = vega_to_ab(catalog[zmag], band="z")
        catalog[Kmag] = vega_to_ab(catalog[Kmag], band="K")
    qstr = f"({zmag}-{Kmag}) >= -0.022"
    print(qstr)
    queries = (qstr)
    print(queries)
    output = Query(*queries).filter(catalog)
    if transform_to_AB:
        output[gmag] = ab_to_vega(output[gmag], band="g")
        output[zmag] = ab_to_vega(output[zmag], band="z")
        output[Kmag] = ab_to_vega(output[Kmag], band="K")
    return output

def select_passive(
    catalog, g: str, z: str, K: str, transform_to_AB=True, g_lim=99., z_lim=99., K_lim=99., zK_offset=0.0
):
    if transform_to_AB:
        catalog[gmag] = vega_to_ab(catalog[gmag], "g")
        catalog[zmag] = vega_to_ab(catalog[zmag], "z")
        catalog[Kmag] = vega_to_ab(catalog[Kmag], "K")
    qstr = f"{zmag} - {Kmag} - 1.27 * {gmag} + 1.27 * {zmag} >= -0.022"
    queries = (qstr, f"{zmag} - {Kmag} >= 2.55")
    output = Query(*queries).filter(catalog)
    if transform_to_AB:
        output[gmag] = ab_to_vega(output[gmag], "g")
        output[zmag] = ab_to_vega(output[zmag], "z")
        output[Kmag] = ab_to_vega(output[Kmag], "K")
    return output"""



