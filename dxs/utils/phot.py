import yaml

from easyquery import Query

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

dM_lookup = survey_config["ab_vega_offset"]

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



