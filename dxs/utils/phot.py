from dxs import paths

def ab_to_vega(mag, dM):
    return mag + dM

def vega_to_ab(mag, dM):
    return mag - dM

def vega_to_flux(mag, f0):
    return f0*10**(-mag/2.5)

def ab_to_flux(mag, f0):
    raise NotImplementedError
