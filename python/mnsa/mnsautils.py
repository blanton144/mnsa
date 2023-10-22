import numpy as np


# encoding: utf-8

import astropy.cosmology
import astropy.units


cosmo = astropy.cosmology.Planck18


def log_flux_to_luminosity(redshift=None):
    """Return term to turn log flux to log luminosity

    Parameters
    ----------

    redshift : np.float32 or ndarray of np.flaot32
        redshift of galaxy or galaxies

    Returns
    -------

    logterm : np.float32 or ndarray of np.float32
        term to add to flux to get luminosity
"""
    logterm = - 17.
    dm = cosmo.distmod(redshift).to_value(astropy.units.mag)
    log_dfactor = 0.4 * dm
    log_10pcfactor = np.log10(4. * np.pi) + 2. * (np.log10(3.086) + 19.)
    logterm = logterm + log_10pcfactor + log_dfactor
    return(logterm)
