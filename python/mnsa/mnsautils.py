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

    Notes
    -----

    Assumes flux is in 1.e-17 * Energy / Time / cm^2 units
"""
    logterm = - 17.
    dm = cosmo.distmod(redshift).to_value(astropy.units.mag)
    log_dfactor = 0.4 * dm
    log_10pcfactor = np.log10(4. * np.pi) + 2. * (np.log10(3.086) + 19.)
    logterm = logterm + log_10pcfactor + log_dfactor
    return(logterm)


def in_sample_galaxy(drpall=None):
    """Return if in the Primary, Secondary or Color-Enhanced samples

    Parameters
    ----------

    drpall : ndarray
        DRP summary structure

    Returns
    -------

    in_sample : ndarray of bool
        Whether in sample
"""
    in_sample = (((drpall['mngtarg1'] & 2**10) != 0) |
                 ((drpall['mngtarg1'] & 2**11) != 0) |
                 ((drpall['mngtarg1'] & 2**12) != 0))
    return(in_sample)
