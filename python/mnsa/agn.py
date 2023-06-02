import numpy as np
import scipy.interpolate


w1w2_grid = np.array([-0.5, -0.5, -0.3, -0.3], dtype=np.float32)
logssfr_grid = np.array([-20., -11., -10., 20], dtype=np.float32)
w1w2_interp = scipy.interpolate.interp1d(logssfr_grid, w1w2_grid,
                                         bounds_error=False,
                                         fill_value='extrapolate')


def w1w2_condition(logssfr=None):
    """Return condition on W1-W2 given galaxy properties

    Parameters
    ----------

    logssfr : np.float32, or ndarray of np.float32
        log10 of specific SFR

    Returns
    -------

    w1w2 : np.float32, or ndarray of np.float32
        condition(s) on W1-W2 (redder than condition means AGN)
"""
    w1w2 = w1w2_interp(logssfr)
    return(w1w2)


def absmag_to_lognuLnu(absmag=None, wave=None):
    """Convert an absolute magnitude to log nu L_nu

    Parameters
    ----------

    absmag : np.float32
        absolute magnitude (AB)

    wave : np.float32
        wavelength (Angstrom)

    Returns
    -------

    lognulnu : np.float32
       log_10 nu L_nu for this observation
"""
    logtenpc2_to_cm2 = 2. * (np.log10(3.086) + 19.)
    logcspeed = np.log10(2.99792) + 18.  # Ang/s

    logabsmaggies = (- 0.4 * absmag)
    logLnu = logabsmaggies + np.log10(3631.) + logtenpc2_to_cm2 + np.log10(4. * np.pi) - 23.
    lognu = logcspeed - np.log10(wave)
    lognuLnu = lognu + logLnu
    return(lognuLnu)


def agn_luminosity_w2w3(sps=None, wave=None, spec=None, agnspec=None):
    """Return estimated AGN luminosities in W2 and W3

    Parameters
    ----------

    sps : ndarray
        structure with sps information (including 'redshift' and 'absmag')

    wave : ndarray of np.float32
        wavelengths of spectrum (Angstroms)

    agnspec : ndarray of np.float32
        AGN spectrum in erg/cm^2/s/A

    spec : ndarray of np.float32
        spectrum in erg/cm^2/s/A

    Returns
    -------

    lagn : dict
        dictionary with 'log_nuLnu_w2', 'log_nuLnu_w3' in erg/s
"""
    spec_interp = scipy.interpolate.interp1d(wave, spec)
    agnspec_interp = scipy.interpolate.interp1d(wave, agnspec)

    lagn = dict()

    w2_wave = 34000.
    log_nuLnu_w2_total = absmag_to_lognuLnu(absmag=sps['absmag'][5],
                                            wave=w2_wave)
    w2_agn = agnspec_interp(w2_wave)
    w2_spec = spec_interp(w2_wave)
    w2_agnfrac = w2_agn / w2_spec
    if(w2_agnfrac > 0):
        lagn['log_nuLnu_w2'] = np.log10(w2_agnfrac) + log_nuLnu_w2_total
    else:
        lagn['log_nuLnu_w2'] = -9999.

    w3_wave = 46000.
    log_nuLnu_w3_total = absmag_to_lognuLnu(absmag=sps['absmag'][6],
                                            wave=w3_wave)
    w3_agn = agnspec_interp(w3_wave)
    w3_spec = spec_interp(w3_wave)
    w3_agnfrac = w3_agn / w3_spec
    if(w3_agnfrac > 0):
        lagn['log_nuLnu_w3'] = np.log10(w3_agnfrac) + log_nuLnu_w3_total
    else:
        lagn['log_nuLnu_w3'] = -9999.

    return(lagn)
