import math
import numpy as np
import numpy.matlib as matlib
import scipy.interpolate as interpolate
import os


def image(band=None, wave=None, cube=None):
    """ calculate FWHM for given image

    Parameters:
    ----------

    band : str, 
        band name ('g'/'r'/'i'/'z')

    wave : ndarray of np.float32
        [Nwave] wavelengths

    cube : ndarray of np.float32
        [Nwave, N, M] fluxes in 10^{-17} erg/cm^2/s/A

    Returns:
    -------

    image : ndarray of np.float32
        [N, N] image values in nanomaggies

"""
    filterfile = os.path.join(os.getenv('MNSA_DIR'),
                              'python', 'data', band + '_filter.dat')
    band0 = np.loadtxt(filterfile)
    band1 = np.arange(3400, band0[0, 0], 25)
    band2 = np.arange(band0[-1, 0], 11000, 25)
    weight1 = np.zeros(band1.shape)
    weight2 = np.zeros(band2.shape)
    band = np.concatenate((np.concatenate((band1, band0[:, 0]), axis=0),
                           band2), axis=0)
    weight = np.concatenate((np.concatenate((weight1, band0[:, 1]), axis=0),
                             weight2), axis=0)
    fun_band = interpolate.interp1d(band, weight)
    band_value = fun_band(wave) 

    nWave = len(wave)
    dwave = wave[1:nWave] - wave[0:nWave - 1]
    dwave = np.append(dwave, np.array([dwave[-1]]))

    n = cube.shape[1]
    m = cube.shape[2]
    numer = (matlib.repmat(band_value * dwave, n * m, 1).T *
             (cube.reshape(nWave, n * m))).reshape(nWave, n, m).sum(axis=0)

    abspec_Hz = 3631.e-23 * np.ones(nWave)  # erg / cm^2 / s / Hz
    cspeed = 3e+18   # A/s = A Hz
    abspec_A = (abspec_Hz * cspeed) / wave**2  # erg / cm^2 / s / A
    abspec_A_norm = abspec_A * 1.e+17  # 10^{-17} erg / cm^2 / s / A
    abspec_A_norm = abspec_A_norm * 1.e-9  # To produce nanomaggies
    denom_val = (band_value * dwave * abspec_A_norm).sum()
    denom = np.ones((n, m)) * denom_val

    image = numer / denom

    return(image)
