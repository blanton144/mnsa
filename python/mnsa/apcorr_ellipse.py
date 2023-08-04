import numpy as np
import scipy.interpolate
import astropy.convolution as convolution


def resample_psf(psf_in=None, psf_limit=10., pixscale_in=None,
                 n_out=None, pixscale_out=None):
    """Resample a PSF on a different pixel scale

    Parameters
    ----------

    psf_in : ndarray of np.float32
        PSF to resample

    psf_limit : np.float32 or float
        large radius cutoff to apply

    pixscale_in : np.float32 or float
        pixel scale for psf_in

    pixscale_out : np.float32 or float
        pixel scale for output PSF

    n_out : np.int32 or int
        size of output PSF
"""
    psf_in = psf_in / psf_in.sum()
    xpsf = np.arange(psf_in.shape[1])
    ypsf = np.arange(psf_in.shape[0])
    xxpsf = np.outer(np.ones(psf_in.shape[0]), xpsf)
    yypsf = np.outer(ypsf, np.ones(psf_in.shape[1]))
    rrpsf = np.sqrt((xxpsf - (psf_in.shape[1] // 2))**2 +
                    (yypsf - (psf_in.shape[0] // 2))**2)
    psf_in = psf_in * (rrpsf < psf_limit)  # based on where PSF gets noisy
    psf_in = psf_in / psf_in.sum()

    # psf_interp takes X and Y in PSF image pixel units
    psf_interp = scipy.interpolate.RectBivariateSpline(xpsf, ypsf,
                                                       psf_in)

    # xxpsf_out and yypsf_out are in MaNGA pixel units
    xpsf_out = np.arange(n_out, dtype=np.float32)
    ypsf_out = np.arange(n_out, dtype=np.float32)
    ocx = np.float32(n_out // 2)
    ocy = np.float32(n_out // 2)
    xxpsf_out = np.outer(np.ones(n_out), xpsf_out).flatten() - ocx
    yypsf_out = np.outer(ypsf_out, np.ones(n_out)).flatten() - ocy

    # lx, ly are converted to PSF image pixel units
    cx = np.float32(psf_in.shape[1] // 2)
    cy = np.float32(psf_in.shape[0] // 2)
    lx = xxpsf_out * pixscale_out / pixscale_in
    ly = yypsf_out * pixscale_out / pixscale_in

    # psf_out is the PSF desired, in MaNGA pixel space
    psf_out = psf_interp(lx + cx, ly + cy, grid=False).flatten()
    psf_out = psf_out.reshape(n_out, n_out)
    psf_out = psf_out * (pixscale_in / pixscale_out)**2

    return(psf_out)


def elliptical_sma(nx=None, ny=None, ba=1., phi=0., xcen=None, ycen=None):
    """Create image of semi-major axis distance

    Parameters
    ----------

    nx : int
        x size of output image

    ny : int
        y size of output image

    ba : float
        axis ratio (b/a) between 0 and 1

    phi : float
        position angle in deg

    xcen : float
        X center to use

    ycen : float
        Y center to use

    Returns
    -------

    sma : ndarray of np.float32
        elliptical distance to each pixel
"""
    x = np.outer(np.ones(nx), np.arange(ny))
    x = x - xcen
    y = np.outer(np.arange(nx), np.ones(ny))
    y = y - ycen
    xp = ((np.cos(np.pi / 180. * phi) * x +
           np.sin(np.pi / 180. * phi) * y))
    yp = (- np.sin(np.pi / 180. * phi) * x +
          np.cos(np.pi / 180. * phi) * y)
    return(np.sqrt(yp**2 + (xp / ba)**2))


def cog2image(sma=None, cog=None, nx=None, ny=None,
              ba=1., phi=0., xcen=None, ycen=None):
    """Converts an elliptical curve of growth an image.

    Parameters
    ----------
    sma : np.float32
        1-D ndarray with semimajor axies in pixels

    cog : np.float32
        1-D ndarray with flux internal to each aperture

    nx : int
        x size of output image

    ny : int
        y size of output image

    ba : float
        axis ratio (b/a) between 0.2 and 1 (default 1)

    phi : float
        position angle of major exis in deg (default 0)

    xcen : float
        X center to use (default float(nx)*0.5)

    ycen : float
        Y center to use (default float(ny)*0.5)

    Returns
    -------

    image : np.float32
        [nx, ny] 2-D array with image in it

    Notes
    -----

    The resulting image has an axis ratio and position angle
    as input, with the given curve of growth.

    If ba < 0.2, then 0.2 is used.

    For a North-up (+y), East-left (-x) image, position angle
    phi definition corresponds to astronomical standard
    (deg East of North).
"""
    if((nx % 2) == 0):
        raise ValueError("nx must be odd")

    if((ny % 2) == 0):
        raise ValueError("ny must be odd")

    # Interpret input
    if(xcen is None):
        xcen = np.float32(nx // 2)
    if(ycen is None):
        ycen = np.float32(ny // 2)
    if(ba > 0.2):
        ba_use = ba
    else:
        ba_use = 0.2

    # Create radius array defining how far pixels are from
    # center, accounting for axis ratio.
    smaimage = elliptical_sma(nx=nx, ny=ny, ba=ba_use, phi=phi,
                              xcen=xcen, ycen=ycen)

    # Now create image
    tmp_sma = np.append(np.array([0.], dtype=sma.dtype), sma)
    tmp_cog = np.append(np.array([0.], dtype=sma.dtype), cog)
    grid_sma = 0.5 * (tmp_sma[0:-1] + tmp_sma[1:])
    grid_sb = ((tmp_cog[1:] - tmp_cog[0:-1]) /
               (ba * np.pi * (tmp_sma[1:]**2 - tmp_sma[0:-1]**2)))
    tmp_cog2 = np.append(np.array([cog[0]], dtype=sma.dtype), cog)
    cog_interp = scipy.interpolate.interp1d(tmp_sma, tmp_cog2)
    sb_interp = scipy.interpolate.interp1d(grid_sma, grid_sb)
    indx = np.nonzero((smaimage.flat < max(grid_sma) - 1) &
                      (smaimage.flat != 0.))
    image = np.zeros(nx * ny)
    image[indx] = sb_interp(smaimage.flat[indx])
    image = image.reshape((nx, ny))
    minsma = smaimage.flat[indx].min()
    image[int(xcen), int(ycen)] = cog_interp(minsma)
    image = np.float32(image)
    return image


def apflux(image=None, aprad=None, ba=1., phi=0., xcen=None, ycen=None):
    """Calculates elliptical aperture flux

    Parameters
    ----------

    image : np.float32
        2-D ndarray

    aprad : np.float32
        aperture radius to calculate

    ba : float
        axis ratio (b/a) between 0 and 1 (default 1)

    phi : float
        position angle in deg (default 0)

    xcen : float
        X center to use (default center of image)

    ycen : float
        Y center to use (default center of image)

    Returns
    -------

    flux : np.float32
      File name for the corresponding dust model, or None if non-existent.

    Notes
    -----

    For a North-up (+y), East-left (-x) image, position angle phi
    definition corresponds to astronomical standard (deg East of North).
"""

    # Interpret input
    nx = image.shape[0]
    ny = image.shape[1]
    if(xcen is None):
        xcen = nx // 2
    if(ycen is None):
        ycen = ny // 2
    if(ba > 0.2):
        ba_use = ba
    else:
        ba_use = 0.2

    # Create radius array defining how far pixels are from center,
    # accounting for axis ratio.
    smaimage = elliptical_sma(nx=nx, ny=ny, ba=ba_use, phi=phi,
                              xcen=xcen, ycen=ycen)
    ap = (smaimage < aprad)

    flux = image[ap].sum()

    return flux


def apcorr(sma=None, cog=None, aprad=None, psf=None,
           ba=1., phi=0., apmin=1.e-10, apmax=1.e+10):
    """Calculates an aperture correction given elliptical profile

    Parameters
    ----------

    sma : np.float32 array
        1-D ndarray with semi-major radius in pixels

    cog : np.float32 array
        1-D ndarray with flux enclosed at each radius

    aprad : int
        aperture radius, in pixels

    psf : np.float32 array
        2-D ndarray with PSF to calculate for

    ba : float
        axis ratio (b/a) between 0 and 1 (default 1)

    phi : float
        major axis position angle in deg (default 0)

    apmin : float
        minimum aperture correction (default 1.e-10)

    apmax : float
        maximum aperture correction (default 1.e+10)

    Returns
    -------

    apcorr : np.float32
        correction factor to apply to PSF-convolved flux for this aperture

    Notes
    -----

    For a North-up (+y), East-left (-x) image, position angle phi
    definition corresponds to astronomical standard
    (deg East of North).
"""

    # Make image
    factor = 1.5
    nx = int(aprad * 2. * factor / 2.) * 2 + 1
    if(nx < 71):
        nx = 71
    ny = nx
    xcen = nx // 2
    ycen = ny // 2
    image = cog2image(sma=sma, cog=cog, nx=nx, ny=ny,
                      xcen=xcen, ycen=ycen,
                      ba=ba, phi=phi)

    # Measure image
    orig = apflux(image=image, aprad=aprad, ba=ba, phi=phi,
                  xcen=xcen, ycen=ycen)

    # Convolve image with PSF
    psf2 = np.float32(psf)
    cimage = convolution.convolve_fft(image, psf2)

    # Measure convolved image
    convolved = apflux(image=cimage, aprad=aprad, ba=ba, phi=phi,
                       xcen=xcen, ycen=ycen)

    apcorr = orig / convolved

    if(apcorr < apmin):
        apcorr = apmin
    if(apcorr > apmax):
        apcorr = apmax

    del cimage
    del image

    return apcorr
