import os
import numpy as np
import scipy.signal
import scipy.interpolate


class Kernel(object):
    """Kernel values for fiber convolved with seeing

    Parameters
    ----------

    rough_length : float, np.float32
        rough size of grid to use to calculate kernel in arcsec (default 9)

    dkernel : float, np.float32
        pixel scale for this kernel grid in arcsec (default 0.01)

    nseeing : int, np.int32
        number of seeing FWHM values on FWHM grid (default 200)

    minseeing : int, np.int32
        minimum FWHM value on FWHM grid in arcsec (default 0.5)

    maxseeing : int, np.int32
        maximum FWHM value on FWHM grid in arcsec (default 2.5)

    radius : float, np.float32
        radius of the fiber in arcsec (default 1.0)

    Attributes
    ----------

    rough_length : float, np.float32
        rough size of grid to use to calculate kernel in arcsec

    dkernel : float, np.float32
        pixel scale for this kernel grid in arcsec

    nseeing : int, np.int32
        number of FWHM values on FWHM grid

    minseeing : int, np.int32
        minimum FWHM value on FWHM grid

    maxseeing : int, np.int32
        maximum FWHM value on FWHM grid

    xmin : np.float32
        outer edge of lowest X pixel

    ymin : np.float32
        outer edge of lowest Y pixel

    xmax : np.float32
        outer edge of highest X pixel

    ymax : np.float32
        outer edge of highest Y pixel

    length : np.float32
        length of kernel grid in arcsec (based on outer pixel edges)

    x2k : 2D ndarray of np.float32
        array of X positions on grid

    y2k : 2D ndarray of np.float32
        array of Y positions on grid

    xk : ndarray of np.float32
        1D array of X positions

    yk : ndarray of np.float32
        1D array of Y positions

    fiber : 2D ndarray of np.float32
        image of fiber

    seeing : ndarray of np.float32
        grid of FWHM values

    kernel : 3D ndarray of np.float32
        [nseeing, nkernel, nkernel] grid of 2D kernel images (in per arcsec units)

    nradial : np.int32
        number of radial locations for radial kernel grid

    d_radial : np.float32
        radial grid spacing in arcsec

    r_radial : ndarray of np.float32
        [nradial] radial grid locations in arcsec

    kernel_radial : 2D ndarray of np.float32
        [nseeing, nradial] grid of radial profiles (in per arcsec units)

    Notes
    -----

    All of the distance units are technically arbitrary but for use
    with MaNGA data should be treated as in arcsec. All kernel values
    are in per arcsec^2 units.

    The kernel is a fiber of radius 1 arcsec, convolved with the
    atmospheric seeing.

    The atmospheric seeing is modeled as a double Gaussian. The narrow
    Gaussian has sigma1 = FWHM / 2.355 / 1.05. The broad Gaussian has
    sigma2 = 2 * sigma1, and has a total integral 1/9 of the narrow
    one.

    Examples
    --------

    The use of this function will typically be to set the kernel:

    k = kernel.Kernel()

    The kernel object will take a few seconds to create under the 
    default settings.

    Then you can output the radial function for some FWHM:

    rs = np.arange(100) / 100. * 5.
    ks = k.radial(0.6, rs)

    The kernel is normalized in per unit arcsec^2 so this quantity
    should be near unity for this choice of rs:

    2. * np.pi * (rs* ks).sum() * (5. / 100.)

    (Why 5./100. and not the square of that? Because you are already
    multiplying by arcsec in the radial dimension, by multiplying by
    rs in the ().sum() term).
"""

    def __init__(self, rough_length=9., dkernel=0.01, nseeing=200,
                 minseeing=0.5, maxseeing=2.5, radius=1.0, extra=None):
        self.rough_length = rough_length
        self.dkernel = dkernel
        self.extra = extra
        self.radius = radius
        self._set_kernel_grid(nseeing=nseeing, minseeing=minseeing,
                              maxseeing=maxseeing)
        self._set_kernel_radial_grid()
        return

    def _create_grid(self, rough_length=None, d=None):
        """Create grid corresponding to length and pixel size"""
        nside = ((np.int32(np.ceil(rough_length / d)) // 2) * 2 + 1)
        length = np.float32(nside * d)
        self.xmin = -0.5 * length
        self.xmax = 0.5 * length
        self.ymin = -0.5 * length
        self.ymax = 0.5 * length
        xi = np.linspace(self.xmin + 0.5 * d, self.xmax - 0.5 * d, nside,
                         dtype=np.float32)
        yi = np.linspace(self.ymin + 0.5 * d, self.ymax - 0.5 * d, nside,
                         dtype=np.float32)
        x2i, y2i = np.meshgrid(xi, yi)
        return (nside, length, x2i, y2i, xi, yi)

    def _psf(self, seeing=None, x=None, y=None):
        """PSF values at given x and y locations

        Parameters
        ----------

        seeing : float, float32
            FWHM of PSF to assume

        x : float32 or ndarray of float32
            x position(s)

        y : float32 or ndarray of float32
            y position(s)

        Returns
        -------

        vals : ndarray of float32
            value of normalized PSF at requested locations

        Notes
        -----

        This code assumes a double Gaussian, with one Gaussian twice
        the wide of the other and with a total integral of 1/9 the
        size. This is a rough approximation to atmospheric seeing.
"""
        sig1 = seeing / 2.355 / 1.05  # factor 1.05 to handle second Gaussian
        sig2 = 2. * sig1
        gaus1 = (np.exp(- (x**2 + y**2) / (2 * sig1**2)) /
                 (2 * np.pi * sig1**2))
        gaus2 = (np.exp(- (x**2 + y**2) / (2 * sig2**2)) /
                 (2 * np.pi * sig2**2))
        scale21 = 1. / 9.
        gaus = (gaus1 + scale21 * gaus2) / (1. + scale21)
        return gaus

    def _set_kernel_grid(self, nseeing=100, minseeing=0.5, maxseeing=2.5):
        """Create the kernel for each wavelength"""

        self.nseeing = nseeing
        self.minseeing = minseeing
        self.maxseeing = maxseeing

        (self.nkernel, self.length, self.x2k, self.y2k, self.xkernel,
         self.ykernel) = self._create_grid(self.rough_length, self.dkernel)

        # Create the fiber image (just a circular aperture)
        radius = np.sqrt(self.x2k**2 + self.y2k**2)
        fiber = np.zeros([self.nkernel, self.nkernel], dtype=np.float32)
        ifiber = np.where(radius < self.radius)
        fiber[ifiber] = 1.
        self.fiber = fiber / fiber.sum()

        # Now convolve with the PSF of various widths
        self.seeing = minseeing + ((maxseeing - minseeing) * np.arange(nseeing) /
                                   np.float32(nseeing - 1))
        nseeing = len(self.seeing)
        self.kernel = np.zeros([nseeing, self.nkernel, self.nkernel],
                               dtype=np.float32)
        for index, iseeing in enumerate(self.seeing):
            psf0 = self._psf(iseeing, self.x2k, self.y2k)
            self.kernel[index, :, :] = scipy.signal.fftconvolve(self.fiber,
                                                                psf0,
                                                                mode='same')
        if(self.extra is not None):
            norm = 1. / (2. * np.pi * self.extra**2) * self.dkernel**2
            r2k2 = self.x2k**2 + self.y2k**2
            psf_extra = norm * np.exp(- 0.5 * r2k2 / self.extra**2)
            for index, iseeing in enumerate(self.seeing):
                self.kernel[index, :, :] = scipy.signal.fftconvolve(self.kernel[index, :, :],
                                                                    psf_extra,
                                                                    mode='same')
        return

    def _set_kernel_radial_grid(self):
        """Create the radial kernel for each wavelength"""

        # Create the fiber image (just a circular aperture)
        radii = np.sqrt(self.x2k**2 + self.y2k**2)
        uradii, uindex = np.unique(radii, return_index=True)

        self.nradial = 1000
        kernel_radial_min = 0.
        kernel_radial_max = 6.
        self.d_radial = ((kernel_radial_max - kernel_radial_min) /
                          np.float32(self.nradial - 1))
        self.r_radial = (kernel_radial_min +
                         np.arange(self.nradial) * self.d_radial)
        self.kernel_radial = np.zeros((len(self.seeing), self.nradial),
                                      dtype=np.float32)
        for index, iseeing in enumerate(self.seeing):
            tmp_radial = self.kernel[index, :, :].flatten()[uindex]
            tmp_radial_interp = scipy.interpolate.interp1d(uradii, tmp_radial,
                                                           kind='linear',
                                                           bounds_error=False,
                                                           fill_value=0.,
                                                           assume_sorted=True)
            self.kernel_radial[index, :] = tmp_radial_interp(self.r_radial)

        self._radial_function = (
            scipy.interpolate.interp2d(self.seeing, self.r_radial,
                                       self.kernel_radial.transpose(),
                                       kind='cubic', bounds_error=False,
                                       fill_value=None))

        return

    def radial(self, seeing=None, radii=None):
        """Kernel values at various radii for some FWHM

        Parameters
        ----------

        seeing : float, float32
            FWHM of seeing to assume

        radii : ndarray of float32
            radial values to evaluate

        Returns
        -------

        vals : ndarray of float32
            kernel values at radii
"""
        isort = np.argsort(radii)
        tvals = self._radial_function(seeing, radii[isort]).flatten()
        vals = np.zeros(len(radii), dtype=np.float32)
        vals[isort] = tvals
        return(vals)

    def fwhm(self, seeing=None):
        """FWHM of kernel associated with a given seeing FWHM

        Parameters
        ----------

        seeing : float, float32
            FWHM of seeing

        Returns
        -------

        fwhm : float
            full-width half maximum of kernel
"""
        rs = self.r_radial
        vs = self._radial_function(seeing, rs).flatten()
        rs = np.flip(rs)
        vs = np.flip(vs)
        vmax = vs.max()
        rinterp = scipy.interpolate.interp1d(vs, rs,
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=0.,
                                             assume_sorted=True)
        fwhm = 2. * rinterp(vmax * 0.5)

        return(fwhm)
