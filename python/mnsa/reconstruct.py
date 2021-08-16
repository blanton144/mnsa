import math
import numpy as np
import numpy.matlib as matlib
import scipy.interpolate as interpolate
import marvin
import marvin.tools.rss as rss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from pydl.pydlutils.sdss import sdss_flagval
from scipy import sparse
import mnsa.kernel
import os

# Set Marvin configuration so it gets everything local
marvin.config.setRelease('DR15')
marvin.config.mode = 'local'
marvin.config.download = True


# Rough grid length in arcsec to use for each IFU size
gridsize = {7: 12, 19: 17, 37: 22, 61: 27, 91: 32, 127: 36}


class Reconstruct(object):
    """Base class for reconstruction of cubes from RSS files

    Parameters:
    ----------

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    release : str
        data release (default 'DR15')

    waveindex : int, np.int32
        indices of wavelengths to reconstruct, or None to reconstruct all
        (default None)

    pixelscale : float, np.float32
        pixel scale of output grid in arcsec (default 0.75)

    dkernel : float, np.float32
        underlying resolution of kernel in arcsec (default 0.05)

    verbose : bool
        If True, be verbose

    Attributes:
    ----------

    verbose : bool
        If True, be verbose

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    nfiber : int
        number of fibers

    release : str
        data release

    rss : RSS object
        Marvin RSS output

    waveindex : int, np.int32
        indices of wavelengths to reconstruct

    dkernel : np.float32
        underlying resolution of kernel in arcsec

    nExp : int, np.int32
        number of exposures

    xpos : 2D ndarray of np.float32
        X positions of each fiber at each wavelength [nExp * nfiber, nWave]

    ypos : 2D ndarray of np.float32
        Y positions of each fiber at each wavelength [nExp * nfiber, nWave]

    wave : ndarray of np.float32
        wavelength grid

    nWave : int, np.int32
        size of wavelength grid

    obsinfo : ndarray 
        observing information object

    seeing0 : ndarray of np.float32
        FWHM of seeing at guider wavelength (5400 Angstroms) [nExp]

    seeing : 2D ndarray of np.float32
        FWHM of seeing at each wavelength [nWave, nExp]

    bbwave : ndarray of np.float32
        [4] g, r, i, z band reference wavelengths

    bbindex : ndarray of np.float32
        [4] index of wave closest to g, r, i, z reference wavelengths

    pixelscale : np.float32
        pixel scale for output grid (arcsec)

    dkernel : np.float32
        pixel scale for kernel (arcsec)

    kernel : Kernel class object
        object to return radial kernel function

    conversion : np.float32
        factor to multiply per fiber units by to get per pixel units

    nside : np.int32
        number of pixels on a side for the image grid

    nimage : np.int32
        number of pixels total for the image grid

    length : np.float32
        length of image grid in arcsec (based on outer pixel edges)

    x2i : 2D ndarray of np.float32
        array of X positions on grid

    y2i : 2D ndarray of np.float32
        array of Y positions on grid

    xi : ndarray of np.float32
        1D array of X positions

    yi : ndarray of np.float32
        1D array of Y positions

    xmin : np.float32
        outer edge of lowest X pixel

    ymin : np.float32
        outer edge of lowest Y pixel

    xmax : np.float32
        outer edge of highest X pixel

    ymax : np.float32
        outer edge of highest Y pixel
"""
    def __init__(self, plate=None, ifu=None, release='DR15', waveindex=None,
                 pixelscale=0.75, dkernel=0.05, verbose=True):
        self.plate = plate
        self.ifu = ifu
        self.plateifu = "{plate}-{ifu}".format(plate=self.plate, ifu=self.ifu)
        self.release = release
        self.nfiber = int(self.ifu) // 100  # depends on MaNGA naming
        self.rss = None
        self.waveindex = waveindex
        self.pixelscale = pixelscale
        self.verbose = verbose
        self.dkernel = dkernel
        if self.waveindex is not None:
            self.waveindex = self._arrayify_int32(self.waveindex)
        self.fiberradius = 1.
        self.GPSF = None
        self.RPSF = None
        self.IPSF = None
        self.ZPSF = None
        self.GFWHM = None
        self.RFWHM = None
        self.IFWHM = None
        self.ZFWHM = None
        return

    def _arrayify_int32(self, quantity):
        """Cast quantity as ndarray of numpy.int32"""
        try:
            length = len(quantity)
        except TypeError:
            length = 1
        return np.zeros(length, dtype=np.int32) + quantity

    def run(self):
        """Perform all steps to create a cube"""
        if(self.verbose):
            print("Importing RSS fluxes.")
        self.set_rss()
        if(self.verbose):
            print("Creating output grid.")
        self.set_image_grid()
        if(self.verbose):
            print("Setting up kernel.")
        self.set_kernel()
        if(self.verbose):
            print("Constructing data flux and ivar arrays to use.")
        self.set_flux_rss()
        if(self.verbose):
            print("Constructing PSF model flux array.")
        self.set_flux_psf()
        if(self.verbose):
            print("Calculate cube.")
        self.calculate_cube()
        if (len(self.wave) == self.rss.data['FLUX'].data.shape[1]):
            self.set_band()
        return

    def set_rss(self):
        """Acquire the RSS data and set related attributes

        Notes:
        -----

        Uses Marvin tools to get RSS data.

        Sets attributes:

         .rss - Marvin RSS object
         .nExp - Number of exposures
         .xpos - X positions of each fiber [nExp * nfiber, nWave]
         .ypos - Y positions of each fiber [nExp * nfiber, nWave]
         .wave - Selected wavelength grid [nWave]
         .nWave - Size of wavelength grid
         .obsinfo - Observing information object
         .seeing0 - FWHM of seeing at guider wavelength (5400 Angstroms) [nExp]
         .seeing - FWHM at each wavelength [nWave, nExp]
         .bbwave - g, r, i, z band reference wavelengths
         .bbindex - index of wave closest to g, r, i, z reference wavelengths
"""
        self.rss = rss.RSS(plateifu=self.plateifu, release=self.release)
        self.nExp = self.rss.data[0].header['NEXP']
        self.xpos = self.rss.data['XPOS'].data
        self.ypos = self.rss.data['YPOS'].data

        self.wave = self.rss.data['WAVE'].data
        self.nWave = self.rss.data['FLUX'].shape[1]

        # Find wavelengths near griz
        gpivot = 4702.50
        rpivot = 6176.58
        ipivot = 7496.12
        zpivot = 8946.71
        self.bbwave = np.array([gpivot, rpivot, ipivot, zpivot],
                               dtype=np.float32)
        self.bbindex = np.zeros(len(self.bbwave), dtype=np.int32)
        for j, wave in enumerate(self.bbwave):
            self.bbindex[j] = min(range(len(self.wave)),
                                  key=lambda i: abs(self.wave[i] - wave))

        # Use waveindex if that was set
        if (self.waveindex is not None):
            self.nWave = len(self.waveindex)
            self.wave = self.wave[self.waveindex]
            self.xpos = self.xpos[:, self.waveindex]
            self.ypos = self.ypos[:, self.waveindex]

        # Set FWHM values as a function of wavelength
        self.obsinfo = self.rss.data['OBSINFO'].data
        self.seeing0 = (self.obsinfo.field('SEEING') *
                        self.obsinfo.field('PSFFAC'))

        lambda0 = 5400.
        self.seeing = [[self.seeing0[i] * math.pow(lambda0 / self.wave[j], 1. / 5.)
                        for i in range(self.seeing0.shape[0])]
                       for j in range(self.nWave)]
        self.seeing = np.array(self.seeing)
        return

    def _create_grid(self, d=None):
        """Create a grid (used for image and kernel)"""
        rough_length = gridsize[self.nfiber]
        nside = ((np.int32(np.ceil(rough_length / d)) // 2) * 2 + 1)
        length = np.float32(nside * d)
        xmin = -0.5 * length
        xmax = 0.5 * length
        ymin = -0.5 * length
        ymax = 0.5 * length
        xi = np.linspace(xmin + 0.5 * d, xmax - 0.5 * d, nside,
                         dtype=np.float32)
        yi = np.linspace(ymin + 0.5 * d, ymax - 0.5 * d, nside,
                         dtype=np.float32)
        x2i, y2i = np.meshgrid(xi, yi)
        return (nside, length, x2i, y2i, xi, yi)

    def set_image_grid(self):
        """Create the image output grid

        Notes:
        -----

        Sets attributes:

         .conversion - factor to multiply per fiber units by to get
                       per pixel units
         .nside - number of pixels on a side for the image grid
         .nimage - number of pixels total in image
         .length - length of image grid in arcsec (based on outer pixel edges)
         .x2i - 2-D array of X positions
         .y2i - 2-D array of Y positions
         .xi - 1-D array of X positions
         .yi - 1-D array of Y positions
         .xmin - outer edge of lowest X pixel
         .ymin - outer edge of lowest Y pixel
         .xmax - outer edge of highest X pixel
         .ymax - outer edge of highest Y pixel
"""
        self.conversion = self.pixelscale ** 2 / np.pi
        (self.nside, self.length, self.x2i, self.y2i, self.xi, self.yi) = self._create_grid(self.pixelscale)
        self.nimage = self.nside ** 2
        self.xmin = -0.5 * self.length
        self.xmax = 0.5 * self.length
        self.ymin = -0.5 * self.length
        self.ymax = 0.5 * self.length
        return

    def set_kernel(self):
        """Set the kernel for each wavelength and exposure

        Notes:
        -----

        Sets attributes:

         .kernel - Kernel class object
"""
        rough_length = gridsize[self.nfiber]
        self.kernel = mnsa.kernel.Kernel(rough_length=rough_length,
                                         dkernel=self.dkernel,
                                         nseeing=100)
        return

    def set_flux_psf(self, xcen=0., ycen=0., alpha=1, noise=None):
        """Set the fiber fluxes to a PSF corresponding to the kernel

        Parameters:
        ----------

        xcen : float, np.float32
            X center of PSF desired

        ycen : float, np.float32
            Y center of PSF desired

        alpha : float, np.float32
            scale to apply to PSF size (default 1)

        noise : float, np.float32
            noise scaling log factor

        Notes:
        -----

        Requires set_kernel() to have already been called to set the
        flux for each exposure and wavelength.

        Only uses wavelengths specified by the object's waveindex
        attribute or all the wavelengths if waveindex not given.

        It adds noise by multiplying the fluxes by 10^noise, sampling
        from a Poisson distribution using that as a mean, and then
        dividing by 10^noise.

        Sets attributes:

         .flux_psf - flux in each fiber for simulation [nExp * nfiber, nWave]
         .flux_psf_ivar - inverse variance of flux in each fiber for simulation
                      [nExp * nfiber, nWave]
         .flux_psf0 - noise free version of flux_psf [nExp * nFiber, nWave]
"""
        if (xcen is None):
            xcen = 0.
        if (ycen is None):
            ycen = 0.

        self.flux_psf = np.zeros([self.xpos.shape[0], self.nWave])
        self.flux_psf_ivar = np.ones([self.xpos.shape[0], self.nWave],
                                     dtype=np.float32)
        self.flux_psf_mask = np.zeros([self.xpos.shape[0], self.nWave])
        for iWave in np.arange(self.nWave):
            for iExp in np.arange(self.nExp):
                xsample = self.xpos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                ysample = self.ypos[iExp * self.nfiber:(iExp + 1) * self.nfiber, iWave]
                dx = xsample - xcen
                dy = ysample - ycen
                rsample = np.sqrt(dx**2 + dy**2)
                self.flux_psf[iExp * self.nfiber:(iExp + 1) * self.nfiber,
                              iWave] = self.kernel.radial(seeing=self.seeing[iWave, iExp] * alpha, radii=rsample) * np.pi * self.fiberradius**2

        self.flux_psf0 = self.flux_psf.copy()

        # Add noise if desired
        if(noise is not None):
            for i in range(len(self.flux_psf)):
                self.flux_psf[i] = np.random.poisson(self.flux_psf[i] * 10 ** noise) / 10. ** noise

        return

    def set_flux_rss(self):
        """Set the flux to the RSS input values

        Notes:
        -----

        Only uses wavelengths specified by the object's waveindex
        attribute or all the wavelengths if waveindex not given.

        Sets attributes:

         .flux - flux in each fiber [nExp * nfiber, nWave]
         .flux_ivar - inverse variance of flux in each fiber
                      [nExp * nfiber, nWave]
         .flux_mask - mask of each fiber [nExp * nfiber, nWave]

         .flux_disp - LSF dispersion [nExp * nfiber, nWave]
         .flux_predisp - LSF pre-dispersion [nExp * nfiber, nWave]
         .lsf_exist - True if flux_disp, flux_predisp available, False if not
"""
        self.flux = self.rss.data['FLUX'].data
        self.flux_ivar = self.rss.data['IVAR'].data
        self.flux_mask = self.rss.data['MASK'].data

        if (self.waveindex is not None):
            self.flux = self.flux[:, self.waveindex]
            self.flux_ivar = self.flux_ivar[:, self.waveindex]
            self.flux_mask = self.flux_mask[:, self.waveindex]

        try:
            self.flux_disp = self.rss.data['DISP'].data
            self.flux_predisp = self.rss.data['PREDISP'].data
            if (self.waveindex is not None):
                self.flux_disp = self.flux_disp[:, self.waveindex]
                self.flux_predisp = self.flux_predisp[:, self.waveindex]
            self.lsf_exist = True
        except:
            self.lsf_exist = False

        return

    def create_weights(self, xsample=None, ysample=None, ivar=None,
                       waveindex=None):
        """Calculate weights based on nearest fiber

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        ivar : ndarray of np.float32
            inverse variance of samples

        Returns:
        -------

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        This version just sets the weight to unity for the nearest fiber.
"""
        iok = np.where(ivar > 0.)
        w = np.zeros([len(xsample), self.nside * self.nside], dtype=np.float32)
        for i in np.arange(self.nside):
            for j in np.arange(self.nside):
                dx = xsample - self.x2i[i, j]
                dy = ysample - self.y2i[i, j]
                r = np.sqrt(dx ** 2 + dy ** 2)
                iclosest = r[iok].argmin()
                w[iclosest, i * self.nside + j] = 1.
        wwT = self.normalize_weights(w)
        return (wwT)

    def normalize_weights(self, w):
        """Normalize weights

        Parameters:
        ----------

        w : ndarray of np.float32
            weights [nExp * nfiber, nside * nside]

        Returns:
        -------

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        Normalizes the contributions of each pixel over all fiber-exposures.
        If the pixel has no contributions, sets weight to zero.
"""
        wsum = w.sum(axis=0)

        ww = np.zeros(w.shape, dtype=np.float32)
        for i in np.arange(self.nimage):
            if wsum[i] == 0:
                ww[:, i] = 0
            else:
                ww[:, i] = w[:, i] / wsum[i]
        wwT = ww.T
        return (wwT)

    def calculate_cube(self):
        """Calculate cube and cube inverse variance

        Notes:
        ------

        Sets attributes:

        .cube : ndarray of np.float32
            [nWave, nside, nside] reconstruction result
        .cube_psf : ndarray of np.float32
            [nWave, nside, nside] reconstruction result for PSF
        .cube_ivar : ndarray of np.float32
            [nWave, nside, nside] inverse variance of reconstruction result
        .cube_corr : list of correlation matrix in sparse array format.
            If waveindex is None, return the correlation matrix at SDSS 
            g,r,i,z broadband effective wavelengths; else, return the
            correlation matrix for first four wavelength indexes
        .cube_mask : ndarray of np.int32
            [nWave, nside, nside] mask of reconstruction pixels
"""
        if self.waveindex is None:
            waveindex = np.arange(self.nWave)
            nWave = len(waveindex)
        else:
            nWave = self.nWave
            waveindex = self.waveindex

        self.cube = np.zeros([nWave, self.nside, self.nside],
                             dtype=np.float32)
        self.cube_psf = np.zeros([nWave, self.nside, self.nside],
                                 dtype=np.float32)
        self.cube_ivar = np.zeros([nWave, self.nside, self.nside],
                                  dtype=np.float32)

        i = 0
        self.cube_corr = []
        self.slice_fail = []
        self.cube_mask = np.zeros([nWave, self.nside, self.nside],
                                  dtype=np.int32)
        if(self.lsf_exist):
            self.disp = np.zeros([nWave, self.nside, self.nside],
                                 dtype=np.float32)
            self.predisp = np.zeros([nWave, self.nside, self.nside],
                                    dtype=np.float32)

        print(nWave)
        for iWave in np.arange(nWave):
            print(iWave, flush=True)
            try:
                w0, weights = self.create_weights(xsample=self.xpos[0:self.nExp * self.nfiber, iWave],
                                                  ysample=self.ypos[0:self.nExp * self.nfiber, iWave],
                                                  ivar=self.flux_ivar[:, iWave],
                                                  waveindex=iWave)
            except np.linalg.LinAlgError:
                print('failing to converge', iWave)
                self.slice_fail.append(iWave)
            self.w0 = w0
            fcube = ((weights.dot(self.flux[:, iWave])).reshape(self.nside,
                                                                self.nside) *
                     self.conversion)
            self.cube[i, :, :] = fcube
            fcube = ((weights.dot(self.flux_psf[:, iWave])).reshape(self.nside,
                                                                    self.nside) *
                     self.conversion)
            self.cube_psf[i, :, :] = fcube

            # covariance
            covar = (self.covar(iWave, self.flux_ivar, weights) *
                     self.conversion**2)
            var = np.diagonal(covar)
            igt0 = np.where(var > 0)[0]
            ivar = np.zeros(self.nside * self.nside, dtype=np.float32)
            ivar[igt0] = 1. / var[igt0]
            self.cube_ivar[i, :, :] = ivar.reshape([self.nside,
                                                    self.nside])

            # correlation matrix only available for up to four wavelength slices
            if self.waveindex is None:
                if iWave in self.bbindex:
                    corr = covar / np.outer(np.sqrt(var), np.sqrt(var))
                    corr[np.where(covar == 0)] = 0
                    self.cube_corr.append(sparse.csr_matrix(corr))
            elif i < 4:
                corr = covar / np.outer(np.sqrt(var), np.sqrt(var))
                corr[np.where(covar == 0)] = 0
                self.cube_corr.append(sparse.csr_matrix(corr))

            # mask
            self.cube_mask[i, :, :] = self.mask(iWave, self.flux_ivar,
                                                self.flux_mask, w0,
                                                weights).reshape([self.nside,
                                                                  self.nside])

            i = i + 1

            if(self.lsf_exist):
                self.disp[iWave] = (
                (np.abs(weights) / (matlib.repmat(np.abs(weights).sum(axis=1), weights.shape[-1], 1).T)).dot(
                    self.flux_disp[:, iWave])).reshape(self.nside, self.nside)
                self.predisp[iWave] = (
                (np.abs(weights) / (matlib.repmat(np.abs(weights).sum(axis=1), weights.shape[-1], 1).T)).dot(
                    self.flux_predisp[:, iWave])).reshape(self.nside, self.nside)

        if(self.lsf_exist):
            indices = np.isnan(self.disp)
            self.disp[indices] = 0
            indices = np.isnan(self.predisp)
            self.predisp[indices] = 0
        return

    def covar(self, iWave=None, flux_ivar=None, weights=None):
        """Return cube covariance matrix for a wavelength

        Parameters:
        ----------

        iWave : int, np.int32
            index of wavelength to calculate for

        flux_ivar: ndarray of np.float32
            flux inverse variance of each fiber [nExp * nfiber]

        weights: ndarray of np.float32
            [nside * nside, nExp * nfiber]
            weights matrix between pixels and fibers

        Returns:
        -------

        covar : ndarray of np.float32
            [nside * nside, nside * nside] covariance matrix
"""
        iok = np.where(flux_ivar[:, iWave] > 0)[0]
        wwT = (weights[:, :])[:, iok]
        covar = wwT.dot(np.diag(1 / (flux_ivar[iok, iWave]))).dot(wwT.T)
        return (covar)

    def mask(self, iWave=None, flux_ivar=None, flux_mask=None, w0=None,
             weights=None):
        """Return mask matrix for a typical wavelength given the weights matrix

        Parameters:
        ----------

        iWave : int, np.int32
            index of wavelength to calculate for

        flux_ivar: ndarray of np.float32
            flux inverse variance of each fiber [nExp * nfiber]

        flux_mask : ndarray of np.int32
            mask of each fiber [nExp * nfiber]

        w0: ndarray of np.float32 
            [nside * nside, nExp * nfiber]
            unnormalized weights without bad fibers

        weights: ndarray of np.float32
            [nside * nside, nExp * nfiber]
            weights matrix between pixels and fibers

        Returns:
        -------

        maskimg : ndarray of np.int32
            [nside * nside] bitmask of pixels

        Notes:
        -----

        Uses MANGA_DRP3PIXMASK bit values:
           DEADFIBER - set if a dead fiber would have contributed to pixel
           LOWCOV - set if variance would be higher than twice median
                    were all fiber-exposures equally noisy
           NOCOV - set if no coverage by any fiber
           DONOTUSE - set if LOWCOV and/or NOCOV set
"""
        flagdeadfiber = sdss_flagval('MANGA_DRP3PIXMASK', 'DEADFIBER')
        flaglocov = sdss_flagval('MANGA_DRP3PIXMASK', 'LOWCOV')
        flagnocov = sdss_flagval('MANGA_DRP3PIXMASK', 'NOCOV')
        flagnouse = sdss_flagval('MANGA_DRP3PIXMASK', 'DONOTUSE')
        (mask_dead, mask_lowcov, mask_nocov,
         mask_dnu) = [np.zeros(self.nside**2) for i in range(4)]

        index_nocov = np.where(w0.sum(axis=0) == 0)[0]
        mask_nocov[index_nocov] = flagnocov
        mask_dnu[index_nocov] = flagnouse

        cov = np.diag(weights.dot(weights.T))
        cov = cov / np.median(cov[np.where(cov)])
        indices = np.logical_or(cov == 0, cov > 2)
        mask_lowcov[indices] = flaglocov
        mask_dnu[indices] = flagnouse

        index_deadfibers = np.where(np.bitwise_and(np.uint32(flagdeadfiber),
                                                   np.uint32(flux_mask[:,
                                                                       iWave])))
        mask_dead = (((w0[index_deadfibers] != 0).sum(axis=0) != 0) *
                     flagdeadfiber)

        maskimg = np.uint32(mask_nocov)
        maskimg = np.bitwise_or(maskimg, np.uint32(mask_lowcov))
        maskimg = np.bitwise_or(maskimg, np.uint32(mask_dead))
        maskimg = np.bitwise_or(maskimg, np.uint32(mask_dnu))

        return maskimg

    def plot_slice(self, iWave=0, keyword=None, vmax=None, vmin=0.):
        """Plot a slice of the cube

        Parameters:
        ----------

        iWave : default=0,int, np.int32
            index of wavelength to plot, order in waveindex, not the value.

        keyword : 'simulation' or else
            if keyword == 'simulation': plot reconstruction from simulated flux
            else plot reconstruction from real flux

        vmax, vmin: float
            maximum and minimum of plot
            default vmax is set to maximum of flux
            default vmin is set to 0.
"""
        if keyword == 'simulation':
            target = self.cube_psf[iWave, :, :]
        else:
            target = self.cube[iWave, :, :]
            keyword = ''
        if (vmax is None):
            vmax = target.max() * 1.02
        extent = (self.xmin, self.xmax, self.ymin, self.ymax)
        plt.figure(figsize=(6.5, 5.))
        font = {'family': 'sans-serif',
                'size': 15}
        plt.rc('font', **font)
        plt.imshow(target, extent=extent, vmin=vmin,
                   vmax=vmax, cmap=cm.gray_r, origin='lower')
        plt.xlabel('X (arcsec)')
        plt.ylabel('Y (arcsec)')
        plt.title('reconstruction of ' + keyword + ' slice')
        plt.colorbar(label='flux')

    def set_band(self):
        """Set average results for each band include FWHM estimate

        Notes:
        -----

        Only uses the full range wavelengths will give the band average

        Sets attributes:

         .GPSF/RPSF/IPSF/ZPSF - ndarray of np.float32
            [nside, nside] simulation image for each broadband
         .GIMG/RIMG/IIMG/ZIMG - ndarray of np.float32
             [nside, nside] real image for each broadband
         .GFWHM/RFWHM/IFWHM/ZFWHM - float, np.float32
             FWHM for each simulation image
"""
        self.GPSF = self.PSFaverage('g', self.wave, self.cube_psf)
        self.RPSF = self.PSFaverage('r', self.wave, self.cube_psf)
        self.IPSF = self.PSFaverage('i', self.wave, self.cube_psf)
        self.ZPSF = self.PSFaverage('z', self.wave, self.cube_psf)
        self.GIMG = self.PSFaverage('g', self.wave, self.cube)
        self.RIMG = self.PSFaverage('r', self.wave, self.cube)
        self.IIMG = self.PSFaverage('i', self.wave, self.cube)
        self.ZIMG = self.PSFaverage('z', self.wave, self.cube)
        self.GFWHM = self.fit_fwhm(self.GPSF)
        self.RFWHM = self.fit_fwhm(self.RPSF)
        self.IFWHM = self.fit_fwhm(self.IPSF)
        self.ZFWHM = self.fit_fwhm(self.ZPSF)
        return

    def fit_fwhm(self, image, xcen=None, ycen=None):
        """Fit FWHM to an image using radial kernel as a model
"""
        seeings = self.kernel.seeing

        nx = image.shape[0]
        ny = image.shape[1]

        if(xcen is None):
            xcen = (np.float32(nx) * 0.5) - 0.5
        if(ycen is None):
            ycen = (np.float32(ny) * 0.5) - 0.5

        xx = np.outer(np.arange(nx, dtype=np.float32) - xcen,
                      np.ones(ny, dtype=np.float32))
        yy = np.outer(np.ones(nx, dtype=np.float32),
                      np.arange(ny, dtype=np.float32) - ycen)
        rr = np.sqrt(xx**2 + yy**2) * self.pixelscale

        chi2 = np.zeros(len(seeings), dtype=np.float32)
        for i, seeing in enumerate(seeings):
            model = self.kernel.radial(seeing=seeing, radii=rr.flatten())
            model = model * self.pixelscale**2

            A = ((model * image.flatten()).sum() /
                 (model ** 2).sum())
            chi2[i] = ((A * model - image.flatten()) ** 2).sum()

        ind = np.argmin(chi2)
        seeing = seeings[ind]

        fwhm = self.kernel.fwhm(seeing=seeing)

        return(fwhm)

    def PSFaverage(self, color=None, wave=None, PSF=None):
        """ calculate FWHM for given image

        Parameters:
        ----------

        color : str, the color of band, can choose 'g'/'r'/'i'/'z'

        wave : ndarray of np.float32
            the wavelengths for the cube

        PSF : ndarray of np.float32 [nside,nside,nWave]
            the spectrum of cube

        Returns:
        -------

        PSF_ave : ndarray of np.float32 [nside,nside]
            the average of cube for given band
"""
        filterfile = os.path.join(os.getenv('MNSA_DIR'),
                                  'python', 'data', color + '_filter.dat')
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

        n = PSF.shape[1]
        nWave = len(wave)
        PSF_ave = (matlib.repmat(band_value, n**2, 1).T *
                   (PSF.reshape(nWave, n**2))).reshape(nWave, n, n).sum(axis=0) / band_value.sum()
        return PSF_ave

    def write(self, filename=None):
        """Write to a FITS file as MaNGA cube form

        Parameters:
        ----------
        filename: str
            the name of fits file
"""

        # Headers
        card_GFWHM = fits.Card('GFWHM', self.GFWHM,
                               'Reconstructed FWHM in g-band (arcsec)')
        card_RFWHM = fits.Card('RFWHM', self.RFWHM,
                               'Reconstructed FWHM in r-band (arcsec)')
        card_IFWHM = fits.Card('IFWHM', self.IFWHM,
                               'Reconstructed FWHM in i-band (arcsec)')
        card_ZFWHM = fits.Card('ZFWHM', self.ZFWHM,
                               'Reconstructed FWHM in z-band (arcsec)')
        card_FWHM_list = [card_GFWHM, card_RFWHM, card_IFWHM, card_ZFWHM]

        card_BSCALE = fits.Card('BSCALE', 1.00000, 'Intensity unit scaling')
        card_BZERO = fits.Card('BZERO', 0.00000, 'Intensity zeropoint')
        card_BSCALE_2 = fits.Card('BSCALE', 1.00000, 'Flux unit scaling')
        card_BZERO_2 = fits.Card('BZERO', 0.00000, 'Flux zeropoint')

        card_WCS_1 = fits.Card('CRPIX1', (self.nside + 1) / 2,
                               'Reference pixel (1-indexed)')
        card_WCS_2 = fits.Card('CRPIX2', (self.nside + 1) / 2,
                               'Reference pixel (1-indexed)')
        card_WCS_3 = fits.Card('CRVAL1',
                               self.rss.data['FLUX'].header['IFURA'])
        card_WCS_4 = fits.Card('CRVAL2',
                               self.rss.data['FLUX'].header['IFUDEC'])
        card_WCS_5 = fits.Card('CD1_1',
                               round(- self.length / self.nside / 3600, 9))
        card_WCS_6 = fits.Card('CD2_2',
                               round(self.length / self.nside / 3600, 9))
        card_WCS_7 = fits.Card('CTYPE1', 'RA---TAN')
        card_WCS_8 = fits.Card('CTYPE2', 'DEC--TAN')
        card_WCS_9 = fits.Card('CUNIT1', 'deg')
        card_WCS_10 = fits.Card('CUNIT2', 'deg')
        card_WCS_list = [card_WCS_1, card_WCS_2, card_WCS_3, card_WCS_4,
                         card_WCS_5, card_WCS_6, card_WCS_7, card_WCS_8,
                         card_WCS_9, card_WCS_10]

        # Primary
        hp = fits.PrimaryHDU(header=self.rss.data[0].header)
        hp.header['BUNIT'] = ('1E-17 erg/s/cm^2/Ang/spaxel',
                              'Specific intensity (per spaxel)')
        hp.header['MASKNAME'] = ('MANGA_DRP3PIXMASK',
                                 'Bits in sdssMaskbits.par used by mask extension')
        hp = self._insert_cardlist(hdu=hp, insertpoint='EBVGAL',
                                   cardlist=card_FWHM_list, after=True)
        if 'BSCALE' not in list(hp.header.keys()):
            hp.header.insert('BUNIT', card_BSCALE, after=False)
        if 'BZERO' not in list(hp.header.keys()):
            hp.header.insert('BUNIT', card_BZERO, after=False)

        # Flux
        cubehdr = fits.ImageHDU(name='FLUX', data=self.cube,
                                header=self.rss.data['FLUX'].header)
        cubehdr.header['BUNIT'] = ('1E-17 erg/s/cm^2/Ang/spaxel',
                                   'Specific intensity (per spaxel)')
        cubehdr.header['MASKNAME'] = ('MANGA_DRP3PIXMASK',
                                      'Bits in sdssMaskbits.par used by mask extension')
        cubehdr.header['HDUCLAS1'] = 'CUBE'
        cubehdr.header.rename_keyword('CTYPE1', 'CTYPE3')
        cubehdr.header.rename_keyword('CRPIX1', 'CRPIX3')
        cubehdr.header.rename_keyword('CRVAL1', 'CRVAL3')
        cubehdr.header.rename_keyword('CD1_1', 'CD3_3')
        cubehdr.header.rename_keyword('CUNIT1', 'CUNIT3')
        cubehdr = self._insert_cardlist(hdu=cubehdr, insertpoint='EBVGAL',
                                        cardlist=card_FWHM_list, after=True)
        cubehdr = self._insert_cardlist(hdu=cubehdr, insertpoint='CUNIT3',
                                        cardlist=card_WCS_list, after=True)
        if 'BSCALE' not in list(cubehdr.header.keys()):
            cubehdr.header.insert('BUNIT', card_BSCALE, after=False)
        if 'BZERO' not in list(cubehdr.header.keys()):
            cubehdr.header.insert('BUNIT', card_BZERO, after=False)

        try:
            card_flux_fail = fits.Card('FAILSLIC', str(self.slice_fail),
                                       'slices failed to converge')
            cubehdr.header.insert('ZFWHM', card_flux_fail, after=True)
        except:
            pass

        # IVAR
        ivar_hdr = fits.ImageHDU(name='IVAR', data=self.cube_ivar,
                                 header=self.rss.data['IVAR'].header)
        ivar_hdr.header['HDUCLAS1'] = 'CUBE'

        # MASK
        mask_hdr = fits.ImageHDU(name='MASK', data=self.cube_mask,
                                 header=self.rss.data['MASK'].header)
        mask_hdr.header['HDUCLAS1'] = 'CUBE'

        # DISP
        disp_hdr = fits.ImageHDU(name='DISP', data=self.disp,
                                 header=self.rss.data['DISP'].header)

        # PREDISP
        predisp_hdr = fits.ImageHDU(name='PREDISP', data=self.predisp,
                                    header=self.rss.data['PREDISP'].header)

        # IMG & PSF for each band
        card_BUNIT = fits.Card('BUNIT', 'nanomaggies/pixel')
        loc = ['IFURA', 'IFUDEC', 'OBJRA', 'OBJDEC']
        card_loc_list = self._set_cardlist(cubehdr, loc) + [card_BSCALE_2,
                                                            card_BZERO_2,
                                                            card_BUNIT]

        GIMG_hdr = fits.ImageHDU(name='GIMG', data=self.GIMG)
        GIMG_hdr = self._insert_cardlist(hdu=GIMG_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_GFWHM],
                                         after=False)

        RIMG_hdr = fits.ImageHDU(name='RIMG', data=self.RIMG)
        RIMG_hdr = self._insert_cardlist(hdu=RIMG_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_RFWHM],
                                         after=False)

        IIMG_hdr = fits.ImageHDU(name='IIMG', data=self.IIMG)
        IIMG_hdr = self._insert_cardlist(hdu=IIMG_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_IFWHM],
                                         after=False)

        ZIMG_hdr = fits.ImageHDU(name='ZIMG', data=self.GIMG)
        ZIMG_hdr = self._insert_cardlist(hdu=ZIMG_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_ZFWHM],
                                         after=False)

        GPSF_hdr = fits.ImageHDU(name='GPSF', data=self.GPSF)
        GPSF_hdr = self._insert_cardlist(hdu=GPSF_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_GFWHM],
                                         after=False)

        RPSF_hdr = fits.ImageHDU(name='RPSF', data=self.RPSF)
        RPSF_hdr = self._insert_cardlist(hdu=RPSF_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_RFWHM],
                                         after=False)

        IPSF_hdr = fits.ImageHDU(name='IPSF', data=self.IPSF)
        IPSF_hdr = self._insert_cardlist(hdu=IPSF_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_IFWHM],
                                         after=False)

        ZPSF_hdr = fits.ImageHDU(name='ZPSF', data=self.GPSF)
        ZPSF_hdr = self._insert_cardlist(hdu=ZPSF_hdr, insertpoint='EXTNAME',
                                         cardlist=card_WCS_list +
                                         card_loc_list + [card_ZFWHM],
                                         after=False)

        # CORR
        CORR_hdr = []
        for i in range(4):
            corr = self._table_correlation(correlation=self.cube_corr[i].toarray(), thresh=1E-14)
            corr.header.append(fits.Card('BBWAVE', self.bbwave[i],
                                         'Wavelength (Angstroms)'))
            corr.header.append(fits.Card('BBINDEX', self.bbindex[i],
                                         'Slice number'))
            corr.header.append(fits.Card('COVTYPE', 'Correlation'))
            corr.header.append(fits.Card('COVSHAPE', '(%s,%s)' %
                                         (self.nimage, self.nimage)))
            CORR_hdr.append(corr)
        CORR_hdr[0].header.append(fits.Card('EXTNAME', 'GCORREL'))
        CORR_hdr[1].header.append(fits.Card('EXTNAME', 'RCORREL'))
        CORR_hdr[2].header.append(fits.Card('EXTNAME', 'ICORREL'))
        CORR_hdr[3].header.append(fits.Card('EXTNAME', 'ZCORREL'))

        hduRSS = []
        for i in range(1, len(self.rss.data)):
            if ((self.rss.data[i].header['XTENSION'] == 'IMAGE') and
                (len(self.rss.data[i].data) == len(self.rss.data['WAVE'].data))):
                hduRSS.append(self.rss.data[i])

        try:
            # PSF
            PSF_hdr = fits.ImageHDU(name='PSF', data=self.cube_psf,
                                    header=cubehdr.header)
            hdu = fits.HDUList([hp, cubehdr, PSF_hdr, ivar_hdr, mask_hdr,
                                disp_hdr, predisp_hdr] + hduRSS +
                               [self.rss.data['OBSINFO'],
                                GIMG_hdr, RIMG_hdr, IIMG_hdr, ZIMG_hdr,
                                GPSF_hdr, RPSF_hdr, IPSF_hdr, ZPSF_hdr] +
                               CORR_hdr)
        except:
            hdu = fits.HDUList([hp, cubehdr, ivar_hdr, mask_hdr,
                                self.rss.data['WAVE'],
                                self.rss.data['SPECRES'],
                                self.rss.data['SPECRESD'],
                                self.rss.data['OBSINFO'],
                                GIMG_hdr, RIMG_hdr, IIMG_hdr, ZIMG_hdr,
                                GPSF_hdr, RPSF_hdr, IPSF_hdr, ZPSF_hdr] +
                               CORR_hdr)

        # Doesn't actually gzip??
        if filename and len(filename) > 8 and filename[-8:] == 'fits.gz':
            data = filename
        else:
            data = filename + ".fits.gz".format(filename=filename)
        hdu.writeto(data, overwrite=True, checksum=True)

        hdu.close()
        return

    def _insert_cardlist(self, hdu=None, insertpoint=None, cardlist=None,
                         after=False):
        """insert a cardlist into the header of a FITS hdu

        Parameters:
        ----------
        hdu: FITS hdu

        insertpoint: int
            The index into the list of header keywords before which
            the new keyword should be inserted, or the name of a
            keyword before which the new keyword should be inserted.
            Can also accept a (keyword, index) tuple for inserting
            around duplicate keywords.

        cardlist: list
            list of header cards to be inserted. Header
            cards will be inserted as the order of the list

        after: bool
            If set to True, insert after the specified index or
            keyword, rather than before it. Defaults to False.

        Return:
        --------

        hdu: fits file that have the cards inserted

        """
        for i in range(len(cardlist)):
            if after:
                hdu.header.insert(insertpoint, cardlist[i], after=after)
                insertpoint = cardlist[i].keyword
            else:
                hdu.header.insert(insertpoint, cardlist[i], after=after)
        return hdu

    def _set_cardlist(self, hdu=None, keyword_list=None):
        """ Extract header card list from a FITS hdu

        Parameters:
        ----------
        hdu: FITS hdu

        keyword_list: list
            keywords to be extracted, including value and comments

        Return:
        --------
        cardlist: list
            cards of FITS headers
"""
        cardlist = []
        for index, keyword in enumerate(keyword_list):
            cardlist.append(fits.Card(keyword, hdu.header[keyword],
                                      hdu.header.comments[keyword]))
        return cardlist

    def _table_correlation(self, correlation=None, thresh=1E-12):
        """create a BinTableHDU for the correlation in sparse matrix form

        Parameters:
        ----------
        correlation: ndarray of float32
           [nside*nside,nside*nside] correlation matrix 

        thresh: float32
            threshold for the correlation entries to be stored.

        Return:
        --------
        hdu: BinTableHDU that includes the information of the correlation matrix

        Note:
        --------

        Five columns of the table are value, the location (C1,c2) of
        the first point in the grid, the location (C1,c2) of the
        second point in the grid.

"""
        nside = int(np.sqrt(correlation.shape[0]))
        index_G = np.where(np.abs(correlation) > thresh)
        corr_G = correlation[index_G]
        triangle = np.where(index_G[1] >= index_G[0])[0]
        index_G = np.array([index_G[0][triangle], index_G[1][triangle]])
        corr_G = corr_G[triangle]
        i_c1, i_c2, j_c1, j_c2 = [[] for i in range(4)]
        for i in range(len(corr_G)):
            i_c2.append(index_G[0, i] // nside)
            i_c1.append(index_G[0, i] % nside)
            j_c2.append(index_G[1, i] // nside)
            j_c1.append(index_G[1, i] % nside)
        i1 = fits.Column(name='INDXI_C1', array=np.array(i_c1), format='J')
        i2 = fits.Column(name='INDXI_C2', array=np.array(i_c2), format='J')
        j1 = fits.Column(name='INDXJ_C1', array=np.array(j_c1), format='J')
        j2 = fits.Column(name='INDXJ_C2', array=np.array(j_c2), format='J')
        value = fits.Column(name='RHOIJ', array=np.array(corr_G), format='D')
        hdu = fits.BinTableHDU.from_columns([i1, i2, j1, j2, value])
        return hdu


class ReconstructShepard(Reconstruct):
    """Reconstruction of cubes from Shepards method

    Attributes:
    ----------

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    nfiber : int
        number of fibers

    release : str
        data release (default 'DR15')

    rss : RSS object
        Marvin RSS output

    waveindex : int, np.int32
        indices of wavelengths to reconstruct (default None)

    Notes:
    ------

    Unless waveindex is set, uses all wavelengths.
"""
    def create_weights(self, xsample=None, ysample=None,
                       ivar=None, waveindex=None, shepard_sigma=0.7):
        """Calculate weights for Shepards method

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        ivar : ndarray of np.float32
            inverse variance of samples

        shepard_sigma : float, np.float32
            sigma for Gaussian in kernel, in arcsec (default 0.7)

        Returns:
        -------

        w0 : ndarray of np.float32
            unnormalized weights without bad fibers, [nExp * nfiber,nside * nside]

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

        This version uses Shepards method.
"""
        nsample = len(xsample)
        dx = (np.outer(xsample, np.ones(self.nimage, dtype=np.float32)) -
              np.outer(np.ones(nsample, dtype=np.float32), self.x2i.flatten()))
        dy = (np.outer(ysample, np.ones(self.nimage)) -
              np.outer(np.ones(nsample, dtype=np.float32), self.y2i.flatten()))
        dr = np.sqrt(dx ** 2 + dy ** 2)

        w0 = np.exp(- 0.5 * dr ** 2 / shepard_sigma ** 2)
        ifit = np.where(dr > 1.6)
        w0[ifit] = 0
        w = np.transpose(matlib.repmat(ivar != 0, self.nside ** 2, 1)) * w0

        wwT = self.normalize_weights(w)
        return (w0, wwT)


class ReconstructCRR(Reconstruct):
    """Reconstruction of cubes from linear least square method

    Attributes:
    ----------

    plate : int, np.int32
        plate number

    ifu : int, np.int32
        IFU number

    nfiber : int
        number of fibers

    release : str
        data release (default 'DR15')

    rss : RSS object
        Marvin RSS output

    waveindex : int, np.int32
        indices of wavelengths to reconstruct (default None)

    Notes:
    ------

    Additional attributes are set by the methods, as documented.

    Unless waveindex is set, uses all wavelengths.
"""
    def __init__(self, plate=None, ifu=None, release='DR15',
                 waveindex=None, pixelscale=0.75, dkernel=0.05,
                 verbose=True, lam=1.e-4):
        super().__init__(plate=plate, ifu=ifu, release=release,
                         waveindex=waveindex, pixelscale=pixelscale,
                         dkernel=dkernel, verbose=verbose)
        self.lam = lam
        return

    def set_Amatrix(self, xsample=None, ysample=None, ivar=None,
                    waveindex=None):
        """Calculate kernel matrix for linear least square method

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        Returns:
        -------

        ifit: indices of pixels selected

        A : ndarray of np.float32
            kernel matrix [nExp * nfiber, nfit]

        Notes:
        -----

        indices will be used to recover A matrix back to regular grid
"""
        # Find pixels where there is at least one fiber within 1.6 arcsec.
        nsample = len(xsample)
        dx = np.outer(xsample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.x2i.flatten())
        dy = np.outer(ysample, np.ones(self.nimage)) - np.outer(np.ones(nsample), self.y2i.flatten())
        dr = np.sqrt(dx ** 2 + dy ** 2)
        ifit = np.where(dr.min(axis=0) <= 1.6)[0]
        nfit = len(ifit)
        dr = dr[:, ifit]
        dr = dr.reshape((self.nExp, self.nfiber, nfit))

        # Create A matrix, zeroing out any very low kernel values
        A = np.zeros((self.nExp, self.nfiber * nfit), dtype=np.float64)
        for iExp in np.arange(self.nExp, dtype=np.int32):
            radii = dr[iExp, :, :].flatten()
            kv = self.kernel.radial(seeing=self.seeing[waveindex, iExp],
                                    radii=radii) * self.pixelscale**2
            indices = np.where(kv > 1.e-4)[0]
            A[iExp, indices] = kv[indices]

        # Now shape it right
        A = A.reshape(self.nExp * self.nfiber, nfit)
        return (ifit, A)

    def create_weights(self, xsample=None, ysample=None, ivar=None,
                       waveindex=None):
        """Calculate weights for linear least square method

        Parameters:
        ----------

        xsample : ndarray of np.float32
            X position of samples

        ysample : ndarray of np.float32
            Y position of samples

        ivar : ndarray of np.float32
            inverse variance of samples

        kernel : float, np.float32
            kernel at each and exposure [nExp, nkernel, nkernel]

        Returns:
        -------
        w0 : ndarray of np.float32
            unnormalized weights without bad fibers, [nExp * nfiber,nside * nside]

        wwT : ndarray of np.float32
            normalized weights [nside * nside, nExp * nfiber]

        Notes:
        -----

"""
        self.ifit, A = self.set_Amatrix(xsample, ysample, ivar, waveindex)
        ivar = (ivar != 0)
        [U, D, VT] = np.linalg.svd(np.dot(np.diag(np.sqrt(ivar)), A),
                                   full_matrices=False)
        Dinv = 1 / D

        for i in range(len(D)):
            if D[i] < 1E-6:
                Dinv[i] = 0
        filt = 1 / (1 + self.lam**2 * Dinv**2)

        A_plus = np.dot(np.dot(VT.T, np.dot(np.diag(filt), np.diag(Dinv))),
                        U.T)

        Q = (np.dot(np.dot(VT.transpose(), np.dot(np.diag(1 / filt),
                                                  np.diag(D))), VT))
        sl = Q.sum(axis=1)
        Rl = (Q.T / sl.T).T
        where_are_NaNs = np.isnan(Rl)
        Rl[where_are_NaNs] = 0

        T = np.dot(np.dot(Rl, A_plus), np.diag(np.sqrt(ivar)))
        self.A_plus = self.set_reshape(np.dot(A_plus, np.diag(np.sqrt(ivar))))
        wwT = self.set_reshape(T)
        return (self.set_reshape(A.T).T, wwT)

    def set_reshape(self, inp):
        """ reshape the size of weights from selected pixels to a regular grid

        Parameters:
        ----------
        inp : ndarray of np.float32
            input array, [nfit, size]

        Return:
        --------
        output : ndarray of np.float32
            output array, [nside * nside, size]

"""
        output = np.zeros([self.nside ** 2, inp.shape[1]])
        output[self.ifit] = inp
        return output
