import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import astropy.wcs as wcs
import astropy.visualization as viz
import fitsio
import mnsa.utils.configuration as configuration


bandset_settings = dict()
bandset_settings['dlis'] = {'bands':[{'name':'z', 'scale':1.0},
                                     {'name':'r', 'scale':1.3},
                                     {'name':'g', 'scale':2.2}],
                            'stretch':5.,
                            'Q':1.5,
                            'minimum':- 0.1}
bandset_settings['wise'] = {'bands':[{'name':'W3', 'scale':0.005},
                                     {'name':'W2', 'scale':0.10},
                                     {'name':'W1', 'scale':0.10}],
                            'stretch':2.,
                            'Q':3.5,
                            'minimum':- 0.1}

bandset_settings['galex'] = {'bands':[{'name':'g', 'scale':1.0},
                                      {'name':'NUV', 'scale':40.5},
                                      {'name':'FUV', 'scale':40.8}],
                            'stretch':2.,
                            'Q':7.5,
                            'minimum':- 0.001}


def image(plateifu=None, version=None, clobber=True,
          bandset=None):
    """Write images for a plate-ifu

    Parameters
    ----------

    plateifu : str
        plate-ifu designation

    version : str
        version of data

    clobber : bool
        if True, clobber existing file

    bandset : str
        band set to use
"""

    plate, ifu = [int(x) for x in plateifu.split('-')]

    resampled_dir = os.path.join(os.getenv('MNSA_DATA'),
                                 version, 'resampled',
                                 str(plate), str(plateifu))

    png_dir = os.path.join(os.getenv('MNSA_DATA'),
                           version, 'pngs',
                           str(plate), str(plateifu))

    settings = bandset_settings[bandset]

    images = [None, None, None]
    for indx, band in enumerate(settings['bands']):
        name = band['name']
        scale = band['scale']

        resampled_file = os.path.join(resampled_dir,
                                      'resampled-{plateifu}-{name}.fits')
        resampled_file = resampled_file.format(plateifu=plateifu,
                                               name=name)

        images[indx] = fitsio.read(resampled_file) * scale

    outfile = os.path.join(png_dir,
                           'manga-{plateifu}-resampled-{bandset}.png')
    outfile = outfile.format(bandset=bandset, plateifu=plateifu)

    viz.make_lupton_rgb(images[0], images[1], images[2],
                        minimum=settings['minimum'],
                        stretch=settings['stretch'], Q=settings['Q'],
                        filename=outfile)

    return


class Resample(object):
    """Resample object for resampling images

    Parameters
    ----------

    image : ndarray of float
        native sampling image

    invvar : ndarray of float
        native sampling inverse variance

    input_header : dict
        header with native WCS 

    Attributes
    ----------

    image : ndarray of float
        native sampling image

    input_header : dict
        header with native WCS 

    input_pixscale : float
        native pixel scale in arcsec

    input_wcs : astropy.wcs.WCS object
        native WCS information

    invvar : ndarray of float
        native sampling inverse variance

    output_header : dict
        header for output

    output_pixscale : float
        output pixel scale in arcsec

    output_psf : ndarray of float
        output point spread function

    output_wcs : astropy.wcs.WCS object
        output WCS information

    Notes
    -----

    In general, this is only useful if the output PSF is 
    much bigger than the input PSF.

    Assumes a nearly constant PSF and pixelscale, and same
    orientation of output and input.
"""
    def __init__(self, image=None, invvar=None, input_header=None):
        self.image = image
        self.invvar = invvar
        self.input_header = input_header
        self.input_wcs = wcs.WCS(header=self.input_header)
        self.input_pixscale = self._find_pixscale(self.input_wcs,
                                                  self.input_header)
        self.output_header = None
        self.output_psf = None
        self.output_pixscale = None
        return

    def _find_pixscale(self, wcs, header):
        """Find pixel scale with small offsets"""
        offset = 5.
        raref = np.float64(header['CRVAL1'])
        decref = np.float64(header['CRVAL2'])
        decoff = decref + offset / 3600.
        radec = np.zeros((2, 2), dtype=np.float64)
        radec[0, 0] = raref
        radec[0, 1] = decref
        radec[1, 0] = raref
        radec[1, 1] = decoff
        xy = wcs.all_world2pix(radec, 1, ra_dec_order=True)
        xyoff = np.sqrt((xy[0, 0] - xy[1, 0])**2 +
                        (xy[0, 1] - xy[1, 1])**2)
        return(offset / xyoff)

    def set_output_header(self, header=None):
        """Set the output header

        Parameters
        ----------

        header : dict
            header for output

        Notes
        -----

        Sets the attributes output_header, output_wcs (to
        the corresponding astropy.wcs.WCS object), and
        output_pixscale (to arcsec per pixel for outputs).
"""
        self.output_header = header
        self.output_wcs = wcs.WCS(header=self.output_header, naxis=[1, 2])
        self.output_pixscale = self._find_pixscale(self.output_wcs,
                                                   self.output_header)
        return

    def set_output_psf(self, psf=None):
        self.output_psf = psf
        return

    def _set_output_psf_interp(self):
        """Set _output_psf_interp attribute based on output_psf"""
        nx, ny = self.output_psf.shape
        x = np.arange(nx, dtype=np.float32) - np.float32(nx) / 2. + 0.5
        y = np.arange(ny, dtype=np.float32) - np.float32(ny) / 2. + 0.5
        self._output_psf_interp = interpolate.interp2d(x, y,
                                                       self.output_psf,
                                                       kind='cubic',
                                                       copy=True,
                                                       bounds_error=False,
                                                       fill_value=0.)
        return

    def _set_output_psf_resampled(self):
        """Set output_psf_resampled attribute based on output_psf"""
        nxo, nyo = self.output_psf.shape
        nxi = np.float32(nxo) * self.output_pixscale / self.input_pixscale
        nxi = (np.int32(nxi) // 2) * 2 + 1
        nyi = np.float32(nyo) * self.output_pixscale / self.input_pixscale
        nyi = (np.int32(nyi) // 2) * 2 + 1
        self.output_psf_resampled = np.zeros((nxi, nyi), dtype=np.float32)
        xi_axis = np.arange(nxi, dtype=np.float32) - np.float32(nxi) / 2. + 0.5
        yi_axis = np.arange(nyi, dtype=np.float32) - np.float32(nyi) / 2. + 0.5
        xo_axis = xi_axis * self.input_pixscale / self.output_pixscale
        yo_axis = yi_axis * self.input_pixscale / self.output_pixscale
        self.output_psf_resampled = self._output_psf_interp(xo_axis, yo_axis).reshape(nxi, nyi)
        self.output_psf_resampled = np.float32(self.output_psf_resampled)
        return

    def downsample(self):
        """Downsample image to output sampling and PSF

        Returns
        -------

        image : ndarray of float
            downsampled image

        var : ndarray of float
            variance of downsampled image
"""

        if((self.output_header is None) | (self.output_psf is None)):
            print("Must set output_header and output_psf to downsample")

        self._set_output_psf_interp()
        self._set_output_psf_resampled()

        image_smoothed = signal.fftconvolve(self.image,
                                            self.output_psf_resampled,
                                            mode='same')
        nxi, nyi = image_smoothed.shape
        xi = np.arange(nxi, dtype=np.float32)
        yi = np.arange(nyi, dtype=np.float32)

        nxo = np.int32(self.output_header['NAXIS1'])
        nyo = np.int32(self.output_header['NAXIS2'])

        y = np.outer(np.ones(nxo, dtype=np.float32),
                     np.arange(nyo, dtype=np.float32))
        x = np.outer(np.arange(nxo, dtype=np.float32),
                     np.ones(nyo, dtype=np.float32))

        invvar_fixed = self.invvar

        iz = np.where(invvar_fixed <= 0.)[0]
        if(len(iz) > 0):
            invvar_fixed[iz] = np.median(self.invvar)
        var = 1. / invvar_fixed
        var_smoothed = signal.fftconvolve(var, self.output_psf_resampled**2,
                                          mode='same')

        # Find output pixel locations in input pixel grid
        rao, deco = self.output_wcs.all_pix2world(x.flatten(), y.flatten(), 0,
                                                  ra_dec_order=True)
        xoi, yoi = self.input_wcs.all_world2pix(rao, deco, 0, ra_dec_order=True)

        image_interp = interpolate.RectBivariateSpline(xi, yi, image_smoothed)
        var_interp = interpolate.RectBivariateSpline(xi, yi, var_smoothed)

        image_downsampled = np.float32(image_interp(xoi, yoi, grid=False).reshape(nxo, nyo))
        var_downsampled = np.float32(var_interp(xoi, yoi, grid=False).reshape(nxo, nyo))

        return(image_downsampled, var_downsampled)
