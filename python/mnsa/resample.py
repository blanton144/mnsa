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
                            'stretch':4.8,
                            'Q':2.5,
                            'minimum':- 0.5}
bandset_settings['wise'] = {'bands':[{'name':'W3', 'scale':0.05},
                                     {'name':'W2', 'scale':1.5},
                                     {'name':'W1', 'scale':1.8}],
                            'stretch':10.,
                            'Q':2.5,
                            'minimum':- 0.5}

bandset_settings['galex'] = {'bands':[{'name':'g', 'scale':0.5},
                                      {'name':'NUV', 'scale':5.5},
                                      {'name':'FUV', 'scale':5.8}],
                            'stretch':1.,
                            'Q':2.5,
                            'minimum':- 0.05}


def image(plateifu=None, version=None, clobber=True,
          bandset=None):

    mnsa_config = configuration.MNSAConfig(version=version)
    cfg = mnsa_config.cfg

    release = cfg['MANGA']['release']
    plate, ifu = [int(x) for x in plateifu.split('-')]

    resampled_dir = os.path.join(os.getenv('MNSA_DATA'),
                                 version, 'resampled',
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

    outfile = os.path.join(resampled_dir,
                           'resampled-{plateifu}-{bandset}.png')
    outfile = outfile.format(bandset=bandset, plateifu=plateifu)
        
    viz.make_lupton_rgb(images[0], images[1], images[2],
                        minimum=settings['minimum'],
                        stretch=settings['stretch'], Q=settings['Q'],
                        filename=outfile)


class Resample(object):
    """Resample object for resampling images

    Assumes a nearly constant PSF and pixelscale, and same orientation of output and input.
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
        self.output_header = header
        self.output_wcs = wcs.WCS(header=self.output_header, naxis=[1, 2])
        self.output_pixscale = self._find_pixscale(self.output_wcs,
                                                   self.output_header)
        return

    def set_output_psf(self, psf=None):
        self.output_psf = psf
        return

    def _set_output_psf_interp(self):
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
        nxo, nyo = self.output_psf.shape
        nxi = np.float64(nxo) * self.output_pixscale / self.input_pixscale
        nxi = (np.int32(nxi) // 2) * 2 + 1
        nyi = np.float64(nyo) * self.output_pixscale / self.input_pixscale
        nyi = (np.int32(nyi) // 2) * 2 + 1
        self.output_psf_resampled = np.zeros((nxi, nyi), dtype=np.float64)
        xi_axis = np.arange(nxi, dtype=np.float64) - np.float64(nxi) / 2. + 0.5
        yi_axis = np.arange(nyi, dtype=np.float64) - np.float64(nyi) / 2. + 0.5
        xo_axis = xi_axis * self.input_pixscale / self.output_pixscale
        yo_axis = yi_axis * self.input_pixscale / self.output_pixscale
        self.output_psf_resampled = self._output_psf_interp(xo_axis, yo_axis).reshape(nxi, nyi)
        return

    def downsample(self):
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

        image_downsampled = image_interp(xoi, yoi, grid=False).reshape(nxo, nyo)
        var_downsampled = var_interp(xoi, yoi, grid=False).reshape(nxo, nyo)

        return(image_downsampled, var_downsampled)
