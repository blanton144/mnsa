import os
import re
import numpy as np
import matplotlib.image
import matplotlib.cm
import mnsa.kernel
import mnsa.bandscale as bandscale
import fitsio
import mnsa.utils.configuration as configuration
import mnsa.imagetypes as imagetypes


i = imagetypes.ImageTypes()


class MNSA(object):
    """MNSA object to read cube and images

    Parameters
    ----------

    version : str
        version of results

    plateifu : str
        Plate-IFU of result

    dr17 : bool
        if True, read standard DR17 versions instead of MNSA

    Attributes
    ----------

    cfg : mnsa.utils.configuration.MNSAConfig object
        configuration information for this version

    cube : fitsio.FITS object
        cube information from FITS file (initialized to None)

    dr17 : bool
        if True, read standard DR17 versions instead of MNSA

    ifu : int
        IFU number

    manga_base : str
        MaNGA file name base

    maps : fitsio.FITS object
        maps information from FITS file (initialized to None)

    plate : int
        plate number

    plateifu : str
        plate-ifu string

    psf : dict
        dictionary to store PSF information (initialized to empty)

    version : str
        version of data to use

    Notes
    -----

    Creating the object does not automatically read in the 
    cube or maps information. This needs to be done explicitly
    with read_cube() or read_maps().

    Reads data under the directory $MNSA_DATA/{version}
    (unless DR17 is set to true).
"""
    def __init__(self, version=None, plateifu=None, dr17=False):
        mnsa_config = configuration.MNSAConfig(version=version)
        cfg = mnsa_config.cfg

        plate, ifu = [int(x) for x in plateifu.split('-')]

        self.plate = plate
        self.ifu = ifu
        self.plateifu = plateifu
        self.version = version
        self.cfg = cfg
        self.dr17 = dr17

        manga_dir = os.path.join(os.getenv('MNSA_DATA'),
                                 self.version, 'manga', 'redux',
                                 self.version,
                                 str(self.plate), 'stack')

        resampled_dir = os.path.join(os.getenv('MNSA_DATA'),
                                     self.version, 'resampled',
                                     str(self.plate),
                                     self.plateifu)

        native_dir = os.path.join(os.getenv('MNSA_DATA'),
                                  self.version, 'imaging',
                                  str(self.plate),
                                  self.plateifu)

        png_dir = os.path.join(os.getenv('MNSA_DATA'),
                               self.version, 'pngs',
                               str(self.plate),
                               self.plateifu)

        if(self.dr17):
            manga_dir = os.path.join('/uufs', 'chpc.utah.edu',
                                     'common', 'home', 'sdss50',
                                     'dr17', 'manga', 'spectro',
                                     'redux', 'v3_1_1', str(self.plate),
                                     'stack')
                                     # 'analysis', 'v3_1_1', '3.1.0',

        manga_base = os.path.join(manga_dir,
                                  'manga-{plateifu}-LOGCUBE')
        self.manga_base = manga_base.format(plateifu=self.plateifu)

        png_base = os.path.join(png_dir,
                                'manga-{plateifu}')
        self.png_dir = png_dir
        self.png_base = png_base.format(plateifu=self.plateifu)

        if(self.dr17):
            resampled_base = os.path.join(resampled_dir,
                                          'resampled-dr17-{plateifu}')
        else:
            resampled_base = os.path.join(resampled_dir,
                                          'resampled-{plateifu}')
        self.resampled_base = resampled_base.format(plateifu=self.plateifu)

        apimages_base = os.path.join(resampled_dir,
                                     'apimages-{plateifu}')
        self.apimages_base = apimages_base.format(plateifu=self.plateifu)

        native_base = os.path.join(native_dir,
                                   '{plateifu}-custom')
        self.native_base = native_base.format(plateifu=self.plateifu)

        self.cube = None
        self.maps = None
        self.rss = None
        self.resampled_mask = None
        self.resampled = dict()
        self.apimages = None
        self.native = dict()
        self.psf = dict()
        return

    def read_cube(self):
        """Read FITS file into cube attribute"""
        if(self.cube is None):
            try:
                self.cube = fitsio.FITS(self.manga_base + '.fits.gz')
            except OSError:
                raise OSError("Could not read in CUBE: {dm}".format(dm=self.manga_base + '.fits.gz'))
        return

    def read_rss(self):
        """Read RSS FITS file into rss attribute"""
        if(self.rss is None):
            rssfile = os.path.join(os.getenv('MANGA_ROOT'),
                                   'spectro', 'redux',
                                   'v3_1_1', str(self.plate),
                                   'stack', 'manga-{p}-LOGRSS.fits.gz')
            rssfile = rssfile.format(p=self.plateifu)
            self.rss = fitsio.FITS(rssfile)
        return

    def read_drp(self):
        """Read DRP row"""
        drpfile = os.path.join(os.getenv('MANGA_SPECTRO_REDUX'),
                               'v3_1_1', 'drpall-v3_1_1.fits')
        drpall = fitsio.read(drpfile)
        ii = np.where(drpall['plateifu'] == self.plateifu)[0][0]
        self.drp = drpall[ii]
        return

    def read_resampled_mask(self):
        """Read resampled mask FITS file into resampled_mask attribute"""
        if(self.resampled_mask is None):
            self.resampled_mask = fitsio.FITS(self.resampled_base + '-mask.fits')
            self.resampled_mask_ix = self.resampled_mask['X'].read()
            self.resampled_mask_iy = self.resampled_mask['Y'].read()
        return

    def read_native(self, band=None, itype=None):
        """Read native image FITS file into native dict attribute"""
        if((band, itype) not in self.native):
            self.native[band, itype] = fitsio.FITS(self.native_base + '-{i}-{b}.fits.fz'.format(b=band, i=itype))
        return

    def read_resampled(self, band=None):
        """Read resampled image FITS file into resampled dict attribute"""
        if(band not in self.resampled):
            self.resampled[band] = fitsio.FITS(self.resampled_base + '-{b}.fits'.format(b=band))
        return

    def read_apimages(self, apband=None):
        """Read aperture correction image file"""
        if(self.apimages is None):
            self.apimages = fitsio.FITS(self.apimages_base + '-{b}.fits'.format(b=apband))
        return

    def read_maps(self):
        """Read FITS file into maps attribute"""
        daptype = self.cfg['MANGA']['daptype']
        dap_dir = os.path.join(os.getenv('MNSA_DATA'),
                               self.version, 'manga', 'analysis',
                               self.version, self.version, daptype,
                               str(self.plate), str(self.ifu))
        if(self.dr17):
            dap_dir = os.path.join(os.getenv('MANGA_ROOT'),
                                   'spectro', 'analysis',
                                   'v3_1_1', '3.1.0',
                                   daptype,
                                   str(self.plate), str(self.ifu))
        dap_file = os.path.join(dap_dir, 'manga-{p}-{t}-{d}.fits.gz')
        dap_maps = dap_file.format(p=self.plateifu, d=daptype,
                                   t='MAPS')
        try:
            self.maps = fitsio.FITS(dap_maps)
        except OSError:
            raise OSError("Could not read in MAPS: {dm}".format(dm=dap_maps))
        
        exts = [x.get_extname() for x in self.maps]

        self.ichannel = dict()
        for ext in exts:
            if(ext == ''):
                continue
            self.ichannel[ext] = dict()
            try:
                hdr = self.maps[ext].read_header()
            except IOError:
                raise ValueError("Could not read extension {ext}".format(ext=ext))
            for k in hdr:
                m = re.match('^C([0-9]*)$', k)
                if(m is not None):
                    self.ichannel[ext][hdr[k]] = int(m[1]) - 1
        return

    def read_image(self, channel=None, ext=None, file=None):
        """Read a specific image from the cube or maps

        Parameters
        ----------

        channel : str
            channel of maps extension to return (if file == 'maps')

        ext : str or int
            name or number of extension of cube or maps

        file : str
            'maps', 'cube', or 'resampled', depending on which type of image wanted

        Notes
        -----
"""
        if(file == 'maps'):
            if(self.maps is None):
                self.read_maps()
                if(self.maps is None):
                    return
            hdr = self.maps[ext].read_header()
            if(channel):
                ichannel = -1
                for x in hdr:
                    if(x[0] == 'C'):
                        if(hdr[x] == channel):
                            ichannel = int(x[1:]) - 1
                            break
                if(ichannel == -1):
                    print("No channel {ch}".format(ch=channel))
                    return
                im = self.maps[ext][ichannel, :, :]
                shape = im.shape
                im = im.reshape(shape[1:])
            else:
                im = self.maps[ext][:, :]
            return(im)
        if(file == 'cube'):
            self.read_cube()
            if(ext != 'FLUX'):
                im = self.cube[ext].read()
                return(im)
            wave = self.cube['WAVE'].read()
            flux = self.cube['FLUX'].read()
            im = bandscale.image(band=channel, wave=wave, cube=flux)
            return(im)
        if(file == 'resampled'):
            self.read_resampled(band=channel)
            return(self.resampled[channel][ext].read())
        if(file == 'native'):
            self.read_native(band=channel, itype=ext)
            return(self.native[channel, ext][1].read())

        print("No file type {f}".format(f=file))
        return(None)

    def hex_mask(self):
        """Calculate and return 2D mask with good spectrum"""
        self.read_cube()
        mask = self.cube['MASK'].read()
        nokwave = (mask == 0).sum(axis=0)
        maxokwave = nokwave.max()
        hex_mask = (nokwave > (maxokwave * 0.9))
        return(hex_mask)

    def ellipse_mask(self, ba=1., pa=0., sma=None, manga=True):
        """Create elliptical mask

        Parameters
        ----------

        ba : np.float32
            axis ratio b/a for applied elliptical aperture

        pa : np.float32
            position angle (deg E of N) for applied elliptical aperture

        sma : np.float32
            semi-major axis of applied elliptical aperture (arcsec)

        manga : bool
            output manga mask

        Returns
        -------

        mask : ndarray of np.bool
            pixels to keep within ellipse
"""
        self.read_resampled_mask()
        hdr = self.resampled_mask['MASK'].read_header()
        pixscale = np.abs(np.float32(hdr['PC1_1'])) * 3600.
        nx = np.int32(hdr['NAXIS1'])
        ny = np.int32(hdr['NAXIS2'])
        x1 = np.arange(nx, dtype=np.int32) - np.float32(nx // 2)
        y1 = np.arange(ny, dtype=np.int32) - np.float32(ny // 2)
        x, y = np.meshgrid(x1, y1, indexing='ij')
        parad = pa * np.pi / 180.
        xp = x * np.cos(parad) - y * np.sin(parad)
        yp = x * np.sin(parad) + y * np.cos(parad)
        yp = yp / ba
        rp = np.sqrt(xp**2 + yp**2) * pixscale
        ellipse_mask = rp < sma
        if(manga):
            x_manga2full = self.resampled_mask['X'].read()
            y_manga2full = self.resampled_mask['Y'].read()
            ellipse_mask = ellipse_mask[x_manga2full,
                                        y_manga2full]
        return(ellipse_mask)

    def hex_spectrum(self, ba=1., pa=0., sma=None, return_mask=False):
        """Calculate and return total spectrum in MaNGA footprint

        Parameters
        ----------

        ba : np.float32
            axis ratio b/a for applied elliptical aperture

        pa : np.float32
            position angle (deg E of N) for applied elliptical aperture

        sma : np.float32
            semi-major axis of applied elliptical aperture (arcsec)

        return_mask : bool
            if True, return the mask used to sum the spectra

        Returns
        -------

        hex_spectrum : ndarray of np.float32
            summed spectrum across cube (masked if specified)

        hex_ivar : ndarray of np.float32
            inverse variance of hex_spectrum (masked if specified)

        hex_mask : ndarray of bool
            2D mask used (if return_mask set)

        Notes
        -----

        If sma is not None, an elliptical aperture is applied
        with the specified parameters in addition to the hexagonal
        mask.
"""
        hex_mask = self.hex_mask()
        if(sma is not None):
            ellipse_mask = self.ellipse_mask(ba=ba, pa=pa, sma=sma, manga=True)
            hex_mask = hex_mask & ellipse_mask
        flux = self.cube['FLUX'].read()
        ivar = self.cube['IVAR'].read()
        mask = self.cube['MASK'].read()
        hex_mask_expanded = np.tile(hex_mask, (flux.shape[0], 1, 1))
        ok = ((mask == 0) & (hex_mask_expanded > 0))
        nok = np.float32(ok.sum(axis=1).sum(axis=1))
        scale = np.float32(hex_mask.sum()) / nok
        hex_spec = flux[:, hex_mask].sum(axis=1) * scale
        hex_ivar = 1. / (1. / ivar[:, hex_mask]).sum(axis=1) / scale**2
        if(return_mask):
            return(hex_spec, hex_ivar, hex_mask)
        else:
            return(hex_spec, hex_ivar)

    def fiber_spectrum(self, radius=None, seeing_fwhm=None,
                       return_aperture=False):
        """Calculate and return total spectrum in seeing-apodized fiber

        Parameters
        ----------

        radius : np.float32
            fiber aperture radius (arcsec)

        seeing_fwhm : np.float32
            FWHM of seeing (arcsec)

        return_aperture : bool
            if True, return the mask used to sum the spectra

        Returns
        -------

        fiber_spectrum : ndarray of np.float32
            summed spectrum across cube

        fiber_spectrum_mask: ndarray of bool
            True or False if spectrum is OK at each wavelength

        fiber_aperture : ndarray of bool
            2D aperture used (if return_mask set)
"""
        hex_mask = self.hex_mask()
        nx = hex_mask.shape[0]
        k = mnsa.kernel.Kernel(seeing=seeing_fwhm, radius=radius)
        if(self.dr17):
            dx = 0.5
        else:
            dx = 0.75
        aperture = k.grid(dx=dx, nx=nx)

        aperture = aperture * np.float32(hex_mask)
        inaperture = aperture > 0

        flux = self.cube['FLUX'].read()
        ivar = self.cube['IVAR'].read()
        mask = self.cube['MASK'].read()
        aperture_expanded = np.tile(aperture, (flux.shape[0], 1, 1))
        ok = ((mask == 0) & (aperture_expanded > 0))
        sumok = (aperture_expanded * np.float32(ok)).sum(axis=1).sum(axis=1)
        scale = (np.float32(sumok > 0) * np.float32(aperture.sum()) /
                 (sumok + np.float32(sumok == 0)))
        aperture_spec = (flux * aperture_expanded).sum(axis=1).sum(axis=1) * scale
        aperture_ok = (sumok > 0)
        if(return_aperture):
            return(aperture_spec, aperture_ok, aperture)
        else:
            return(aperture_spec, aperture_ok)

    def hex_maggies(self, band=None, ba=1., pa=0., sma=None,
                    return_mask=False):
        """Calculate and return a flux in the MaNGA footprint

        Parameters
        ----------

        band : str
            band to measure

        ba : np.float32
            axis ratio b/a for applied elliptical aperture

        pa : np.float32
            position angle (deg E of N) for applied elliptical aperture

        sma : np.float32
            semi-major axis of applied elliptical aperture (arcsec)

        return_mask : bool
            if True, return the mask used to sum the spectra

        Returns
        -------

        hex_maggies : np.float32
            flux across cube (masked if specified)

        hex_ivar : np.float32
            inverse variance of hex_maggies (masked if specified)

        hex_mask : ndarray of bool
            2D mask used (if return_mask set)

        Notes
        -----

        If sma is not None, an elliptical aperture is applied
        with the specified parameters in addition to the hexagonal
        mask.

        Inverse variance is based on MC images in resampled.
"""
        self.read_resampled(band=band)
        image = self.resampled[band]['IMAGE'].read()
        ivar = self.resampled[band]['IVAR'].read()
        mc = self.resampled[band]['MC'].read()

        results = self._image_hex_maggies(image=image, ivar=ivar,
                                          ba=ba, pa=pa, sma=sma,
                                          return_mask=True)
        hex_maggies, dum_ivar, hex_mask = results

        nmc = mc.shape[0]
        hex_maggies_mc = np.zeros(nmc, dtype=np.float32)
        for imc in np.arange(nmc, dtype=np.int32):
            results = self._image_hex_maggies(image=mc[imc, :, :],
                                              ivar=ivar,
                                              input_mask=hex_mask)
            tmp_hex_maggies_mc, dum_ivar = results
            hex_maggies_mc[imc] = tmp_hex_maggies_mc

        hex_maggies_ivar = 1. / hex_maggies_mc.std()**2

        if(return_mask):
            return(hex_maggies, hex_maggies_ivar, hex_mask)
        else:
            return(hex_maggies, hex_maggies_ivar)

    def hex_apcorr(self, band=None, apband=None, ba=1., pa=0., sma=None,
                   return_mask=False):
        """Calculate and return aperture correction factor

        Parameters
        ----------

        band : str
            band of aperture correction

        apband : str
            band to correct to

        ba : np.float32
            axis ratio b/a for applied elliptical aperture

        pa : np.float32
            position angle (deg E of N) for applied elliptical aperture

        sma : np.float32
            semi-major axis of applied elliptical aperture (arcsec)

        return_mask : bool
            if True, return the mask used to sum the spectra

        Returns
        -------

        apcorr : np.float32
            aperture correction

        hex_mask : ndarray of bool
            2D mask used (if return_mask set)

        Notes
        -----

        If sma is not None, an elliptical aperture is applied
        with the specified parameters in addition to the hexagonal
        mask.
"""
        self.read_apimages(apband=apband)
        apimage = self.apimages[apband].read()
        image = self.apimages[band].read()
        ivar = np.ones(image.shape, dtype=np.float32)

        results = self._image_hex_maggies(image=apimage, ivar=ivar,
                                          ba=ba, pa=pa, sma=sma,
                                          return_mask=return_mask)
        if(return_mask):
            ap_maggies, ap_maggies_ivar, hex_mask = results
        else:
            ap_maggies, ap_maggies_ivar = results

        results = self._image_hex_maggies(image=image, ivar=ivar,
                                          ba=ba, pa=pa, sma=sma,
                                          return_mask=return_mask)
        if(return_mask):
            maggies, maggies_ivar, hex_mask = results
        else:
            maggies, maggies_ivar = results

        apcorr = ap_maggies / maggies

        if(return_mask):
            return(apcorr, hex_mask)
        else:
            return(apcorr)

    def _image_hex_maggies(self, image=None, ivar=None, ba=1., pa=0., sma=None,
                           input_mask=None, return_mask=False):
        """Calculate and return a flux from a given image in hex footprint

        Parameters
        ----------

        image : ndarray of np.float32
           image to calculate

        ivar : ndarray of np.float32
           ivar of image to calculate

        ba : np.float32
            axis ratio b/a for applied elliptical aperture

        pa : np.float32
            position angle (deg E of N) for applied elliptical aperture

        sma : np.float32
            semi-major axis of applied elliptical aperture (arcsec)
        
        input_mask : ndarray of bool
            input mask (overrides hex and other parameters)

        return_mask : bool
            if True, return the mask used to sum the spectra

        Returns
        -------

        hex_maggies : np.float32
            flux across cube (masked if specified)

        hex_ivar : np.float32
            inverse variance of hex_maggies (masked if specified)

        hex_mask : ndarray of bool
            2D mask used (if return_mask set)

        Notes
        -----

        If sma is not None, an elliptical aperture is applied
        with the specified parameters in addition to the hexagonal
        mask.

        Inverse variance is propagated from ivar image (only
        accurate if the image has diagonal covariance).
"""
        self.read_resampled_mask()
        ix = self.resampled_mask_ix
        iy = self.resampled_mask_iy

        if(input_mask is None):
            hex_mask = self.hex_mask()
            if(sma is not None):
                ellipse_mask = self.ellipse_mask(ba=ba, pa=pa, sma=sma, manga=True)
                hex_mask = hex_mask & ellipse_mask
        else:
            hex_mask = input_mask

        hex_maggies = (image[ix, iy] * hex_mask).sum()
        hex_maggies_ivar = 1. / ((1. / ivar[ix, iy]) * hex_mask).sum()

        if(return_mask):
            return(hex_maggies, hex_maggies_ivar, hex_mask)
        else:
            return(hex_maggies, hex_maggies_ivar)

    def image(self, imagetype=None, filename=None, invert=False,
              stretch=None, Q=None, minimum=None, Rscale=None,
              Gscale=None, Bscale=None):
        """Make image based on cube and maps

        Parameters
        ----------

        imagetype : str
            type of image (must be specified in mnsa.imagetypes.ImageTypes)

        filename : str
            output file name

        invert : bool
            whether to invert image (default False)

        stretch : np.float32
            stretch value; if not None overrides imagetype-based value (default None)

        Q : np.float32
            nonlinearity Q value; if not None overrides imagetype-based value (default None)

        minimum : np.float32
            minimum offset value; if not None overrides imagetype-based value (default None)

        Rscale : np.float32
            R-channel scale value; if not None overrides imagetype-based value (default None)

        Gscale : np.float32
            G-channel scale value; if not None overrides imagetype-based value (default None)

        Bscale : np.float32
            B-channel scale value; if not None overrides imagetype-based value (default None)

        Notes
        -----

        mnsa.imagetypes.ImageTypes is a singleton with a dictionary
        attribute "types" that contains the parameters defining
        the image appearance.

        If dr17=True, then the image is multiplied by 1.5**2
        in order to put it on the same effective scale
        (i.e. the same appearance for the same surface
        brightness value).

        The scaling is such that fluxes near zero are black.
        If you want higher flux to be darker, set invert=True.
"""
        if(imagetype not in i.types):
            print("No such image type {i}".format(i=imagetype))
            return
        itype = i.types[imagetype]
        if(Rscale is None):
            Rscale = itype['Rscale']
        if(Gscale is None):
            Gscale = itype['Gscale']
        if(Bscale is None):
            Bscale = itype['Bscale']
        if(stretch is None):
            stretch = itype['stretch']
        if(Q is None):
            Q = itype['Q']
        if(minimum is None):
            minimum = itype['minimum']

        images = dict()
        for imagename in itype['images']:
            spec = itype['images'][imagename]
            im = self.read_image(channel=spec['channel'],
                                 ext=spec['ext'],
                                 file=spec['file'])
            images[imagename] = im
            shape = im.shape
        rimage = np.zeros(shape, dtype=np.float32)
        for imagename in itype['R']:
            rimage = rimage + itype['R'][imagename] * images[imagename]
        rimage = rimage * Rscale
        gimage = np.zeros(shape, dtype=np.float32)
        for imagename in itype['G']:
            gimage = gimage + itype['G'][imagename] * images[imagename]
        gimage = gimage * Gscale
        bimage = np.zeros(shape, dtype=np.float32)
        for imagename in itype['B']:
            bimage = bimage + itype['B'][imagename] * images[imagename]
        bimage = bimage * Bscale

        if(self.dr17):
            rimage = rimage * 2.25
            gimage = gimage * 2.25
            bimage = bimage * 2.25

        rgb = imagetypes.rgb_stretch(rimage=rimage, gimage=gimage,
                                     bimage=bimage, stretch=stretch, Q=Q,
                                     minimum=minimum, filename=filename,
                                     invert=invert)
        return(rgb)

    def velocity_image(self, ext=None, filename=None,
                       vlimit=200.):
        """Make velocity image

        Parameters
        ----------

        ext : str
            extension to treat as a velocity

        filename : str
            output file name

        Notes
        -----

        In DAP MAPS files, 'STELLAR_VEL' and 'EMLINE_GVEL'
        are the stellar and gas velocity extension names.
"""
        im = self.read_image(channel=0, ext=ext, file='maps')

        im_scaled = (im + vlimit) / (2. * vlimit)
        im_scaled[im_scaled < 0.] = 0.
        im_scaled[im_scaled > 1.] = 1.
        rgb = matplotlib.cm.bwr(im_scaled)
        return(rgb)

    def read_psf(self, band=None):
        """Read PSF image

        Parameters
        ----------

        band : str
            band for PSF ("g", "r", "i", "z")

        Notes
        -----

        Puts psf (2-D ndarray of np.float32) entry of 
        attribute "psf", a dictionary
"""
        self.read_cube()
        if(band in self.psf):
            return
        extname = "{band}PSF".format(band=band.upper())
        self.psf[band] = self.cube[extname].read()
        return

    def central_flux(self, channel=None, ext='EMLINE_GFLUX',
                     psfband='r', rlimit=3.5):
        """Calculate a central PSF-weighted flux

        Parameters
        ----------

        channel : str
            name of emission line channel to calculate

        ext : str
            name of extension for calculated spaxel fluxes

        psfband : str
            band to use for PSF weighting

        thlimit : np.float32
            radius limit for PSF in arcsec
        
        Returns
        -------

        flux : np.float32
            measured PSF-weighted flux in center of image

        ivar : np.float32
            inverse variance of measured PSF-weighted flux in center of image

        nbad : np.int32
            number of pixels within theta limit with ivar=0
"""
        self.read_maps()
        if(self.maps is None):
            return(0., 0., 0)

        self.read_psf(band=psfband)

        ichannel = self.ichannel[ext][channel]

        # Read in data
        im = self.maps[ext][ichannel, :, :].squeeze()
        ivar = self.maps[ext + '_IVAR'][ichannel, :, :].squeeze()

        # Create PSF weighting mask
        #   - first limit in radius
        circle_mask = self.ellipse_mask(sma=rlimit)

        # Count how many ivar=0 pixels there are
        try:
            nbad = np.int32(((ivar == 0) & (circle_mask > 0)).sum())
        except:
            raise ValueError("ivar.shape = {ivs}, circle_mask.shape = {cms}".format(ivs=ivar.shape,
                                                                                    cms=circle_mask.shape))
        
        psf = self.psf[psfband] * circle_mask
        #   - then excise negatives        
        psf[psf < 0.] = 0.
        #   - if no pixels left, return flux=0, ivar=0
        if((psf > 0.).sum() == 0):
            return(0., 0., nbad)
        #   - normalize to this PSF model
        psf = psf / psf.sum()
        #   - now excise poorly measured pixels (but do not renorm)
        psf[ivar <= 0.] = 0.

        # If no pixels left, return flux=0, ivar=0
        if((psf > 0.).sum() == 0):
            return(0., 0., nbad)

        central_flux = (im * psf).sum() / (psf * psf).sum()

        is_positive = (psf > 0)
        if(self.dr17):
            nx, ny = im.shape
            x1 = np.arange(nx, dtype=np.int32) - np.float32(nx // 2)
            y1 = np.arange(ny, dtype=np.int32) - np.float32(ny // 2)
            x, y = np.meshgrid(x1, y1, indexing='ij')
            ipositive = np.where(is_positive & (ivar > 0.))
            x = x[ipositive].flatten()
            y = y[ipositive].flatten()
            sigma = (1. / np.sqrt(ivar[ipositive])).flatten()
            covar = np.zeros((len(x), len(x)), dtype=np.float32)
            for i in np.arange(len(x), dtype=np.int32):
                dd = (x[i] - x)**2 + (y[i] - y)**2
                covar[i, :] = np.exp(- dd / 7.37) * sigma[i] * sigma
            numer = ((psf[ipositive]**2).sum())**2
            denom = psf[ipositive].dot(covar).dot(psf[ipositive])
            central_ivar = numer / denom
            #
            #central_ivar2 = (((psf[is_positive]**2).sum())**2 /
            #                 (((psf[is_positive]**2) / ivar[is_positive]).sum()))
            #print(central_ivar, central_ivar2)
        else:
            central_ivar = (((psf[is_positive]**2).sum())**2 /
                            (((psf[is_positive]**2) / ivar[is_positive]).sum()))

        return(central_flux, central_ivar, nbad)
