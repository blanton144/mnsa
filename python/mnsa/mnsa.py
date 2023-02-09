import os
import numpy as np
import matplotlib.image
import matplotlib.cm
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

        resampled_base = os.path.join(resampled_dir,
                                      'resampled-{plateifu}')
        self.resampled_base = resampled_base.format(plateifu=self.plateifu)

        native_base = os.path.join(native_dir,
                                   '{plateifu}-custom')
        self.native_base = native_base.format(plateifu=self.plateifu)

        self.cube = None
        self.maps = None
        self.resampled_mask = None
        self.resampled = dict()
        self.native = dict()
        return

    def read_cube(self):
        """Read FITS file into cube attribute"""
        self.cube = fitsio.FITS(self.manga_base + '.fits.gz')
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
        self.resampled_mask = fitsio.FITS(self.resampled_base + '-mask.fits')
        return

    def read_native(self, band=None, itype=None):
        """Read native image FITS file into native dict attribute"""
        self.native[band, itype] = fitsio.FITS(self.native_base + '-{i}-{b}.fits.fz'.format(b=band, i=itype))
        return

    def read_resampled(self, band=None):
        """Read resampled image FITS file into resampled dict attribute"""
        self.resampled[band] = fitsio.FITS(self.resampled_base + '-{b}.fits'.format(b=band))
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
        self.maps = fitsio.FITS(dap_maps)
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
"""
        im = self.read_image(channel=0, ext=ext, file='maps')

        im_scaled = (im + vlimit) / (2. * vlimit)
        im_scaled[im_scaled < 0.] = 0.
        im_scaled[im_scaled > 1.] = 1.
        rgb = matplotlib.cm.bwr(im_scaled)
        return(rgb)
