import os
import numpy as np
import fitsio
import mnsa.utils.configuration as configuration
import mnsa.imagetypes as imagetypes
import astropy.visualization as viz


i = imagetypes.ImageTypes()


class MNSA(object):
    """MNSA object to read cube and images

    Parameters
    ----------

    version : str
        version of results

    plateifu : str
        Plate-IFU of result

    Notes
    -----

    Reads data from $MNSA_DATA/{version}.
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

        if(self.dr17):
            manga_dir = os.path.join('/uufs', 'chpc.utah.edu',
                                     'common', 'home', 'sdss50',
                                     'dr17', 'manga', 'spectro',
                                     'redux', 'v3_1_1', str(self.plate),
                                     'stack')
                                     # 'analysis', 'v3_1_1', '3.1.0',
                                     
        manga_file = os.path.join(manga_dir,
                                  'manga-{plateifu}-LOGCUBE.fits.gz')
        self.manga_file = manga_file.format(plateifu=self.plateifu)

        self.manga_irg_png = self.manga_file.replace('.fits.gz',
                                                     '.irg.png')
        return

    def read_cube(self):

        self.cube = fitsio.FITS(self.manga_file)
        return

    def read_maps(self):

        daptype = self.cfg['MANGA']['daptype']
        dap_dir = os.path.join(os.getenv('MNSA_DATA'),
                               self.version, 'manga', 'analysis',
                               self.version, self.version, daptype,
                               str(self.plate), str(self.ifu))
        if(self.dr17):
            dap_dir = os.path.join('/uufs', 'chpc.utah.edu',
                                   'common', 'home', 'sdss50',
                                   'dr17', 'manga', 'spectro',
                                   'analysis', 'v3_1_1', '3.1.0',
                                   daptype,
                                   str(self.plate), str(self.ifu))
        dap_file = os.path.join(dap_dir, 'manga-{p}-{t}-{d}.fits.gz')
        dap_maps = dap_file.format(p=self.plateifu, d=daptype,
                                   t='MAPS')
        self.maps = fitsio.FITS(dap_maps)
        return

    def rgb_image(self, rimage=None, gimage=None, bimage=None,
                  minimum=0., stretch=1., Q=10., filename=None):
        
        rgb = viz.make_lupton_rgb(rimage, gimage, bimage, minimum=minimum,
                                  stretch=stretch, Q=Q, filename=filename)
        return(rgb)

    def read_image(self, channel=None, ext=None, file=None):
        if(file == 'maps'):
            hdr = self.maps[ext].read_header()
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
            return(im)
        if(file == 'cube'):
            im = self.cube[ext].read()
            return(im)
            
        print("No file type {f}".format(f=file))
        return(None)

    def image(self, imagetype=None, minimum=None, stretch=None, Q=None, 
              rscale=None, gscale=None, bscale=None, filename=None):
        if(imagetype not in i.types):
            print("No such image type {i}".format(i=imagetype))
            return
        itype = i.types[imagetype]
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
        rimage = rimage * itype['Rscale']
        gimage = np.zeros(shape, dtype=np.float32)
        for imagename in itype['G']:
            gimage = gimage + itype['G'][imagename] * images[imagename]
        gimage = gimage * itype['Gscale']
        bimage = np.zeros(shape, dtype=np.float32)
        for imagename in itype['B']:
            bimage = bimage + itype['B'][imagename] * images[imagename]
        bimage = bimage * itype['Bscale']

        if(self.dr17):
            rimage = rimage * 2.25
            gimage = gimage * 2.25
            bimage = bimage * 2.25
        
        stretch = itype['stretch']
        Q = itype['Q']
        minimum = itype['minimum']
        rgb = self.rgb_image(rimage=rimage, gimage=gimage, bimage=bimage,
                             stretch=stretch, Q=Q, minimum=minimum,
                             filename=filename)
        return(rgb)
