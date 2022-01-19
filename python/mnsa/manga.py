import os
import numpy as np
import mnsa.reconstruct as reconstruct
import mnsa.utils.configuration as configuration
import astropy.visualization as viz
import marvin.tools.cube
import fitsio


def image(plateifu=None, version=None, clobber=True):

    mnsa_config = configuration.MNSAConfig(version=version)
    cfg = mnsa_config.cfg

    release = cfg['MANGA']['release']
    plate, ifu = [int(x) for x in plateifu.split('-')]

    manga_dir = os.path.join(os.getenv('MNSA_DATA'),
                             version, 'manga', 'redux',
                             str(plate), 'stack')

    manga_file = os.path.join(manga_dir,
                              'manga-{plateifu}-LOGCUBE.fits.gz')
    manga_file = manga_file.format(plateifu=plateifu)

    gimage = fitsio.read(manga_file, ext='GIMG') * 4.7
    rimage = fitsio.read(manga_file, ext='RIMG') * 5.
    iimage = fitsio.read(manga_file, ext='IIMG') * 6.

    viz.make_lupton_rgb(iimage, rimage, gimage, minimum=-0.1,
                        stretch=7.0, Q=1.0,
                        filename=manga_file.replace('.fits.gz',
                                                    '.irg.png'))

    c = marvin.tools.cube.Cube(plateifu=plateifu)
    ogimage = fitsio.read(c.filename, ext='GIMG') * 4.7 / 2.
    orimage = fitsio.read(c.filename, ext='RIMG') * 5. / 2.
    oiimage = fitsio.read(c.filename, ext='IIMG') * 6. / 2.

    viz.make_lupton_rgb(oiimage, orimage, ogimage, minimum=-0.1,
                        stretch=7.0, Q=1.0,
                        filename=manga_file.replace('.fits.gz',
                                                    '.orig.irg.png'))


    return


def crr(plateifu=None, version=None, clobber=True):

    mnsa_config = configuration.MNSAConfig(version=version)
    cfg = mnsa_config.cfg

    crr_lambda = np.float32(cfg['MANGA']['crr_lambda'])
    release = cfg['MANGA']['release']
    pixscale = np.float32(cfg['General']['pixscale'])
    plate, ifu = [int(x) for x in plateifu.split('-')]

    manga_dir = os.path.join(os.getenv('MNSA_DATA'),
                             version, 'manga', 'redux',
                             str(plate), 'stack')

    manga_file = os.path.join(manga_dir,
                              'manga-{plateifu}-LOGCUBE')
    manga_file = manga_file.format(plateifu=plateifu)

    print("{plateifu}: File to write: {manga_file}".format(plateifu=plateifu,
                                                           manga_file=manga_file),
          flush=True)

    if((os.path.exists(manga_file + '.fits.gz') is True)
       & (clobber is False)):
        print("{plateifu}: Result exists in {manga_file}".format(manga_file=manga_file, plateifu=plateifu), flush=True)
        return

    os.makedirs(manga_dir, exist_ok=True)

    print("{plateifu}: Reconstruction commencing".format(plateifu=plateifu), flush=True) 
    cube = reconstruct.ReconstructCRR(plate=plate,
                                      ifu=ifu,
                                      release=release,
                                      lam=crr_lambda,
                                      pixelscale=pixscale)
    cube.run()
    print("{plateifu}: Writing file: {manga_file}".format(plateifu=plateifu,
                                                          manga_file=manga_file),
          flush=True)
    cube.write(filename=manga_file)

    return
