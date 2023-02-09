import os
import numpy as np
import mnsa.mnsa as mnsa
import mnsa.reconstruct as reconstruct
import mnsa.bandscale as bandscale
import mnsa.utils.configuration as configuration
import astropy.visualization as viz
import marvin.tools.cube
import fitsio


def image(plateifu=None, version=None, clobber=True):
    """Write images for a plate-ifu

    Parameters
    ----------

    plateifu : str
        plate-ifu designation

    version : str
        version of data

    clobber : bool
        if True, clobber existing file
"""

    m = mnsa.MNSA(version=version, plateifu=plateifu)
    m.read_cube()

    gscale = 2.2
    rscale = 1.3
    iscale = 1.0
    stretch = 4.8
    Q = 2.5
    minimum = - 0.5

    wave = m.cube['WAVE'].read()
    flux = m.cube['FLUX'].read()
    gimage = bandscale.image(band='g', wave=wave, cube=flux) * gscale
    rimage = bandscale.image(band='r', wave=wave, cube=flux) * rscale
    iimage = bandscale.image(band='i', wave=wave, cube=flux) * iscale

    viz.make_lupton_rgb(iimage, rimage, gimage, minimum=minimum,
                        stretch=stretch, Q=Q,
                        filename=m.png_base + '-LOGCUBE-irg.png')

    c = marvin.tools.cube.Cube(plateifu=plateifu)
    wave = fitsio.read(c.filename, ext='WAVE')
    flux = fitsio.read(c.filename, ext='FLUX')
    ogimage = bandscale.image(band='g', wave=wave, cube=flux) * gscale
    orimage = bandscale.image(band='r', wave=wave, cube=flux) * rscale
    oiimage = bandscale.image(band='i', wave=wave, cube=flux) * iscale
    ogimage = ogimage * 2.25
    orimage = orimage * 2.25
    oiimage = oiimage * 2.25

    viz.make_lupton_rgb(oiimage, orimage, ogimage, minimum=minimum,
                        stretch=stretch, Q=Q,
                        filename=m.png_base + '-LOGCUBE-dr17-irg.png')

    return


def crr(plateifu=None, version=None, clobber=True):
    """Perform CRR for a given plate-ifu

    Parameters
    ----------

    plateifu : str
        plate-ifu designation

    version : str
        version of data

    clobber : bool
        if True, clobber existing file
"""

    mnsa_config = configuration.MNSAConfig(version=version)
    cfg = mnsa_config.cfg

    crr_lambda = np.float32(cfg['MANGA']['crr_lambda'])
    release = cfg['MANGA']['release']
    pixscale = np.float32(cfg['General']['pixscale'])
    plate, ifu = [int(x) for x in plateifu.split('-')]

    manga_dir = os.path.join(os.getenv('MNSA_DATA'),
                             version, 'manga', 'redux',
                             version, str(plate), 'stack')

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
