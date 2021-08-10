import os
import numpy as np
import mnsa.reconstruct as reconstruct
import mnsa.utils.configuration as configuration


def crr(plateifu=None, version=None):

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

    os.makedirs(manga_dir, exist_ok=True)

    cube = reconstruct.ReconstructCRR(plate=plate,
                                      ifu=ifu,
                                      release=release,
                                      lam=crr_lambda,
                                      pixelscale=pixscale)
    cube.run()
    cube.write(filename=manga_file)

    return
