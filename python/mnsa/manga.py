import os
import sys
import numpy as np
import mnsa.reconstruct as reconstruct
import mnsa.utils.configuration as configuration


def manga_native(plateifu=None,
                 version=None):

    mnsa_config = configuration.MNSAConfig(version=version)
    cfg = mnsa_config.cfg

    crr_lambda = np.float32(cfg['MANGA']['crr_lambda'])
    release = cfg['MANGA']['release']
    pixscale = np.float32(cfg['General']['pixscale'])
    plate, ifu = [int(x) for x in plateifu.split('-')]

    SpectrumG = reconstruct.set_G(plate=plate,
                                  ifu=ifu,
                                  release=release,
                                  waveindex=None,
                                  lam=crr_lambda,
                                  dimage=pixscale,
                                  psf_slices=True)

    plateifu_dir = os.path.join(os.getenv('MNSA_DATA'),
                                version, str(plate), str(ifu))
    native_dir = os.path.join(plateifu_dir, 'native')

    manga_file = os.path.join(native_dir,
                              '{plateifu}-manga-logcube-crr')
    manga_file = manga_file.format(plateifu=plateifu)

    os.makedirs(native_dir, exist_ok=True)

    G_cube = reconstruct.write(SpectrumG, manga_file)

    return
