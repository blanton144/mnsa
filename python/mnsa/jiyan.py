import os
import glob
import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import scipy.interpolate
import fitsio


def ratios_to_pspace(n2ha=None, s2ha=None, o3hb=None):
    """Converts log line ratios to Ji & Yan P1/P2/P3 space

    Parameters
    ----------

    n2ha : np.float32 or ndarray of np.float32
        log [NII] 6583 / H-alpha flux ratio

    s2ha : np.float32 or ndarray of np.float32
        log ([SII] 6716 + 6730) / H-alpha flux ratio

    o2hb : np.float32 or ndarray of np.float32
        log ([OIII] 5008) / H-beta flux ratio

    Returns
    -------

    p1 : np.float32 or ndarray of np.float32
        P1 component (SF or AGN-ness)

    p2 : np.float32 or ndarray of np.float32
        P2 component (metallicity-ish)

    p3 : np.float32 or ndarray of np.float32
        P3 component (ionization-ish?)
"""
    p1 = 0.63 * n2ha + 0.51 * s2ha + 0.59 * o3hb
    p2 = - 0.63 * n2ha + 0.78 * s2ha
    p3 = - 0.46 * n2ha - 0.37 * s2ha + 0.81 * o3hb
    return(p1, p2, p3)


class JiYan(object):
    """Class for Ji & Yan models

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

"""
    def __init__(self):
        self.name = None

        self.lines = dict()
        self.lines['o3'] = ['O__3__5006']
        self.lines['hb'] = ['H__1__4861']
        self.lines['n2'] = ['N__2__6583']
        self.lines['ha'] = ['H__1__6562']
        self.lines['s2'] = ['S__2__6716', 'S__2__6730']
        self.lines['o2'] = ['BLND__3727']
        self.lines['o1'] = ['O__1__6300']

        self.raw_grids = None
        self.raw_metallicity_o = None
        self.raw_log_ionization = None

        self.grids = None
        self.metallicity_o = None
        self.log_ionization = None
        return

    def set_pspace(self):
        self.n2ha = np.log10(self.grids['n2'] / self.grids['ha'])
        self.o3hb = np.log10(self.grids['o3'] / self.grids['hb'])
        self.s2ha = np.log10(self.grids['s2'] / self.grids['ha'])

        self.p1, self.p2, self.p3 = ratios_to_pspace(n2ha=self.n2ha, s2ha=self.s2ha, o3hb=self.o3hb)
        return

    def read_model(self, modelname=None):
        """Read in a particular model"""
        # Load photoionization model grids
        self.name = modelname
        modeldir = os.path.join(os.getenv('MNSA_DIR'), 'data', 'jiyan')
        nameList = []
        n_model = len(glob.glob(os.path.join(modeldir, modelname) + '*'))
        for i in np.arange(n_model, dtype=np.int32):
            nameList += [os.path.join(modeldir, modelname) + str(i) + '_line.fits']

        # Hack for metallicity grid
        if('bpl' in modelname):
            self.raw_metallicity_o = np.array([-0.75, -0.5, -0.25, 0.,
                                               0.25, 0.5, 0.75],
                                              dtype=np.float32)
        else:
            self.raw_metallicity_o = np.array([-1.3, -0.7, -0.4, 0., 0.3, 0.5],
                                              dtype=np.float32)

        for iname, name in enumerate(nameList):
            hdu = astropy.io.fits.open(name)

            # extract emission line fluxes
            tmp = dict()
            for line in self.lines:
                tmp[line] = 0.
                for d in self.lines[line]:
                    tmp[line] = tmp[line] + hdu[1].data[d]

            if(self.raw_grids is None):
                self.raw_grids = dict()
                for line in self.lines:
                    nper = len(tmp[line])
                    self.raw_grids[line] = np.zeros((n_model, nper),
                                                    dtype=np.float32)

            for line in self.lines:
                self.raw_grids[line][iname, :] = tmp[line]

            self.raw_log_ionization = hdu[1].data['IONIZATION']

        self.grids = dict()
        refine = 5
        for line in self.lines:
            n0_raw = self.raw_grids[line].shape[0]
            n1_raw = self.raw_grids[line].shape[1]
            x0_raw = np.arange(n0_raw, dtype=np.float32)
            x1_raw = np.arange(n1_raw, dtype=np.float32)
            ginterp = scipy.interpolate.RegularGridInterpolator((x0_raw, x1_raw),
                                                                np.log10(self.raw_grids[line]),
                                                                method='cubic')
            ointerp = scipy.interpolate.CubicSpline(x0_raw,
                                                    self.raw_metallicity_o)
            iinterp = scipy.interpolate.CubicSpline(x1_raw,
                                                    self.raw_log_ionization)

            n0 = (n0_raw - 1) * refine + 1
            x0 = np.arange(n0, dtype=np.float32)
            x0 = x0 / np.float32(n0 - 1) * (n0_raw - 1)
            n1 = (n1_raw - 1) * refine + 1
            x1 = np.arange(n1, dtype=np.float32)
            x1 = x1 / np.float32(n1 - 1) * (n1_raw - 1)
            x1g, x0g = np.meshgrid(x1, x0)
            self.grids[line] = 10.**(ginterp((x0g, x1g)))
            self.metallicity_o = ointerp(x0)
            self.log_ionization = iinterp(x1)

        return

    def plot_mesh(self, x=None, y=None, mask=None, **kwargs):
        """Plot a 2D mesh

        Parameters
        ----------

        x : 2D ndarray of np.float32
           X-axis values

        y : 2D ndarray of np.float32
           Y-axis values

        mask : 2D ndarray of bool
           plot only these points
"""
        n0 = x.shape[0]
        n1 = x.shape[1]
        if(mask is not None):
            outx = np.ma.masked_where(mask == False, x)
            outy = np.ma.masked_where(mask == False, y)
        else:
            outx = x
            outy = y

        for i0 in np.arange(n0, dtype=np.int32):
            plt.plot(outx[i0, :], outy[i0, :], **kwargs)
        for i1 in np.arange(n1, dtype=np.int32):
            plt.plot(outx[:, i1], outy[:, i1], **kwargs)

        return
