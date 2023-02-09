import os
import json
import numpy as np
import matplotlib.image


# Class to define a singleton
class ImageTypesSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(ImageTypesSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageTypes(object, metaclass=ImageTypesSingleton):
    """Image types (singleton)

    Attributes
    ----------

    types : dict
        information about image types

    Notes
    -----

    Initialized with image type information in 
      $MNSA_DIR/etc/image-types.json.

    Each image type has the following structure:

     'images' : a dictionary of images to draw from
     'R' : a dictionary defining contributions of images to R
     'G' : a dictionary defining contributions of images to G
     'B' : a dictionary defining contributions of images to B
     'Rscale' : float to multiply R
     'Gscale' : float to multiply G
     'Bscale' : float to multiply B
     'stretch' : linear stretch
     'Q' : nonlinearity
     'minimum' : value to set to black

    Each value in the 'images' dictionary is itself a dictionary:
     'file' : 'cube', 'maps', or 'resampled' specifying source of image
     'ext' : extension name to draw image from
     'channel' : channel to draw image from if file='maps' or 
                 band to project onto if file='cube' and ext='FLUX'
                 ('' otherwise)

    The 'R', 'G', and 'B' dictionaries have keys that must also
    be keys in the 'images' dictionary; the values are linear 
    scalings of those images; the R image will be the sum of the
    images weighted by the scalings.

    An example is below. This creates a greyscale image from the
    r-band results and overlays H-alpha in red and [OIII] 5007
    in green.

    {"Ha-O3-Grey":
     {"images": {"Ha": {"file": "maps", "ext": "EMLINE_GFLUX", "channel": "Ha-6564"},
                 "O3": {"file": "maps", "ext": "EMLINE_GFLUX", "channel": "OIII-5008"},
                 "r":  {"file": "cube", "ext": "RIMG", "channel": ""}},
      "R": {"r": 1.9, "Ha": 0.9},
      "G": {"r": 1.9, "O3": 1.5},
      "B": {"r": 1.9},
      "Rscale": 0.4,
      "Gscale": 0.4,
      "Bscale": 0.4,
      "stretch": 2.0,
      "Q": 5.0,
      "minimum": 0.0
     }
    }
"""
    def __init__(self):
        self.reset()
        return

    def reset(self):
        """Reset image types from $MNSA_DIR/etc/image-types.json"""
        fp = open(os.path.join(os.getenv('MNSA_DIR'),
                               'etc', 'image-types.json'))
        self.types = json.load(fp)
        fp.close()
        return


def rgb_stretch(rimage=None, gimage=None, bimage=None,
                minimum=0., stretch=1., Q=10., filename=None,
                invert=False):
    """Create stretched RGB image
   
    Parameters
    ----------

    rimage : 2-D ndarray of np.float32 or np.float64
        image for red channel

    gimage : 2-D ndarray of np.float32 or np.float64
        image for green channel

    bimage : 2-D ndarray of np.float32 or np.float64
        image for blue channel

    minimum : float
        zero level for overall stretch (see notes) (default 0)

    stretch : float
        linear stretch (see notes) (default 1)

    Q : float
        non-linearity level, or 0 for no nonlinearity (see notes) (default 10)

    invert : bool
        if True, invert the image (default False)

    filename : str
        if not None, write image to this file with matplotlib.image.imsave() (default None)

    Returns
    -------

    RGB : 3-D ndarray of np.uint8
       [N, M, 3] array of R, G, B channel settings

    Notes
    -----

    Procedure is the one from Lupton et al.:

     intensity = (rimage + gimage + bimage) / 3
     sintensity = arcsinh(stretch * Q * (intensity - minimum)) / Q 

    This makes "sintensity" the stretched value of intensity. At values
    of intensity less than Q, it is close to just stretch*intensity.
    
    If Q is set to zero then we just use the linear version:
     sintensity = stretch * (intensity - minimum)

    Then for each band, wherever sintensity is greater than zero:
      R = sintensity * rimage / intensity
      ...etc...
    This just uses the ratio sintensity / intensity on all the bands,
    which basically just applies the offset ("minimum") and nonlinearity.

    At this point the ratios of RGB are all the same as in the original
    image, but the image values have been stretched. We are going to 
    map the values 0..1 in these bands to 0..255 for uint8 images.

    For each pixel, we ask what the maximum of the three channels RGB is,
    and if it is greater than 1, we map the maximum channel to 1, and 
    scale the other channels by the same factor. This is color-preserving.

    Then the 0..1 values are mapped to 0..255 integers linearly. 
"""

    uint8max = np.float32(np.iinfo(np.uint8).max)

    intensity = (rimage + gimage + bimage) / 3.
    intensity = intensity + 1.e-9 * intensity.max()
    if(Q != 0):
        sintensity = np.arcsinh(stretch * Q * (intensity - minimum)) / Q
    else:
        sintensity = stretch * (intensity - minimum)
    R = np.where(sintensity > 0, sintensity * rimage / intensity, 0.)
    G = np.where(sintensity > 0, sintensity * gimage / intensity, 0.)
    B = np.where(sintensity > 0, sintensity * bimage / intensity, 0.)

    R = np.where(R > 0, R, 0.)
    G = np.where(G > 0, G, 0.)
    B = np.where(B > 0, B, 0.)

    RGB = np.stack([R, G, B], axis=2)
    maxRGB = RGB.max(axis=2)
    scale = np.where(maxRGB > 1., 1. / (maxRGB + 1.e-16), 1.)

    RGB[:, :, 0] *= scale * uint8max
    RGB[:, :, 1] *= scale * uint8max
    RGB[:, :, 2] *= scale * uint8max

    if(invert):
        RGB = uint8max - RGB

    RGB = np.uint8(RGB)

    if(filename is not None):
        matplotlib.image.imsave(filename, RGB, origin='lower')

    return(RGB)
