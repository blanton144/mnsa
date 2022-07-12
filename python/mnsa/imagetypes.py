import os
import json

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
     'file' : either 'cube' or 'maps', specifying source of image
     'ext' : extension name to draw image from
     'channel' : channel to draw image from if 'maps' ('' otherwise)

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
