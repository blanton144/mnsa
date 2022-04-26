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

    Attributes:
    ----------

    types : dict
        information about image types

"""
    def __init__(self):
        self.reset()
        return

    def reset(self):
        """Read in image types"""
        fp = open(os.path.join(os.getenv('MNSA_DIR'),
                               'etc', 'image-types.json'))
        self.types = json.load(fp)
        fp.close()
        return
