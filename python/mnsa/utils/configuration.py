# @Author: Michael R. Blanton
# @Date: Aug 21, 2020
# @Filename: params.py
# @License: BSD 3-Clause
# @Copyright: Michael R. Blanton

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import configparser


# Class to define a singleton
class MNSAConfigSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MNSAConfigSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MNSAConfig(object, metaclass=MNSAConfigSingleton):
    def __init__(self, version='v0'):
        self.reset(version=version)

    def reset(self, version=None):
        cfgfile = os.path.join(os.getenv('MNSA_DIR'), 'etc',
                               'mnsa-{version}.cfg'.format(version=version))
        self.cfg = configparser.ConfigParser(allow_no_value=True)
        self.cfg.optionxform = str
# Put defaults here
#        self.cfg.read_dict({'Assignment': {'fgot_minimum': 0.5,
#                                           'fgot_maximum': 1.5}})
        self.cfg.read(cfgfile)
