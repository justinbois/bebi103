# -*- coding: utf-8 -*-

"""Top-level package for bebi103."""

# Force showing deprecation warnings.
import re
import warnings
warnings.filterwarnings('always', 
                        category=DeprecationWarning,
                        module='^{}\.'.format(re.escape(__name__)))

from . import viz

from . import image

try:
    from . import stan
except:
    warnings.warn('Could not import `stan` submodule. Perhaps pystan is not properly installed.')


__author__ = """Justin Bois"""
__email__ = 'bois@caltech.edu'
__version__ = '0.0.31'
