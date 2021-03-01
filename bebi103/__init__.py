# -*- coding: utf-8 -*-

"""Top-level package for bebi103."""

# Force showing deprecation warnings.
import re
import warnings

warnings.filterwarnings(
    "always", category=DeprecationWarning, module="^{}\.".format(re.escape(__name__))
)

try:
    import multiprocess
except:
    warnings.warn(
        "Unable to import multiprocess. Using multiprocessing (note the"
        " ing) instead. Depending on your operating system, Python"
        " version, and whether or not you are running in Jupyter, "
        " IPython, the Python REPL, etc., your execution may stall if"
        " you try to run jobs on more than one core. See discussion"
        " here: https://github.com/ipython/ipython/issues/12396. As a"
        " workaround, you can install multiprocess"
        " (pip install multiprocess) and everything should work as"
        " expected.",
        ImportWarning,
    )

from . import hv

from . import viz

from . import image

from . import bootstrap

from . import gp

try:
    from . import stan
except:
    warnings.warn(
        "Could not import `stan` submodule. Perhaps ArviZ or PyStan or CmdStanPy is/are"
        " not properly installed."
    )


__author__ = """Justin Bois"""
__email__ = "bois@caltech.edu"
__version__ = "0.1.3"
