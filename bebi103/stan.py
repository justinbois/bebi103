"""Deprecated alias for :mod:`bebi103.hmc`.

The ``bebi103.stan`` submodule has been renamed to ``bebi103.hmc``. Importing
``bebi103.stan`` still works but is deprecated and will be removed in a future
release. Please update your imports to use ``bebi103.hmc``.
"""

import warnings

from . import hmc

warnings.warn(
    "`bebi103.stan` has been renamed to `bebi103.hmc`. `bebi103.stan` is "
    "deprecated and will be removed in a future release; please update your "
    "imports to use `bebi103.hmc`.",
    FutureWarning,
    stacklevel=2,
)


def __getattr__(name):
    # Delegate every attribute access (including private helpers) to the
    # renamed module so `bebi103.stan.<name>` keeps working.
    return getattr(hmc, name)


def __dir__():
    return dir(hmc)
