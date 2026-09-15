# Where noted, some of the diagnostic code and wrapper functions are adopted
# or copied from similar functions in Michael Betancourt's mcmc_diagnostics
# package: https://github.com/betanalpha/mcmc_diagnostics
# That package is licensed under a BSD-4-Clause license, the text of which
# and copyright notice appear below.
#
# BSD 3-Clause License
# 
# Copyright (c) 2022, Michael Betancourt
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import inspect
import re
import datetime
import time
import random
import contextlib
import copy
import fnmatch
import glob
import logging
import os
import shutil
import sys
import tomllib
import textwrap
import urllib
import warnings

import multiprocessing

import tqdm

import numpy as np
import packaging.version
import pandas as pd
import polars as pl
import xarray

from . import utils

try:
    import arviz_base as az

    if packaging.version.parse(az.__version__) < packaging.version.parse("1.0.0"):
        raise RuntimeError("Must use ArviZ version 1.0.0 or above.")
    arviz_success = True
except:
    arviz_success = False
    warnings.warn(
        "Could not import ArviZ version 1.0.0 or above. Functionality "
        "requiring ArviZ will not be available."
    )


def _require_arviz():
    """Raise a RuntimeError if ArviZ 1.0.0 or above is not available."""
    if not arviz_success:
        raise RuntimeError(
            "This function requires ArviZ version 1.0.0 or above, which "
            "could not be imported."
        )

try:
    import cmdstanpy

    cmdstanpy_success = True
except:
    cmdstanpy_success = False

if not cmdstanpy_success:
    raise RuntimeError("CmdStanPy could not be imported.")


def StanModel(
    file=None,
    charset="utf-8",
    model_name="anon_model",
    model_code=None,
    force_compile=False,
    **kwargs,
):
    """ "Utility to load/save cached compiled Stan models using PyStan.
    DEPRECATED
    """
    raise RuntimeError("bebi103.hmc.StanModel is deprecated.")


def install_cmdstan_colab(cmdstan_colab_url=None, quiet=False):
    """Install CmdStan with Google Colab.

    Parameters
    ----------
    cmdstan_colab_url : str, default None
        Full URL for zip file containing version of Stan for
        installation. If None, searches for the most recent version. In
        general, this should be `None`.
    quiet : bool, default False
        If True, suppresses success message.

    Returns
    -------
    None
    """
    if "google.colab" not in sys.modules:
        raise RuntimeError("This is only meant for installation using Google Colab.")

    if cmdstan_colab_url is None:
        from cmdstanpy.install_cmdstan import latest_version

        cmdstan_version = latest_version()

        # First spelling of URL
        try:
            cmdstan_url = f"https://github.com/stan-dev/cmdstan/releases/download/v{cmdstan_version}/"
            fname = f"colab-cmdstan-{cmdstan_version}.tgz"
            urllib.request.urlretrieve(cmdstan_url + fname, fname)
            shutil.unpack_archive(fname)
            os.environ["CMDSTAN"] = f"./cmdstan-{cmdstan_version}"

            if not quiet:
                print(f"CmdStan version {cmdstan_version} successfully installed.")

            return None
        except:
            pass

        # Old misspelling
        try:
            cmdstan_url = f"https://github.com/stan-dev/cmdstan/releases/download/v{cmdstan_version}/"
            fname = f"collab-cmdstan-{cmdstan_version}.tgz"
            urllib.request.urlretrieve(cmdstan_url + fname, fname)
            shutil.unpack_archive(fname)
            os.environ["CMDSTAN"] = f"./cmdstan-{cmdstan_version}"

            if not quiet:
                print(f"CmdStan version {cmdstan_version} successfully installed.")

            return None
        except:
            pass

        # Try each with .tar.gz suffix
        try:
            cmdstan_url = f"https://github.com/stan-dev/cmdstan/releases/download/v{cmdstan_version}/"
            fname = f"colab-cmdstan-{cmdstan_version}.tar.gz"
            urllib.request.urlretrieve(cmdstan_url + fname, fname)
            shutil.unpack_archive(fname)
            os.environ["CMDSTAN"] = f"./cmdstan-{cmdstan_version}"

            if not quiet:
                print(f"CmdStan version {cmdstan_version} successfully installed.")

            return None
        except:
            pass

        try:
            cmdstan_url = f"https://github.com/stan-dev/cmdstan/releases/download/v{cmdstan_version}/"
            fname = f"collab-cmdstan-{cmdstan_version}.tar.gz"
            urllib.request.urlretrieve(cmdstan_url + fname, fname)
            shutil.unpack_archive(fname)
            os.environ["CMDSTAN"] = f"./cmdstan-{cmdstan_version}"

            if not quiet:
                print(f"CmdStan version {cmdstan_version} successfully installed.")

            return None
        except:
            pass

        # The only way we get here is if we never successfully completed a try
        raise RuntimeError(
            "Unable to install CmdStan, most likely because the URL for the most recent version of CmdStan could not be found. Try finding at URL for a downloadable Colab-compatible version of CmdStan here: https://github.com/stan-dev/cmdstan/releases. Then call `install_cmdstan_colab(cmdstan_colab_url)`."
        )
    else:
        try:
            urllib.request.urlretrieve(cmdstan_colab_url)
            shutil.unpack_archive(fname)
            os.environ["CMDSTAN"] = f"./cmdstan-{cmdstan_version}"

            if not quiet:
                print(
                    "CmdStan successfully installed. You should nonetheless test the installation."
                )

            return None
        except:
            raise RuntimeError(
                "Unable to install CmdStan, most likely because the inputted URL for CmdStan was invalid. You can run without providing a URL, and the latest version of CmdStan willbe installed; `install_cmdstan_colab()`."
            )


def include_path():
    """Return path to include files for Stan.

    Returns
    -------
    output : str
        Absolute path to directory containing include files.
    """
    return os.path.join(os.path.dirname(__file__), "stan_include")


def clean_cmdstan(path="./", prefix=None, delete_sampling_output=False):
    """Remove all .hpp, .o, .d, and executable files resulting from
    compilation of Stan models using CmdStanPy.

    Parameters
    ----------
    path : str, default './'
        Path to directory containing files to delete.
    prefix : str, default None
        String of prefix of model name. This is the name of the Stan
        file from which the model was generated is <prefix>.stan. If
        None, then all .stan files are used. Files <prefix>.hpp,
        <prefix>.o, <prefix>.d, and the executable file <prefix> are
        deleted.
    delete_sampling_output: bool, default False
        If True, also delete all output generated by CmdStan.

    Notes
    -----
    If your files are stored in a temporary directory, as for the
    CmdStanPy default, use `cmdstanpy.cleanup_tmpdir()` instead.
    """
    if path == "/":
        raise RuntimeError(
            "No way you can delete stuff from root. You are making a big mistake."
        )

    if prefix is None:
        prefix = ""

    stan_files = glob.glob(os.path.join(path, prefix, "*.stan"))
    prefixes = [fname[:-5] for fname in stan_files]

    hpp_files = [prefix + ".hpp" for prefix in prefixes]
    o_files = [prefix + ".o" for prefix in prefixes]
    d_files = [prefix + ".d" for prefix in prefixes]

    for fname in hpp_files + o_files + d_files:
        if os.path.isfile(fname):
            os.remove(fname)

    for fname in prefixes:
        if os.path.isfile(fname) and os.access(fname, os.X_OK):
            os.remove(fname)

    if delete_sampling_output:
        for pre in prefixes:
            out_files = glob.glob(pre + "*.csv")
            out_files += glob.glob(pre + "*.txt")
            for fname in out_files:
                if os.path.isfile(fname):
                    os.remove(fname)


def cmdstan_version():
    """Determine CmdStan version

    Returns
    -------
    output : str
        Version of installed CmdStan used by CmdStanPy. If unable to
        ascertain version, return a string:
        "Unable to determine CmdStan version."

    Notes
    -----
    Only works if CmdStanPy is installed.
    """
    try:
        cmdstan_path = cmdstanpy.cmdstan_path()
        return cmdstan_path[cmdstan_path.rfind("-") + 1 :]
    except:
        return "Unable to determine CmdStan version."


def df_to_datadict_hier(
    df=None, level_cols=None, data_cols=None, sort_cols=[], cowardly=False
):
    """Convert a tidy data frame to a data dictionary for a hierarchical
    Stan model.

    Parameters
    ----------
    df : DataFrame
        A tidy Polars or Pandas data frame.
    level_cols : list
        A list of column names containing variables that specify the
        level of the hierarchical model. These must be given in order
        of the hierarchy of levels, with the first entry being the
        farthest from the data.
    data_cols : list
        A list of column names containing the data.
    sort_cols : list, default []]
        List of columns names to use in sorting the data. They will be
        first sorted by the level indices, and the subsequently sorted
        according to sort_cols.
    cowardly : bool, default False
        If True, refuse to generate new columns if they already exist
        in the data frame. If you run this function using a data frame
        that was outputted previously by this function, you will get an
        error if `cowardly` is True. Otherwise, the columns may be
        overwritten.

    Returns
    -------
    data : dict
        A dictionary that can be passed to into a Stan program. The
        dictionary contains keys/entries:

            - `'N'`: Total number of data points
            - `'J_1'`: Number of hyper parameters for hierarchical level 1.
            - `'J_2'`: Number of hyper parameters for hierarchical level 2. ... and so on with `'J_3'`, `'J_4'`, ...
            - `'index_1'`: Set of `J_2` indices defining which level 1 parameters condition the level 2 parameters.
            - `'index_2'`: Set of `J_3` indices defining which level 2 parameters condition the level 3 parameters. ...and so on for 'index_3', etc.
            - `'index_k'`: Set of `N` indices defining which of the level k parameters condition the data, for a k-level hierarchical model.
            - `data_col[0]` : Data from first data_col
            - `data_col[1]` : Data from second data_col ...and so on.

    df : DataFrame
        Updated input data frame with added columnes with names given by
        `level_col[0] + '_stan'`, `level_col[1] + '_stan'`, etc. These
        contain the integer indices that correspond to the possibly
        non-integer values in the `level_col`s of the original data
        frame. This enables interpretation of Stan results, which have
        everything integer indexed.

    Notes
    -----
    Assumes no missing data.

    The ordering of data sets is not guaranteed. So, e.g., if you
    have time series data, you should use caution.

    Example
    -------
    >>> import io
    >>> import polars as pl
    >>> import bebi103
    >>> df = pd.read_csv(io.StringIO('''
        day,batch,colony,x
        monday,1,1,9.31
        monday,1,1,8.35
        monday,1,1,10.48
        monday,1,1,9.91
        monday,1,1,10.43
        monday,1,2,9.98
        monday,1,2,9.76
        monday,1,3,9.30
        monday,2,1,10.56
        monday,2,1,11.40
        monday,2,2,10.36
        monday,2,2,12.04
        monday,2,2,9.92
        monday,2,2,10.10
        monday,2,2,8.72
        monday,2,2,10.36
        monday,2,2,11.56
        monday,2,2,10.87
        monday,2,2,10.43
        monday,2,2,10.67
        monday,2,2,9.05
        monday,3,1,10.32
        monday,3,1,9.07
        monday,4,1,9.86
        monday,4,1,9.21
        monday,4,1,11.36
        monday,4,2,8.60
        monday,4,2,10.54
        monday,4,2,8.93
        monday,4,2,9.43
        monday,4,2,9.23
        monday,4,2,9.66
        monday,4,2,11.26
        monday,4,2,9.61
        monday,4,2,11.99
        monday,4,2,10.27
        monday,4,2,9.97
        monday,4,2,9.37
        monday,4,2,10.10
        monday,4,3,10.39
        monday,4,3,8.79
        wednesday,1,1,10.76
        wednesday,1,2,10.72
        wednesday,1,2,8.97
        wednesday,1,2,9.14
        wednesday,1,2,11.31
        wednesday,1,2,9.49
        wednesday,1,2,10.21
        wednesday,1,2,10.04
        wednesday,2,1,13.16
        wednesday,2,1,7.07
        wednesday,2,1,12.74
        wednesday,3,1,9.45
        wednesday,3,1,9.62
        wednesday,3,1,10.46
        wednesday,3,1,11.11
        wednesday,3,1,10.56
        wednesday,3,1,9.93
        thursday,1,1,8.60
        thursday,1,2,11.24
        thursday,1,2,9.10
        thursday,1,2,9.10
        thursday,1,2,11.30
        thursday,1,2,10.65
        thursday,1,2,9.98
        thursday,1,2,9.85
        thursday,1,2,12.41
        thursday,1,3,10.03
        thursday,1,3,10.53
        thursday,1,4,10.85
        '''), skipinitialspace=True)
    >>> data, df = bebi103.hmc.df_to_datadict_hier(df,
                                level_cols=['day', 'batch', 'colony'],
                                data_cols=['x'])
    >>> data
    {'N': 70,
     'J_1': 3,
     'J_2': 8,
     'J_3': 17,
     'index_1': array([1, 1, 1, 1, 2, 3, 3, 3]),
     'index_2': array([1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8]),
     'index_3': array([ 1,  1,  1,  1,  1,  2,  2,  3,  4,  4,  5,  5,  5,  5,  5,  5,  5,
             5,  5,  5,  5,  6,  6,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,
             8,  8,  8,  8,  8,  9,  9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12,
            12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17,
            17, 17]),
     'x': array([ 9.31,  8.35, 10.48,  9.91, 10.43,  9.98,  9.76,  9.3 , 10.56,
            11.4 , 10.36, 12.04,  9.92, 10.1 ,  8.72, 10.36, 11.56, 10.87,
            10.43, 10.67,  9.05, 10.32,  9.07,  9.86,  9.21, 11.36,  8.6 ,
            10.54,  8.93,  9.43,  9.23,  9.66, 11.26,  9.61, 11.99, 10.27,
             9.97,  9.37, 10.1 , 10.39,  8.79,  8.6 , 11.24,  9.1 ,  9.1 ,
            11.3 , 10.65,  9.98,  9.85, 12.41, 10.03, 10.53, 10.85, 10.76,
            10.72,  8.97,  9.14, 11.31,  9.49, 10.21, 10.04, 13.16,  7.07,
            12.74,  9.45,  9.62, 10.46, 11.11, 10.56,  9.93])}

    >>> df.head(10)
          day  batch  colony      x  day_stan  batch_stan  colony_stan
    0  monday      1       1   9.31         1           1            1
    1  monday      1       1   8.35         1           1            1
    2  monday      1       1  10.48         1           1            1
    3  monday      1       1   9.91         1           1            1
    4  monday      1       1  10.43         1           1            1
    5  monday      1       2   9.98         1           1            2
    6  monday      1       2   9.76         1           1            2
    7  monday      1       3   9.30         1           1            3
    8  monday      2       1  10.56         1           2            4
    9  monday      2       1  11.40         1           2            4
    """
    if df is None or level_cols is None or data_cols is None:
        raise RuntimeError("`df`, `level_cols`, and `data_cols` must all be specified.")

    if type(sort_cols) != list:
        raise RuntimeError("`sort_cols` must be a list.")

    # For now, if polars, just convert to pandas
    if isinstance(df, pl.DataFrame):
        polars_in = True
        df = df.to_pandas()
    else:
        polars_in = False

    # Get a copy so we don't overwrite
    new_df = df.copy(deep=True)

    if type(level_cols) not in [list, tuple]:
        level_cols = [level_cols]

    if type(data_cols) not in [list, tuple]:
        data_cols = [data_cols]

    level_cols_stan = [col + "_stan" for col in level_cols]

    if cowardly:
        for col in level_cols_stan:
            if col in df:
                raise RuntimeError(
                    "column "
                    + col
                    + " already in data frame. Cowardly deciding not to overwrite."
                )

    for col_ind, col in enumerate(level_cols):
        new_df[str(col) + "_stan"] = df.groupby(level_cols[: col_ind + 1]).ngroup() + 1

    new_df = new_df.sort_values(by=level_cols_stan + sort_cols)

    data = dict()
    data["N"] = len(new_df)
    for i, col in enumerate(level_cols_stan):
        data["J_" + str(i + 1)] = len(new_df[col].unique())
    for i, _ in enumerate(level_cols_stan[1:]):
        data["index_" + str(i + 1)] = np.array(
            [key[i] for key in new_df.groupby(level_cols_stan[: i + 2]).groups]
        ).astype(int)
    data["index_" + str(len(level_cols_stan))] = new_df[
        level_cols_stan[-1]
    ].values.astype(int)
    for col in data_cols:
        # Check string naming
        new_col = str(col)
        try:
            bytes(new_col, "ascii")
        except UnicodeEncodeError:
            raise RuntimeError("Column names must be ASCII.")
        if new_col[0].isdigit():
            raise RuntimeError("Column name cannot start with a number.")
        for char in new_col:
            if char in "`~!@#$%^&*()- =+[]{}\\|:;\"',<.>/?":
                raise RuntimeError("Invalid column name for Stan variable.")

        data[new_col] = new_df[col].values

    # Return polars if input was polars
    if polars_in:
        new_df = pl.from_pandas(new_df)

    return data, new_df


def convert_samples(
        samples, 
        out_format, 
        inc_warmup=False,
        arviz_kwargs=None
    ):
    """Converts samples to a new format.

    Parameters
    ----------
    samples : dict, cmdstanpy.CmdStanMCMC, xarray.DataTree, pandas.DataFrame, or polars.DataFrame
        Results from an MCMC sampling to convert to desired format.
    out_format : str, one of 'dict', 'arviz', 'pandas', 'polars'
        The desired output format of the samples.
        - dict: Dictionary of expectands and HMC diagnostics. Each item
          is a 2D Numpy array. The first dimension indexes the Markov 
          chains and the second indexes the sequential states within 
          each chain.
        - arviz: An xarray.DataTree instance as used by ArviZ. Only
          valid if `samples` is a cmdstanpy.CmdStanMCMC instance.
        - pandas.DataFrame: Each row of the data frame is a chain/draw
          with expectand values and HMC diagnostic results.
        - polars.DataFrame: Each row of the data frame is a chain/draw
          with expectand values and HMC diagnostic results.
    inc_warmup : bool, default False
        If True, include warmup samples. Only active if `samples` is a
        cmdstanpy.CmdStanMCMC instance.
    arviz_kwargs : dict, default None
        Keyword arguments to be passed to arviz_base.from_cmdstanpy().

    Returns
    -------
    output : various formats
        An single instance of the desired `out_format` is
        returned.
    """
    if out_format not in ('dict', 'arviz', 'pandas', 'polars'):
        raise RuntimeError("`out_format` must be one of 'dict', 'arviz', 'pandas', or 'polars'.")
    
    # Only allow including warmup if from cmdstanpy.CmdStanMCMC instance
    if inc_warmup and not isinstance(samples, cmdstanpy.CmdStanMCMC):
        raise ValueError("`inc_warmup` may only be True if converting a cmdstanpy.CmdStanMCMC instance.")

    if arviz_kwargs is None:
        arviz_kwargs = {}

    # Get input type of samples and validate, if necessary
    if isinstance(samples, dict):
        _validate_dict(samples)
        samples_format = 'dict'
    elif isinstance(samples, cmdstanpy.CmdStanMCMC):
        samples_format = 'cmdstanpy'
    elif isinstance(samples, pl.DataFrame):
        samples_format = 'polars'
    elif isinstance(samples, pd.DataFrame):
        samples_format = 'pandas'
    elif isinstance(samples, xarray.DataTree):
        samples_format = 'arviz'
    else:
        raise ValueError(f"`samples` input is of type {type(samples)}, which is not allowed.")

    # Perform conversions
    if out_format == 'arviz':
        _require_arviz()
        if samples_format == 'cmdstanpy':
            if (
                'save_warmup' in arviz_kwargs 
                and arviz_kwargs['save_warmup'] != inc_warmup
            ):
                raise RuntimeError("Conflicting inputs for arviz_kwargs['save_warmup'] and `inc_warmup`.")
            arviz_kwargs['save_warmup'] = inc_warmup
            result = az.from_cmdstanpy(samples, **arviz_kwargs)
        elif samples_format == 'arviz':
            result = samples
        else:
            raise ValueError("If `out_format` is 'arviz', `samples` must be a cmdstanpy.CmdStanMCMC instance.")

    if out_format == 'dict':
        if samples_format == 'dict':
            result = samples
        elif samples_format == 'cmdstanpy':
            result = _dict_from_cmdstanpy(samples, inc_warmup=inc_warmup)
        elif samples_format == 'arviz':
            result = _dict_from_arviz(samples, inc_warmup=inc_warmup)
        elif samples_format in ['polars', 'pandas']:
            result = _dataframe_to_dict(samples)

    if out_format in ['polars', 'pandas']:
        if samples_format == 'dict':
            result = _dict_to_dataframe(samples, dataframe_package=out_format)
        elif samples_format == 'cmdstanpy':
            result = _dict_to_dataframe(
                _dict_from_cmdstanpy(samples, inc_warmup=inc_warmup), dataframe_package=out_format
            )
        elif samples_format == 'arviz':
            result = _dict_to_dataframe(
                _dict_from_arviz(samples, inc_warmup=inc_warmup), dataframe_package=out_format
            )
        elif samples_format == 'polars':
            result = samples.to_pandas() if out_format == 'pandas' else samples
        elif samples_format == 'pandas':
            result = pl.from_pandas(samples) if out_format == 'polars' else samples
    
    return result


def _dict_to_dataframe(samples_dict, dataframe_package='polars'):
    """Convert dictionary of MCMC results to dataframe.

    Parameters
    ----------
    samples : dict
        A dictionary of two-dimensional float64 arrays for each
        expectand and HMC diagnostic. The first dimension of each 
        array indexes the Markov chains and the second indexes the
        sequential states within each chain.
    dataframe_package : str, default 'polars'
        Which data frame package to use for the output. Either 'polars'
        or 'pandas'.

    Returns
    -------
    output : polars.DataFrame or pandas.DataFrame
        Tidy data frame with one row per draw. It has columns `chain__`
        (one-indexed Markov chain) and `draw__` (one-indexed state within
        the chain), followed by one column per expectand and one column
        per HMC diagnostic. As with the diagnostics, the metadata columns
        `chain__` and `draw__` end in a double underscore to signify that
        they are not expectands.
    """
    _validate_dict(samples_dict)

    n_chains, n_draws = next(iter(samples_dict.values())).shape

    chain = np.array([[i]*n_draws for i in range(1, n_chains + 1)]).flatten()
    draw = np.array([np.arange(1, n_draws + 1) for _ in range(n_chains)]).flatten()
    chain_draw_dict = {"chain__": chain, "draw__": draw}

    df_dict = {key: value.flatten() for key, value in samples_dict.items()}
    df_dict = {**chain_draw_dict, **df_dict}

    if dataframe_package == 'polars':
        return pl.DataFrame(df_dict)
    elif dataframe_package == 'pandas':
        return pd.DataFrame(df_dict)
    else:
        raise ValueError('`dataframe_package` must be either "polars" or "pandas".')


def _dataframe_to_dict(df):
    """Convert a data frame to expectand and HMC diagnostic dictionaries.

    This is the inverse of `_dict_to_dataframe()`.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Tidy data frame, as returned by `_dict_to_dataframe()`. It must
        have one-indexed `chain__` and `draw__` columns identifying each
        draw, followed by one column per expectand (names not ending in
        '__') and one column per HMC diagnostic (names ending in '__').

    Returns
    -------
    output : dict
    """
    if not isinstance(df, (pl.DataFrame, pd.DataFrame)):
        raise ValueError("`df` must be a polars.DataFrame or pandas.DataFrame.")

    columns = list(df.columns)
    if "chain__" not in columns or "draw__" not in columns:
        raise ValueError("`df` must contain 'chain__' and 'draw__' columns.")

    if isinstance(df, pl.DataFrame):
        df = df.sort(["chain__", "draw__"])
    else:
        df = df.sort_values(["chain__", "draw__"])

    # Each column is stored draw-fastest within chain (C-order), so the
    # grid of (chain, draw) reshapes back to (n_chains, n_draws).
    chain = np.asarray(df["chain__"].to_numpy())
    draw = np.asarray(df["draw__"].to_numpy())
    n_chains = len(np.unique(chain))
    n_draws = len(np.unique(draw))

    if n_chains * n_draws != len(chain):
        raise ValueError(
            "`df` is not a complete chain-by-draw grid; cannot reshape."
        )

    samples_dict = {}
    for col in columns:
        if col in ("chain__", "draw__"):
            continue
        samples_dict[col] = (
            df[col]
            .to_numpy()
            .astype(np.float64)
            .reshape(n_chains, n_draws)
        )

    _validate_dict(samples_dict)

    return samples_dict


def _dict_from_cmdstanpy(samples, inc_warmup=False):
    """Extract samples of all expectands and HMC diagnostics (if any) 
    from cmdstanpy.CmdStanMCMC instance.

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance
        Results from MCMC sampling from which to extract expectand
        values.
    inc_warmup : bool, default False
        If True, include warmup draws in the output.

    Returns
    -------
    output : dict
        A dictionary of two-dimensional arrays for each expectand and
        HMC diagnostic in the CmdStanMCMC object. The first dimension 
        of each element indexes the Markov chains and the second 
        dimension indexes the sequential states within each Markov 
        chain.

    Notes
    -----
    .. Adopted from the extract_expectand_vals() function written by 
       Michael Betancourt in his mcmc_diagnostics package.
    """
    # Extract MCMC samples from the cmdstanpy.CmdStanMCMC instance
    # The draws() method returns a Numpy array with shape 
    # (num_draws, num_chains, num_columns)
    draws = samples.draws(inc_warmup=inc_warmup)
    names = samples.column_names

    # Extract all entries
    samples_and_diagnostics = {
        name: draws[:, :, k].T
        for k, name in enumerate(names)
    }

    return samples_and_diagnostics


def _dict_from_arviz(samples, inc_warmup=False):
    """Extract samples of all expectands and HMC diagnostics from an
    ArviZ DataTree into a single dictionary.

    Parameters
    ----------
    samples : xarray.DataTree instance
        Results from MCMC sampling, as would be generated by
        `arviz.from_cmdstanpy()`, from which to extract expectand and
        diagnostic values.
    inc_warmup : bool, default False
        If True, include warmup draws in the output. They are only
        available for groups that have a corresponding `warmup_*` group
        (i.e., the samples were created with `save_warmup=True`).

    Returns
    -------
    output : dict
        A single dictionary of two-dimensional float64 arrays. Expectands
        are collected from the `posterior`, `prior`, `posterior_predictive`,
        `prior_predictive`, and `log_likelihood` groups (whichever are
        present), and HMC diagnostics are collected from the
        `sample_stats` group, keyed by the CmdStanPy diagnostic names that
        end in '__' (e.g., 'divergent__', 'lp__'). The first dimension of
        each element indexes the Markov chains and the second indexes the
        sequential states within each Markov chain.

    Notes
    -----
    .. Multidimensional expectands have a separate entry in the output
       dictionary for each element, with one-indexed names matching those
       used by CmdStanPy. E.g., a two-dimensional variable `theta` has
       names like `theta[1,1]`, `theta[1,2]`, etc.
    .. Diagnostics are distinguished from expectands by keys ending in
       '__'.
    """
    # Groups holding expectands. Their variable names are used as-is.
    expectand_groups = [
        "posterior",
        "prior",
        "posterior_predictive",
        "prior_predictive",
        "log_likelihood",
    ]

    # Map ArviZ `sample_stats` names to the CmdStanPy diagnostic names,
    # which end in '__'. Any other stat is suffixed with '__' so that it
    # is identifiable as a diagnostic.
    diagnostic_name_map = {
        "lp": "lp__",
        "acceptance_rate": "accept_stat__",
        "step_size": "stepsize__",
        "tree_depth": "treedepth__",
        "n_steps": "n_leapfrog__",
        "diverging": "divergent__",
        "energy": "energy__",
    }

    result = {}
    warmup_found = False

    def _group_dataset(group):
        """Return a group's dataset, prepending warmup draws if requested."""
        nonlocal warmup_found
        ds = samples[group].dataset
        if inc_warmup:
            warmup_group = f"warmup_{group}"
            if warmup_group in samples.children:
                ds = xarray.concat(
                    [samples[warmup_group].dataset, ds], dim="draw"
                )
                warmup_found = True
        return ds

    def _assign(key, value):
        """Add an entry to `result`, erroring on a key collision."""
        if key in result:
            raise ValueError(
                f"Duplicate key '{key}' encountered while merging ArviZ "
                "groups into a single dictionary."
            )
        result[key] = value

    def _add_dataset(ds, rename):
        """Flatten each variable in `ds` into `result` under `rename(name)`."""
        for name, da in ds.data_vars.items():
            out_name = rename(name)
            # Dimensions beyond chain and draw index the elements of a
            # multidimensional expectand.
            elem_dims = [dim for dim in da.dims if dim not in ("chain", "draw")]

            if not elem_dims:
                _assign(
                    out_name,
                    da.transpose("chain", "draw").values.astype(np.float64),
                )
            else:
                shape = tuple(da.sizes[dim] for dim in elem_dims)
                for idx in np.ndindex(shape):
                    # One-index the bracketed name, as CmdStanPy does.
                    bracket = ",".join(str(i + 1) for i in idx)
                    sel = dict(zip(elem_dims, idx))
                    _assign(
                        f"{out_name}[{bracket}]",
                        da.isel(sel)
                        .transpose("chain", "draw")
                        .values.astype(np.float64),
                    )

    # Collect expectands from each group that is present.
    for group in expectand_groups:
        if group in samples.children:
            _add_dataset(_group_dataset(group), lambda name: name)

    # Collect HMC diagnostics from the sample_stats group, if present.
    if "sample_stats" in samples.children:
        _add_dataset(
            _group_dataset("sample_stats"),
            lambda name: diagnostic_name_map.get(
                name, name if name.endswith("__") else name + "__"
            ),
        )

    if inc_warmup and not warmup_found:
        warnings.warn(
            "`inc_warmup` is True, but no warmup groups were found; "
            "returning post-warmup draws only."
        )

    return result


def extract_variable(samples, variable, inc_warmup=False):
    """Extracts sample of a variable as a Numpy array.

    Parameters
    ----------
    samples : dict or cmdstanpy.CmdStanMCMC instance
        MCMC samples as either a dictionary where each entry is a 2D
        Numpy array or a cmdstanpy.CmdStanMCMC instance
    variable : string
        The name of the variable to extract
    inc_warmup : bool, default False
        If True, include warmup when extracting from a 
        cmdstanpy.CmdStanMCMC instance. Ignored if `samples` is a dict.

    Returns
    -------
    output : Numpy array of at least two dimensions
        The first dimension of the output is the chain. The second
        is the draw. If the variable is scalar, the result is two-
        dimensional. If the variable is vector-valued, the third
        dimension is the index of the dimension of the vector. If the
        variable is matrix-valued, the third and fourth dimensions
        are the matrix indices. And so on.
    """
    # Convert to a dictionary
    if isinstance(samples, cmdstanpy.CmdStanMCMC):
        samples = _dict_from_cmdstanpy(samples, inc_warmup=inc_warmup)

    # Find all entries belonging to the requested variable.
    keys = [key for key in samples if _get_var_name(key) == variable]
    if not keys:
        raise ValueError(f"Variable '{variable}' not found in samples.")

    # A scalar variable has a single entry whose key is the variable name.
    if keys == [variable]:
        return samples[variable]

    # Otherwise the variable is multidimensional. Parse the one-indexed
    # bracketed indices to determine the shape of each element.
    index_tuples = {
        key: tuple(int(i) for i in key[key.rfind("[") + 1 : -1].split(","))
        for key in keys
    }

    ndim = len(next(iter(index_tuples.values())))
    if any(len(inds) != ndim for inds in index_tuples.values()):
        raise ValueError(
            f"Entries of variable '{variable}' have inconsistent numbers of "
            "indices."
        )

    elem_shape = tuple(
        max(inds[d] for inds in index_tuples.values()) for d in range(ndim)
    )

    # Every element of the multidimensional variable must be present. Since
    # the keys are unique and within [1, max] along each axis, a matching
    # count guarantees a complete, rectangular grid.
    n_expected = int(np.prod(elem_shape))
    if len(index_tuples) != n_expected:
        raise ValueError(
            f"Variable '{variable}' is incomplete: found "
            f"{len(index_tuples)} entries, but expected {n_expected} for "
            f"shape {elem_shape}."
        )

    # The output has shape (n_chains, n_draws, *element dimensions).
    output = np.empty(samples[keys[0]].shape + elem_shape)
    for key, inds in index_tuples.items():
        # Convert the one-indexed bracket indices to zero-indexed positions.
        output[(slice(None), slice(None)) + tuple(i - 1 for i in inds)] = (
            samples[key]
        )

    return output


def _split_dict(samples):
    """Split a dictionary into expectand and diagnostic dictionaries.

    Parameters
    ----------
    samples : dict
        Dictionary with string keys to split. Entries whose keys end in
        '__' are taken to be HMC diagnostics; all others are taken to be
        expectands.

    Returns
    -------
    expectands : dict
        Entries of `samples` whose keys do not end in '__'.
    hmc_diagnostics : dict
        Entries of `samples` whose keys end in '__'.
    """
    expectands = {}
    hmc_diagnostics = {}
    for key, value in samples.items():
        if not isinstance(key, str):
            raise ValueError(f"Key {key} is not a string.")
        if key.endswith("__"):
            hmc_diagnostics[key] = value
        else:
            expectands[key] = value

    return expectands, hmc_diagnostics


def _validate_dict(result_dict, include_all_required_hmc_diagnostics=False):
    """Validate a dictionary of expectand and diagnostic samples.

    Parameters
    ----------
    result_dict : dict
        Dictionary whose values are per-expectand and per-diagnostic
        arrays. Every value must be a two-dimensional float64 NumPy 
        array, and all values must have the same shape. The first 
        dimension indexes the Markov chains and the second indexes
        the sequential states within each chain.
    include_all_required_hmc_diagnostics : bool, default False
        If True, requires that the dictionary include keys 
        "divergent__", "accept_stat__", "stepsize__", "treedepth__",
        "n_leapfrog__", and "energy__".

    Returns
    -------
    None
        Returns None if the dictionary is valid; otherwise raises a
        ValueError.
    """
    if not isinstance(result_dict, dict):
        raise ValueError("`result_dict` must be a dictionary.")

    shape = None
    for key, val in result_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"Key {key} is not a string.")
        if key in ("chain__", "draw__"):
            raise ValueError(f"Key '{key}' is not allowed; it is reserved.")
        if not isinstance(val, np.ndarray):
            raise ValueError(f"Entry '{key}' is not a NumPy array.")
        if val.ndim != 2:
            raise ValueError(
                f"Entry '{key}' is not a 2D array; it has {val.ndim} "
                "dimension(s)."
            )
        if val.dtype != np.float64:
            raise ValueError(
                f"Entry '{key}' has dtype {val.dtype}, not float64."
            )
        if shape is None:
            shape = val.shape
        elif val.shape != shape:
            raise ValueError(
                f"Entry '{key}' has shape {val.shape}, but shape {shape} was "
                "expected. All arrays must have the same number of rows and "
                "columns."
            )

    if include_all_required_hmc_diagnostics:
        _validate_dict_hmc_diagnostics(result_dict)


def _validate_dict_hmc_diagnostics(result_dict):
    """Ensure a dictionary contains all required HMC diagnostics.

    Parameters
    ----------
    result_dict : dict
        Dictionary of sampling results that should contain HMC
        diagnostics.

    Returns
    -------
    None
        Returns None if all required diagnostics are present; otherwise
        raises a ValueError.
    """
    required = (
        "divergent__",
        "accept_stat__",
        "stepsize__",
        "treedepth__",
        "n_leapfrog__",
        "energy__",
    )

    missing = [name for name in required if name not in result_dict]
    if missing:
        raise ValueError(
            "`result_dict` is missing required diagnostic(s): "
            + ", ".join(missing)
            + "."
        )


def _render_diagnostic_report(
    verbosity, no_warning, all_good_message, summary, detail, descriptions
):
    """Print a diagnostic report at the requested level of verbosity.

    Parameters
    ----------
    verbosity : int
        Level of verbosity. 0 prints nothing. 1 prints the summary
        messages. 2 prints the summary messages followed by explanatory
        descriptions of the diagnostics. 3 prints the per-chain detail
        messages followed by the explanatory descriptions.
    no_warning : bool
        True if no diagnostic triggered a warning.
    all_good_message : str
        Message printed (for verbosity of at least 1) when `no_warning`
        is True.
    summary : list of str
        Aggregate summary messages, printed for verbosity 1 and 2.
    detail : list of str
        Per-chain detail messages, printed for verbosity 3.
    descriptions : list of str
        Explanatory descriptions of the triggered diagnostics, printed
        for verbosity 2 and 3.
    """
    if verbosity < 1:
        return

    if no_warning:
        blocks = [all_good_message]
    elif verbosity == 1:
        blocks = summary
    elif verbosity == 2:
        blocks = summary + descriptions
    else:
        blocks = detail + descriptions

    print(
        '\n\n'.join(
            '\n'.join(textwrap.wrap(block, 72)) for block in blocks
        )
    )


def _failed_expectands_phrase(names, singular, plural):
    """Construct a phrase naming the expectands that triggered a
    diagnostic warning, with singular/plural agreement.

    Parameters
    ----------
    names : list of str
        Names of the expectands that triggered the warning.
    singular : str
        Noun phrase used when a single expectand is named, e.g.
        'a tail xi_hat warning'.
    plural : str
        Noun phrase used when more than one expectand is named, e.g.
        'tail xi_hat warnings'.

    Returns
    -------
    output : str
        A sentence naming the offending expectands.
    """
    if len(names) == 1:
        return f"The expectand {names[0]} triggered {singular}."
    return f"The expectands {', '.join(names)} triggered {plural}."


def check_hmc_diagnostics(
        samples, 
        adapt_target=0.801, 
        adapt_target_fraction=0.9, 
        efmi_rule_of_thumb=0.2,
        max_treedepth=10,
        verbosity=2,
    ):
    """Check to all HMC diagnostics that are included in Stan's output
    (divergences, EFMI, tree depth, acceptance rate).

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC, xarray.DataTree, pandas.DataFrame, or polars.DataFrame
        Results from MCMC sampling from which to check HMC diagnostics.
    adapt_target : float, default 0.801
        Target acceptance proxy statistic for step size adaptation. 
        This should be approximately the prescribed value of 
        `adapt_delta` in sampling. The default `adapt_delta` is 0.8,
        which is why the default `adapt_target` is 0.801.
    adapt_target_fraction : float, default 0.9
        Average target acceptance proxy statistic below 
        adapt_target_fraction * adapt_target is flagged as problematic.
    efmi_rule_of_thumb : float, default 0.2
        EFMI values greater than `efmi_rule_of_thumb` are flagged.
    max_treedepth: int, default 10
        Specification of maximum treedepth allowed in MCMC sampling.
    verbosity : int, default 2
        Level of verbosity for messages printed to the screen. 0 prints
        nothing. 1 prints a summary of the diagnostics. 2 prints the
        summary followed by descriptions of what the triggered
        diagnostics mean. 3 prints per-chain detail followed by the
        descriptions.

    Returns
    -------
    output : dict
        A dictionary reporting summaries of diagnostics.

    Notes
    -----
    .. Adapted from `check_all_hmc_diagnostics()` written by Michael
       Betancourt in his mcmc_diagnostics package.
    """
    # Obtain diagnostics in convenient format
    samples_dict = convert_samples(samples, 'dict')
    _, hmc_diagnostics = _split_dict(samples_dict)
    _validate_dict(hmc_diagnostics, include_all_required_hmc_diagnostics=True)

    # Get number of samples and number of chains
    n_chains, n_draws = next(iter(hmc_diagnostics.values())).shape

    # Output dictionary
    output = {}

    # Compute diagnostic summaries for each chain
    output['divergences'] = np.sum(hmc_diagnostics['divergent__'], axis=1)
    output['divergences_success'] = np.sum(output['divergences']) == 0

    output['average_accept_proxy'] = np.mean(hmc_diagnostics['accept_stat__'], axis=1)
    output['average_accept_proxy_success'] = (
        output['average_accept_proxy'] >= adapt_target_fraction * adapt_target
    ).all()

    output['efmi'] = (
        np.sum(np.diff(hmc_diagnostics['energy__'], axis=1) ** 2)
        / (n_draws - 1) 
        / np.var(hmc_diagnostics['energy__'], axis=1)
    )
    output['efmi_success'] = (output['efmi'] >= efmi_rule_of_thumb).all()

    output['treedepth'] = np.sum(
        hmc_diagnostics['treedepth__'] >= max_treedepth, axis=1
    )
    output['treedepth_success'] = np.sum(output['treedepth']) == 0


    # Build summary, per-chain detail, and description messages, then
    # print at the requested level of verbosity.
    no_divergence_warning = output['divergences_success']
    no_accept_warning = output['average_accept_proxy_success']
    no_efmi_warning = output['efmi_success']
    no_treedepth_warning = output['treedepth_success']
    no_warning = (
        no_divergence_warning
        and no_accept_warning
        and no_efmi_warning
        and no_treedepth_warning
    )

    summary = []
    detail = []
    descriptions = []

    accept_threshold = adapt_target_fraction * adapt_target
    adapt_target_percentage = adapt_target_fraction * 100

    # Divergences
    if not no_divergence_warning:
        total_divergences = int(np.sum(output['divergences']))
        summary.append(
            f"There were a total of {total_divergences} divergences."
        )
        for chain in range(n_chains):
            if output['divergences'][chain] > 0:
                detail.append(
                    f"Chain {chain + 1}: {output['divergences'][chain]:.0f} of "
                    f"{n_draws} transitions "
                    f"({output['divergences'][chain] / n_draws:.3%}) diverged."
                )
        descriptions.append(
            'Divergent Hamiltonian transitions result from '
            'unstable numerical trajectories.  These '
            'instabilities are often due to degenerate target '
            'geometry, especially "pinches".  If there are '
            'only a small number of divergences then running '
            'with adept_delta larger '
            f'than {adapt_target:.3f} may reduce the '
            'instabilities at the cost of more expensive '
            'Hamiltonian transitions.'
        )

    # Average target stepsize adaptation
    if not no_accept_warning:
        n_fail = int(np.sum(output['average_accept_proxy'] < accept_threshold))
        summary.append(
            f"The average proxy acceptance statistic was below "
            f"{adapt_target_percentage:.0f}% of the target "
            f"({adapt_target:.3f}) in {n_fail} of {n_chains} chains."
        )
        for chain in range(n_chains):
            if output['average_accept_proxy'][chain] < accept_threshold:
                detail.append(
                    f"Chain {chain + 1}: Average proxy acceptance statistic "
                    f"({output['average_accept_proxy'][chain]:.3f}) is smaller "
                    f"than {adapt_target_percentage:.0f}% of the target "
                    f"({adapt_target:.3f})."
                )
        descriptions.append(
            'A small average proxy acceptance statistic '
            'indicates that the adaptation of the numerical '
            'integrator step size failed to converge. This is '
            'often due to discontinuous or imprecise '
            'gradients.'
        )

    # E-FMI
    if not no_efmi_warning:
        n_fail = int(np.sum(output['efmi'] < efmi_rule_of_thumb))
        summary.append(
            f"E-FMI was below {efmi_rule_of_thumb} in {n_fail} of "
            f"{n_chains} chains."
        )
        for chain in range(n_chains):
            if output['efmi'][chain] < efmi_rule_of_thumb:
                detail.append(
                    f"Chain {chain + 1}: E-FMI = {output['efmi'][chain]:.3f}, "
                    f"which is below {efmi_rule_of_thumb}."
                )
        descriptions.append(
            f'E-FMI below {efmi_rule_of_thumb} arise when a funnel-like geometry '
            'obstructs how effectively Hamiltonian trajectories '
            'can explore the target distribution.'
        )

    # Treedepth
    if not no_treedepth_warning:
        total_treedepth = int(np.sum(output['treedepth']))
        summary.append(
            f"Maximum tree depth of {max_treedepth} was hit "
            f"{total_treedepth} times."
        )
        for chain in range(n_chains):
            if output['treedepth'][chain] > 0:
                detail.append(
                    f"Chain {chain + 1}: {output['treedepth'][chain]:.0f} of "
                    f"{n_draws} transitions "
                    f"({output['treedepth'][chain] / n_draws:.3%}) saturated "
                    f"the maximum treedepth of {max_treedepth}."
                )
        descriptions.append(
            'Numerical trajectories that saturate the '
            'maximum treedepth have terminated prematurely.  '
            f'Increasing max_depth above {max_treedepth} '
            'should result in more expensive, but more '
            'efficient, Hamiltonian transitions.'
        )

    _render_diagnostic_report(
        verbosity,
        no_warning,
        "All HMC diagnostics are consistent with accurate Markov chain "
        "Monte Carlo.",
        summary,
        detail,
        descriptions,
    )

    return output


def _split_chain(chain):
    """Split a Markov chain into initial and terminal Markov chains.

    Parameters
    ----------
    chain : sequence
        A sequence of expectand values derived from a single Markov
        chain.

    Returns
    -------
    output : list
        Two subsequences (the initial and terminal halves) of expectand
        values.
    """
    N = len(chain)
    M = N // 2
    return [chain[0:M], chain[M:N]]


def _compute_split_rhat(expectand_vals):
    """Compute split Rhat for an expectand output ensemble across a
    collection of Markov chains.

    Parameters
    ----------
    expectand_vals : 2D array
        A two-dimensional array of expectand values with the first
        dimension indexing the Markov chains and the second dimension
        indexing the sequential states within each Markov chain.

    Returns
    -------
    output : float
        Split Rhat estimate.
    """
    vals = np.asarray(expectand_vals)
    n_iter = vals.shape[1]

    # Split each chain in half. When n_iter is odd the two halves differ
    # in length by one, so handle the groups separately.
    M = n_iter // 2
    first_half = vals[:, :M]
    second_half = vals[:, M:]

    # Per-(split-)chain means and variances (ddof=1, matching Welford).
    means = np.concatenate([first_half.mean(axis=1), second_half.mean(axis=1)])
    variances = np.concatenate(
        [first_half.var(axis=1, ddof=1), second_half.var(axis=1, ddof=1)]
    )

    N_chains = means.size
    N = vals.size

    total_mean = means.mean()
    W = variances.mean()
    B = N * np.sum((means - total_mean) ** 2) / (N_chains - 1)

    rhat = np.nan
    if abs(W) > 1e-10:
        rhat = np.sqrt((N - 1 + B / W) / N)

    return rhat


def _compute_tau_hat(vals):
    """Compute the empirical integrated autocorrelation time for a
    sequence of expectand values.

    Parameters
    ----------
    vals : 1D array
        A one-dimensional array of sequential expectand values.

    Returns
    -------
    output : float
        Estimate of the integrated autocorrelation time.
    """
    vals = np.asarray(vals)
    N = len(vals)

    # Center the values about their mean.
    zs = vals - np.mean(vals)

    # Compute empirical autocovariances via the FFT, zero-padding to the
    # next power of two at or above N.
    B = int(2 ** np.ceil(np.log2(N)))
    Fs = np.fft.fft(zs, n=B)
    Ss = np.abs(Fs) ** 2
    acov = np.real(np.fft.ifft(Ss))

    if acov[0] == 0:
        return np.inf

    rhos = acov[:N] / acov[0]

    # Number of complete lag pairs. Each pair uses two consecutive lags
    # (2 * p and 2 * p + 1), so N lags yield N // 2 complete pairs; when N
    # is odd, the final unpaired lag is dropped.
    P = N // 2

    # Construct the asymptotic correlation from the initial monotone
    # sequence of autocorrelation pair sums.
    old_pair_sum = rhos[0] + rhos[1]
    for p in range(1, P):
        current_pair_sum = rhos[2 * p] + rhos[2 * p + 1]

        if current_pair_sum < 0:
            rho_sum = np.sum(rhos[1:2 * p])
            if rho_sum <= -0.25:
                rho_sum = -0.25
            return 1.0 + 2 * rho_sum

        if current_pair_sum > old_pair_sum:
            current_pair_sum = old_pair_sum
            rhos[2 * p] = 0.5 * old_pair_sum
            rhos[2 * p + 1] = 0.5 * old_pair_sum

        old_pair_sum = current_pair_sum

    # The empirical autocorrelations did not decay enough to yield a
    # negative pair sum, so the estimate is unreliable.
    return np.nan


def tau_hat(samples, omit=(), omit_array_entry=()):
    """Compute the empirical integrated autocorrelation time for each 
    expectand for each chain.

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to compute tau_hat, either as a
        cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.
    """
    samples = convert_samples(samples, 'dict')
    expectands, _ = _split_dict(samples)

    _validate_dict(expectands, include_all_required_hmc_diagnostics=False)

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)
    result = {
        key: np.array([_compute_tau_hat(val_chain) for val_chain in val])
        for key, val in expectands.items()
        if _screen_param(key, omit, omit_array_entry)
    }

    return result    


def _compute_xi_hat(vals, xi_hat_threshold=0.25):
    """Compute xi_hat, an estimate of the shape of a generalized Pareto
    distribution from a sample of positive values.

    Parameters
    ----------
    vals : 1D array
        A one-dimensional array of positive values.
    xi_hat_threshold : float, default 0.25
        Quantile fraction used to set the scale of the heuristic Pareto
        grid; the empirical quantile is taken at index
        floor(xi_hat_threshold * N + 0.5) of the sorted values.

    Returns
    -------
    output : float
        Shape parameter estimate.

    Notes
    -----
    .. Uses the method of Zhang and Stephens, "A New and Efficient
       Estimation Method for the Generalized Pareto Distribution",
       Technometrics, 2009, https://doi.org/10.1198/tech.2009.08017.
    .. Within the generalized Pareto family all moments up to the mth
       order are finite if and only if xi_hat < 1 / m.
    """
    sorted_vals = np.sort(vals)
    N = len(sorted_vals)

    if sorted_vals[0] == sorted_vals[-1]:
        return -2

    if sorted_vals[0] < 0:
        print("Sequence values must be positive.")
        return np.nan

    # Empirical quantile used to set the scale of the heuristic grid.
    q = sorted_vals[int(np.floor(xi_hat_threshold * N + 0.5))]
    if q == sorted_vals[0]:
        return -2

    # Heuristic Pareto configuration: a grid of b values whose size grows
    # like sqrt(N).
    M = 20 + int(np.floor(np.sqrt(N)))
    m = np.arange(M)
    b_hat = 1 / sorted_vals[-1] + (1 - np.sqrt(M / (m + 0.5))) / (3 * q)

    # Drop any grid points with b == 0 (they do not contribute).
    b_hat = b_hat[b_hat != 0]

    # Profile shape estimate and log weight (profile log likelihood) at
    # each grid point.
    xi_hats = np.mean(np.log(1 - np.outer(b_hat, sorted_vals)), axis=1)
    log_w = N * (np.log(-b_hat / xi_hats) - xi_hats - 1)

    # Weighted average of the grid (posterior mean of b), stabilized by
    # subtracting the maximum log weight.
    weights = np.exp(log_w - np.max(log_w))
    b_hat_avg = np.sum(b_hat * weights) / np.sum(weights)

    return np.mean(np.log(1 - b_hat_avg * sorted_vals))


def _compute_tail_xi_hats(vals, xi_hat_threshold=0.25):
    """Compute the empirical generalized Pareto shape (xi_hat) for the
    upper and lower tails of a sample of expectand values, ignoring any
    autocorrelation between the values.

    Parameters
    ----------
    vals : 1D array
        A one-dimensional array of expectand values.
    xi_hat_threshold : float, default 0.25
        Quantile fraction passed through to `_compute_xi_hat()` for each
        tail.

    Returns
    -------
    xi_hat_left : float
        Shape estimate for the lower tail, or NaN if the tail is
        ill-defined.
    xi_hat_right : float
        Shape estimate for the upper tail, or NaN if the tail is
        ill-defined.

    Notes
    -----
    .. Uses the method of Zhang and Stephens, "A New and Efficient
       Estimation Method for the Generalized Pareto Distribution",
       Technometrics, 2009, https://doi.org/10.1198/tech.2009.08017.
    .. Within the generalized Pareto family all moments up to the mth
       order are finite if and only if xi_hat < 1 / m.        
    """
    vals = np.asarray(vals)
    v_center = np.median(vals)

    def _trim_tail(tail):
        # The tail values are non-negative distances from the center, so
        # sort ascending and drop the M least-extreme values, keeping the
        # most extreme part of the tail.
        N = len(tail)
        M = int(min(0.2 * N, 9 * np.sqrt(N)))
        return np.sort(tail)[M:]

    # Isolate the lower and upper tails as positive distances from the
    # center; these can be adequately modeled by a generalized Pareto
    # shape for sufficiently well-behaved distributions.
    vals_left = _trim_tail(np.abs(vals[vals <= v_center] - v_center))
    vals_right = _trim_tail(vals[vals > v_center] - v_center)

    # Default to NaN if a tail is ill-defined (too few values).
    xi_hat_left = np.nan
    if len(vals_left) > 40:
        xi_hat_left = _compute_xi_hat(
            vals_left, xi_hat_threshold=xi_hat_threshold
        )

    xi_hat_right = np.nan
    if len(vals_right) > 40:
        xi_hat_right = _compute_xi_hat(
            vals_right, xi_hat_threshold=xi_hat_threshold
        )

    return xi_hat_left, xi_hat_right


def tail_xi_hat(samples, xi_hat_threshold=0.25, omit=(), omit_array_entry=()):
    """Compute the empirical generalized Pareto shape (xi_hat) for the
    upper and lower tails of a sample of expectand values, ignoring any
    autocorrelation between the values.

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to compute xi_hat, either as a
        cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    xi_hat_threshold : float, default 0.25
        Quantile fraction passed through to `_compute_xi_hat()` for each
        tail.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.

    Returns
    -------
    output : dict
        Each value in the dictionary is the shape estimate for lower
        and upper tail for each expandand for each chain for each
        chain for each expectand. Each value is a Numpy array of shape
        (n_chains, 2), where the first index indexes the chain and the
        second index gives the shape index of the lower and upper tail.

    Notes
    -----
    .. Uses the method of Zhang and Stephens, "A New and Efficient
       Estimation Method for the Generalized Pareto Distribution",
       Technometrics, 2009, https://doi.org/10.1198/tech.2009.08017.
    .. Within the generalized Pareto family all moments up to the mth
       order are finite if and only if xi_hat < 1 / m.                
    """
    samples = convert_samples(samples, 'dict')
    expectands, _ = _split_dict(samples)

    _validate_dict(expectands, include_all_required_hmc_diagnostics=False)

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)
    result = {
        key: np.array([
            _compute_tail_xi_hats(vals, xi_hat_threshold=xi_hat_threshold) 
            for vals in expectand_samples
        ])
        for key, expectand_samples in expectands.items()
        if _screen_param(key, omit, omit_array_entry)
    }

    return result

def _compute_ess_hat(expectand_vals):
    """Compute the empirical effective sample size for a set of 
    samples.

    Parameters
    ----------
    expectand_vals : 2D array
        A two-dimensional array of expectand values with the first
        dimension indexing the Markov chains and the second dimension
        indexing the sequential states within each Markov chain.

    Returns
    -------
    output : Numpy array
        Empirical ESS value for the expectand for each chain.
    """
    tau_hat = np.array([_compute_tau_hat(vals) for vals in expectand_vals])
    return expectand_vals.shape[1] / tau_hat


def ess_hat(samples, omit=(), omit_array_entry=()):
    """Compute the empirical effective sample size for all expectand 
    output ensembles across a collection of Markov chains.

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to compute ESS_hat, either as a
        cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.

    Returns
    -------
    output : dict
        Empirical ESS estimate for each chain for each expectand. Each
        value in the dictionary is a Numpy array containing the 
        empirical ESS for each chain.
    """
    samples = convert_samples(samples, 'dict')
    expectands, _ = _split_dict(samples)

    _validate_dict(expectands, include_all_required_hmc_diagnostics=False)

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)
    ess = {
        key: _compute_ess_hat(val)
        for key, val in expectands.items()
        if _screen_param(key, omit, omit_array_entry)
    }

    return ess


def split_rhat(samples, omit=(), omit_array_entry=()):
    """Compute split Rhat for all expectand output ensembles across a
    collection of Markov chains.

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to compute split Rhat, either as a
        cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.

    Returns
    -------
    output : dict
        Split Rhat estimate for each expectand.
    """
    samples = convert_samples(samples, 'dict')
    expectands, _ = _split_dict(samples)

    _validate_dict(expectands, include_all_required_hmc_diagnostics=False)

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)
    rhats = {
        key: _compute_split_rhat(val)
        for key, val in expectands.items()
        if _screen_param(key, omit, omit_array_entry)
    }

    return rhats


def _compute_variance_check(vals, epsilon=1e-10):
    """Compute the variance of an array of values and return `True` if
    the variance is above `epsilon`.
    
    Parameters
    ----------
    vals : Numpy array
        Array of which to compute the emprical variance.
    epsilon : float, default 1e-10
        Small number, below which a variance is considered to be zero.

    Returns
    -------
    output : bool
        True if the variance is greater than epsilon and False 
        otherwise.
    """
    return np.var(vals, ddof=0) >= epsilon


def check_variance(samples, epsilon=1e-10, omit=(), omit_array_entry=()):
    """Compute the variance of each variable for each chain and report
    `True` if the variance if nonzero and `False` if the variance is 
    zero.
    
    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to compute split Rhat, either as a
        cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    epsilon : float, default 1e-10
        Small number, below which a variance is considered to be zero.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.

    Returns
    -------
    output : dict
        Each key is the name of a variable and each value is a Numpy
        array whose length is equal to the number of chains in the
        sampling. Entry i is True if the variance is nonzero and
        False otherwise.
    """
    if epsilon < 0:
        raise ValueError("`epsilon` must be > 0.")

    samples = convert_samples(samples, 'dict')
    expectands, _ = _split_dict(samples)

    _validate_dict(expectands, include_all_required_hmc_diagnostics=False)

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)
    result = {
        key: np.array([
            _compute_variance_check(val_chain)
            for val_chain in val
        ])
        for key, val in expectands.items()
        if _screen_param(key, omit, omit_array_entry)
    }

    return result


def check_expectand_diagnostics(
    samples,
    min_ess_hat_per_chain=100,
    xi_hat_threshold=0.25,
    omit=(),
    omit_array_entry=(),
    verbosity=2,
):
    """Check all expectand-specific diagnostics for a collection of
    Markov chains (tail xi_hats, empirical variance, split Rhat,
    incremental integrated autocorrelation time, and effective sample
    size).

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to check expectand diagnostics, either as
        a cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    min_ess_hat_per_chain : float, default 100
        The minimum empirical effective sample size per chain below which
        a warning is flagged.
    xi_hat_threshold : float, default 0.25
        Tail xi_hats at or above this value are flagged as warnings. Also
        passed through to `tail_xi_hat()` when computing the tail
        xi_hats.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.
    verbosity : int, default 2
        Level of verbosity for messages printed to the screen. 0 prints
        nothing. 1 prints a summary of the diagnostics. 2 prints the
        summary followed by descriptions of what the triggered
        diagnostics mean. 3 prints per-chain detail followed by the
        descriptions.

    Returns
    -------
    output : dict
        Dictionary with keys 'xi_hat', 'variance', 'rhat',
        'inc_tau_hat', and 'ess_hat', whose values are the dictionaries
        returned by `tail_xi_hat()`, `check_variance()`, `split_rhat()`,
        `tau_hat()`, and `ess_hat()`, respectively. Each is paired with a
        Boolean success flag ('xi_hat_success', 'variance_success',
        'rhat_success', 'inc_tau_hat_success', and 'ess_hat_success')
        that is True when no expectand triggered the corresponding
        warning.

    Notes
    -----
    .. Adapted from `check_all_expectand_diagnostics()` written by Michael
       Betancourt in his mcmc_diagnostics package.
    """
    samples = convert_samples(samples, 'dict')
    expectands, _ = _split_dict(samples)
    _validate_dict(expectands, include_all_required_hmc_diagnostics=False)

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)

    # Collect the output of each individual diagnostic function.
    output = {
        'xi_hat': tail_xi_hat(
            samples,
            xi_hat_threshold=xi_hat_threshold,
            omit=omit,
            omit_array_entry=omit_array_entry,
        ),
        'variance': check_variance(
            samples, omit=omit, omit_array_entry=omit_array_entry
        ),
        'rhat': split_rhat(
            samples, omit=omit, omit_array_entry=omit_array_entry
        ),
        'inc_tau_hat': tau_hat(
            samples, omit=omit, omit_array_entry=omit_array_entry
        ),
        'ess_hat': ess_hat(
            samples, omit=omit, omit_array_entry=omit_array_entry
        ),
    }

    # Derive a Boolean success flag for each diagnostic check, along with
    # aggregate summary messages and per-chain detail messages, from the
    # diagnostic values.
    no_xi_hat_warning = True
    no_zvar_warning = True
    no_rhat_warning = True
    no_inc_tau_hat_warning = True
    no_ess_hat_warning = True

    detail = []
    xi_hat_summary = []
    zvar_summary = []
    rhat_summary = []
    inc_tau_summary = []
    ess_summary = []
    xi_hat_failed = []
    zvar_failed = []
    rhat_failed = []
    inc_tau_failed = []
    ess_failed = []

    for name in output['xi_hat']:
        n_draws = expectands[name].shape[1]

        xi_hats = output['xi_hat'][name]
        nonzero_variance = output['variance'][name]
        n_chains = len(xi_hats)

        xi_hat_count = 0
        zvar_count = 0
        inc_tau_count = 0
        ess_count = 0

        for c in range(n_chains):
            # Check tail xi_hats in each Markov chain.
            xi_hat_left, xi_hat_right = xi_hats[c]
            xi_hat_triggered = True

            if np.isnan(xi_hat_left) and np.isnan(xi_hat_right):
                detail.append(
                    f"{name}, chain {c + 1}: Both left and right tail "
                    "xi_hats are NaN."
                )
            elif np.isnan(xi_hat_left):
                detail.append(
                    f"{name}, chain {c + 1}: Left tail xi_hat is NaN."
                )
            elif np.isnan(xi_hat_right):
                detail.append(
                    f"{name}, chain {c + 1}: Right tail xi_hat is NaN."
                )
            elif (
                xi_hat_left >= xi_hat_threshold
                and xi_hat_right >= xi_hat_threshold
            ):
                detail.append(
                    f"{name}, chain {c + 1}: Both left and right tail "
                    f"xi_hats ({xi_hat_left:.3f}, {xi_hat_right:.3f}) "
                    f"exceed {xi_hat_threshold}."
                )
            elif xi_hat_right >= xi_hat_threshold:
                detail.append(
                    f"{name}, chain {c + 1}: Right tail xi_hat "
                    f"({xi_hat_right:.3f}) exceeds {xi_hat_threshold}."
                )
            elif xi_hat_left >= xi_hat_threshold:
                detail.append(
                    f"{name}, chain {c + 1}: Left tail xi_hat "
                    f"({xi_hat_left:.3f}) exceeds {xi_hat_threshold}."
                )
            else:
                xi_hat_triggered = False

            if xi_hat_triggered:
                no_xi_hat_warning = False
                xi_hat_count += 1

            # Check empirical variance in each Markov chain.
            if not nonzero_variance[c]:
                no_zvar_warning = False
                zvar_count += 1
                detail.append(
                    f"{name}, chain {c + 1}: Expectand exhibits "
                    "vanishing empirical variance."
                )

        if xi_hat_count:
            xi_hat_summary.append(
                f"Parameter {name} triggered a tail xi_hat warning in "
                f"{xi_hat_count} of {n_chains} chains."
            )
            xi_hat_failed.append(name)
        if zvar_count:
            zvar_summary.append(
                f"Parameter {name} has vanishing empirical variance in "
                f"{zvar_count} of {n_chains} chains."
            )
            zvar_failed.append(name)

        # Check split Rhat across Markov chains.
        rhat = output['rhat'][name]
        if np.isnan(rhat):
            detail.append(f"{name}: Split Rhat is ill-defined.")
        elif rhat > 1.1:
            no_rhat_warning = False
            detail.append(f"{name}: Split Rhat ({rhat:.3f}) exceeds 1.1.")
            rhat_summary.append(f"Parameter {name} has an Rhat of {rhat:.2f}.")
            rhat_failed.append(name)

        for c in range(n_chains):
            tau_hat_c = output['inc_tau_hat'][name][c]

            # Check incremental integrated autocorrelation time.
            if np.isinf(tau_hat_c):
                no_inc_tau_hat_warning = False
                inc_tau_count += 1
                detail.append(
                    f"{name}, chain {c + 1}: The calculation of tau_hat "
                    "is unreliable because the empirical variance is zero."
                )
            elif np.isnan(tau_hat_c):
                no_inc_tau_hat_warning = False
                inc_tau_count += 1
                detail.append(
                    f"{name}, chain {c + 1}: The calculation of tau_hat "
                    "is unreliable because the empirical autocorrelations "
                    "did not decay sufficiently quickly."
                )
            else:
                inc_tau_hat = tau_hat_c / n_draws
                if inc_tau_hat > 5:
                    no_inc_tau_hat_warning = False
                    inc_tau_count += 1
                    detail.append(
                        f"{name}, chain {c + 1}: Incremental tau_hat "
                        f"({inc_tau_hat:.3f}) is too large."
                    )

            # Check empirical effective sample size.
            ess_hat_c = output['ess_hat'][name][c]
            if ess_hat_c < min_ess_hat_per_chain:
                no_ess_hat_warning = False
                ess_count += 1
                detail.append(
                    f"{name}, chain {c + 1}: ESS_hat ({ess_hat_c:.1f}) "
                    f"is smaller than desired "
                    f"({min_ess_hat_per_chain:.0f})."
                )

        if inc_tau_count:
            inc_tau_summary.append(
                f"Parameter {name} triggered an incremental tau_hat warning "
                f"in {inc_tau_count} of {n_chains} chains."
            )
            inc_tau_failed.append(name)
        if ess_count:
            ess_summary.append(
                f"Parameter {name} has an effective sample size below "
                f"{min_ess_hat_per_chain:.0f} in {ess_count} of {n_chains} "
                "chains."
            )
            ess_failed.append(name)

    output['xi_hat_success'] = no_xi_hat_warning
    output['variance_success'] = no_zvar_warning
    output['rhat_success'] = no_rhat_warning
    output['inc_tau_hat_success'] = no_inc_tau_hat_warning
    output['ess_hat_success'] = no_ess_hat_warning

    # Aggregate summaries grouped by diagnostic, and descriptions.
    summary = (
        xi_hat_summary + zvar_summary + rhat_summary
        + inc_tau_summary + ess_summary
    )

    descriptions = []
    if not no_xi_hat_warning:
        descriptions.append(
            _failed_expectands_phrase(
                xi_hat_failed, "a tail xi_hat warning", "tail xi_hat warnings"
            )
            + " Large tail xi_hats suggest that the expectand might not be "
            "sufficiently integrable."
        )
    if not no_zvar_warning:
        descriptions.append(
            _failed_expectands_phrase(
                zvar_failed, "a zero variance warning", "zero variance warnings"
            )
            + " If the expectands are not constant then zero empirical "
            "variance suggests that the Markov transitions may be "
            "misbehaving."
        )
    if not no_rhat_warning:
        descriptions.append(
            _failed_expectands_phrase(
                rhat_failed, "an Rhat warning", "Rhat warnings"
            )
            + " Split Rhat larger than 1.1 suggests that at least one of "
            "the Markov chains has not reached an equilibrium."
        )
    if not no_inc_tau_hat_warning:
        descriptions.append(
            _failed_expectands_phrase(
                inc_tau_failed,
                "an incremental tau_hat warning",
                "incremental tau_hat warnings",
            )
            + " If the incremental empirical integrated autocorrelation "
            "times are unreliable or too large then the Markov chains have "
            "not explored long enough for the autocorrelation estimates to "
            "be reliable."
        )
    if not no_ess_hat_warning:
        descriptions.append(
            _failed_expectands_phrase(
                ess_failed, "an ESS_hat warning", "ESS_hat warnings"
            )
            + " Small empirical effective sample sizes result in imprecise "
            "Markov chain Monte Carlo estimators."
        )

    no_warning = (
        no_xi_hat_warning
        and no_zvar_warning
        and no_rhat_warning
        and no_inc_tau_hat_warning
        and no_ess_hat_warning
    )

    _render_diagnostic_report(
        verbosity,
        no_warning,
        "All expectands checked appear to be behaving well enough for "
        "reliable Markov chain Monte Carlo estimation.",
        summary,
        detail,
        descriptions,
    )

    return output


def check_all_diagnostics(
    samples,
    adapt_target=0.801,
    adapt_target_fraction=0.9,
    efmi_rule_of_thumb=0.2,
    max_treedepth=10,
    min_ess_hat_per_chain=100,
    xi_hat_threshold=0.25,
    omit=(),
    omit_array_entry=(),
    verbosity=2,
    return_diagnostics=False,
):
    """Check all MCMC diagnostics, both HMC and expectand diagnostics.

    Parameters
    ----------
    samples : cmdstanpy.CmdStanMCMC instance or dict
        MCMC samples from which to check expectand diagnostics, either as
        a cmdstanpy.CmdStanMCMC instance or as a dictionary of
        two-dimensional arrays for each expectand. For the dictionary,
        the first dimension of each element indexes the Markov chains and
        the second dimension indexes the sequential states within each
        Markov chain.
    adapt_target : float, default 0.801
        Target acceptance proxy statistic for step size adaptation. 
        This should be approximately the prescribed value of 
        `adapt_delta` in sampling. The default `adapt_delta` is 0.8,
        which is why the default `adapt_target` is 0.801.
    adapt_target_fraction : float, default 0.9
        Average target acceptance proxy statistic below 
        adapt_target_fraction * adapt_target is flagged as problematic.
    efmi_rule_of_thumb : float, default 0.2
        EFMI values greater than `efmi_rule_of_thumb` are flagged.
    max_treedepth: int, default 10
        Specification of maximum treedepth allowed in MCMC sampling.
    min_ess_hat_per_chain : float, default 100
        The minimum empirical effective sample size per chain below which
        a warning is flagged.
    xi_hat_threshold : float, default 0.25
        Tail xi_hats at or above this value are flagged as warnings. Also
        passed through to `check_expectand_diagnostics()`.
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        each variable. Any variable whose base name matches an entry is
        omitted, whether scalar or array valued. For example,
        `omit='*_pred'` omits every variable whose name ends in '_pred'.
        Compiled `re.Pattern` entries are matched as regular
        expressions. A single string or pattern may be given instead of
        an iterable.
    omit_array_entry : str or iterable of str
        Specific array entries to omit, e.g. 'y_pred[1]' or 'beta[1,2]'.
        Each entry must include bracketed, comma-separated integer
        indices. A single string may be given instead of an iterable.
    verbosity : int, default 2
        Level of verbosity for messages printed to the screen. 0 prints
        nothing. 1 prints a summary of the diagnostics. 2 prints the
        summary followed by descriptions of what the triggered
        diagnostics mean. 3 prints per-chain detail followed by the
        descriptions.
    return_diagnostics : bool, default False
        If True, additionally return a dictionary of diagnostic results.

    Returns
    -------
    warning_code : int
        Warning encoded as a nine-bit binary number, with one bit per
        diagnostic check, using the bit ordering of
        `decode_warning_code()`. A bit is set when the corresponding
        check triggered a warning.
    diagnostics : dict, returned if return_diagnostics is True
        The outputs of `check_hmc_diagnostics()` and
        `check_expectand_diagnostics()` merged into a single dictionary,
        `{**hmc_diagnostics, **expectand_diagnostics}`. From the HMC
        diagnostics, it contains per-chain summaries of the number of
        divergences ('divergences'), the average proxy acceptance
        statistic ('average_accept_proxy'), the E-FMI ('efmi'), and the
        number of transitions that saturated the maximum treedepth
        ('treedepth'), each paired with a Boolean success flag
        ('divergences_success', 'average_accept_proxy_success',
        'efmi_success', and 'treedepth_success') that is True when the
        check passed. From the expectand diagnostics, it contains the
        keys 'xi_hat', 'variance', 'rhat', 'inc_tau_hat', and 'ess_hat',
        whose values are the dictionaries returned by `tail_xi_hat()`,
        `check_variance()`, `split_rhat()`, `tau_hat()`, and `ess_hat()`,
        respectively.
    """
    samples_dict = convert_samples(samples, 'dict')

    omit, omit_array_entry = _parse_omit(omit, omit_array_entry)

    hmc_diagnostics = check_hmc_diagnostics(
        samples_dict,
        adapt_target=adapt_target,
        adapt_target_fraction=adapt_target_fraction,
        efmi_rule_of_thumb=efmi_rule_of_thumb,
        max_treedepth=max_treedepth,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print("")

    expectand_diagnostics = check_expectand_diagnostics(
        samples_dict,
        min_ess_hat_per_chain=min_ess_hat_per_chain,
        xi_hat_threshold=xi_hat_threshold,
        omit=omit,
        omit_array_entry=omit_array_entry,
        verbosity=verbosity,
    )

    # Encode the warning code as a nine-bit binary number, with one bit
    # per check, matching the bit ordering of decode_warning_code().
    warning_code = 0
    if not hmc_diagnostics['divergences_success']:
        warning_code = warning_code | (1 << 0)
    if not hmc_diagnostics['treedepth_success']:
        warning_code = warning_code | (1 << 1)
    if not hmc_diagnostics['efmi_success']:
        warning_code = warning_code | (1 << 2)
    if not hmc_diagnostics['average_accept_proxy_success']:
        warning_code = warning_code | (1 << 3)
    if not expectand_diagnostics['xi_hat_success']:
        warning_code = warning_code | (1 << 4)
    if not expectand_diagnostics['variance_success']:
        warning_code = warning_code | (1 << 5)
    if not expectand_diagnostics['rhat_success']:
        warning_code = warning_code | (1 << 6)
    if not expectand_diagnostics['inc_tau_hat_success']:
        warning_code = warning_code | (1 << 7)
    if not expectand_diagnostics['ess_hat_success']:
        warning_code = warning_code | (1 << 8)

    if return_diagnostics:
        return warning_code, {**hmc_diagnostics, **expectand_diagnostics}

    return warning_code


def decode_warning_code(warning_code, quiet=False, return_dict=False):
    """Parse a binary diagnostic warning code from 
    `check_all_diagnostics()` into individual failures and print the 
    results.

    Parameters
    ----------
    warning_code : int
        Integer whose binary representation summarizes the diagnostic
        output, with one bit per check. A bit set to one indicates that
        the corresponding check triggered a warning. The bit ordering,
        from least to most significant, is:

        - bit 0: divergence
        - bit 1: treedepth
        - bit 2: E-FMI
        - bit 3: average acceptance proxy
        - bit 4: tail xi_hat
        - bit 5: zero variance
        - bit 6: split Rhat
        - bit 7: incremental tau_hat
        - bit 8: minimum ess_hat

    quiet : bool, default False
        If True, suppress printing results to the screen.
    return_dict : bool, default False
        If True, return a dictionary containing test passage info.

    Returns
    -------
    output : dict
        If `return_dict` is True, returns a dictionary where each entry
        is True if the respective diagnostic check passed and False if it
        was not.

    Notes
    -----
    .. Adapted from `decode_warning_code()` written by Michael Betancourt
       in his mcmc_diagnostics package.
    """
    if quiet and not return_dict:
        raise RuntimeError(
            "`quiet` is True and `return_dict` is False, "
            + "so there is nothing to do."
        )

    passed_tests = dict(
        divergence=True,
        treedepth=True,
        efmi=True,
        accept_proxy=True,
        xi_hat=True,
        zero_variance=True,
        rhat=True,
        inc_tau_hat=True,
        ess_hat=True,
    )

    if warning_code & (1 << 0):
        passed_tests["divergence"] = False
        if not quiet:
            print("divergence warning")
    if warning_code & (1 << 1):
        passed_tests["treedepth"] = False
        if not quiet:
            print("treedepth warning")
    if warning_code & (1 << 2):
        passed_tests["efmi"] = False
        if not quiet:
            print("E-FMI warning")
    if warning_code & (1 << 3):
        passed_tests["accept_proxy"] = False
        if not quiet:
            print("average acceptance proxy warning")
    if warning_code & (1 << 4):
        passed_tests["xi_hat"] = False
        if not quiet:
            print("xi_hat warning")
    if warning_code & (1 << 5):
        passed_tests["zero_variance"] = False
        if not quiet:
            print("zero variance warning")
    if warning_code & (1 << 6):
        passed_tests["rhat"] = False
        if not quiet:
            print("Rhat warning")
    if warning_code & (1 << 7):
        passed_tests["inc_tau_hat"] = False
        if not quiet:
            print("incremental tau_hat warning")
    if warning_code & (1 << 8):
        passed_tests["ess_hat"] = False
        if not quiet:
            print("minimum ess_hat warning")
    if warning_code == 0:
        if not quiet:
            print("No diagnostic warnings")

    if return_dict:
        return passed_tests


def _multiprocessing_spawn_context():
    """
    Return a multiprocessing spawn context to avoid fork deadlocks
    """
    try:
        return multiprocessing.get_context("spawn")
    except Exception:
        return multiprocessing.get_context()


def cmdstanpy_prior_sds(
):
    """Compute standard deviations of marginal priors for each 
    parameter using a prior predictive Stan file.
    """


def cmdstanpy_prior_sds(
    seed=None,
    stan_file=None,
    sample_kwargs={},
    var_names=(),
    n_prior_draws=1000,
    no_logging=True,
):
    """Compute standard deviations of prior parameters.

    Parameters
    ----------
    seed : int, default None
        Seed for Stan's random number generator.
    stan_file : str
        Name of file containing stan code.
    sample_kwargs : dict, default {}
        Arguments to be passed to stan_model.sample(), where stan_model
        is compiled from `stan_file`.
    var_names : list or tuple of strings
        Name of variables that have prior distributions.
    n_prior_draws : int, default 1000
        Number of draws from prior predictive distribution to use to
        compute standard deviations.
    no_logging : bool, default False
        If True, disable logging from Stan.

    Returns
    -------
    samples : dict
        Dictionary containing standard deviations of the marginal prior
        distributions of all parameters.
    """
    sample_kwargs = _check_kwarg_vs_dict(sample_kwargs, 'iter_sampling', n_prior_draws)

    prior_samples_cmdstanpy = sample_cmdstanpy_prior_predictive(
        seed,
        stan_file,
        sample_kwargs,
        no_logging=no_logging,
    )

    prior_samples = convert_samples(prior_samples_cmdstanpy, 'dict')

    # We always remove sample files here because we don't need them.
    # Prior SDs are all precomputed before we generate any prior predictive
    # data sets and are used only in shrinkage calculations.
    _remove_sampling_files(prior_samples_cmdstanpy.runset)

    # Compute prior sd's over all draws for the parameters of interest.
    expectands, _ = _split_dict(prior_samples)
    prior_sd = {
        name: np.std(val)
        for name, val in expectands.items()
        if name in var_names or _base_name(name) in var_names
    }

    return prior_sd


def sample_cmdstanpy_prior_predictive(
    seed=None,
    stan_file=None,
    sample_kwargs={},
    remove_sample_files=True, 
    no_logging=True,
):
    """Draw a prior predictive data set using cmdstanpy. This sets
    default sample_kwargs for sample_cmdstanpy() for drawing a
    single prior predictive sample.

    Parameters
    ----------
    seed : int, default None
        Seed for Stan's random number generator.
    stan_file : str
        Name of file containing stan code.
    sample_kwargs : dict, default {}
        Arguments to be passed to stan_model.sample(), where stan_model
        is compiled from `stan_file`.
    remove_sample_files : bool, default True
        If True, remove CSV files of samples generated by CmdStan.
    no_logging : bool, default False
        If True, disable logging from Stan.

    Returns
    -------
    samples : dict
        Dictionary containing prior predictive data.
    """
    sample_kwargs['seed'] = seed

    if 'fixed_param' in sample_kwargs and not sample_kwargs['fixed_param']:
        raise RuntimeError("sample_kwargs['fixed_param'] must be unspecified or False.")
    sample_kwargs['fixed_param'] = True

    if 'show_progress' in sample_kwargs and sample_kwargs['show_progress']:
        warnings.warn("Setting sample_kwargs['show_progress'] to False.")
    sample_kwargs['show_progress'] = False

    if 'thin' in sample_kwargs and sample_kwargs['thin'] != 1:
        raise RuntimeError("sample_kwargs['thin'] must be unspecified or 1.")

    if 'chains' in sample_kwargs and sample_kwargs['chains'] != 1:
        raise RuntimeError("sample_kwargs['chains'] must be unspecified or 1.")
    sample_kwargs['chains'] = 1

    return sample_cmdstanpy(
        stan_file=stan_file,
        sample_kwargs=sample_kwargs,
        remove_sample_files=remove_sample_files,
        no_logging=no_logging,
    )


def sample_cmdstanpy(
    stan_file=None,
    sample_kwargs={}, 
    remove_sample_files=True, 
    no_logging=True,
):
    """Compile and sample out of a Stan file.
    
    Parameters
    ----------
    stan_file : str
        Name of file containing stan code.
    sample_kwargs : dict, default {}
        kwargs to be passed to the sample() method of a 
        cmdstanpy.CmdStanModel instance.
    remove_sample_files : bool, default True
        If True, remove CSV files of samples generated by CmdStan.
    no_logging : bool, default False
        If True, disable logging from Stan.

    Returns
    -------
    samples : dict
        Dictionary of expectands and HMC diagnostics. Each item
        is a 2D Numpy array. The first dimension indexes the Markov 
        chains and the second indexes the sequential states within 
        each chain.

    Notes
    -----
    .. We only pass in a Stan file and not a cmdstanpy.CmdStanModel 
       instance because cmdstanpy will check for a compiled executable
       by default and will not recompile. So, when doing 
       multiprocessing, we can pass strings denoting Stan files, which
       are far easier to pickle than cmdstanpy.CmdStanModel instances.
    """
    if stan_file is None:
        raise RuntimeError("stan_file must be specified.")
    
    if no_logging:
        with disable_logging():
            stan_model = cmdstanpy.CmdStanModel(stan_file)
            samples_cmdstanpy = stan_model.sample(**sample_kwargs)
    else:
        stan_model = cmdstanpy.CmdStanModel(stan_file)
        samples_cmdstanpy = stan_model.sample(**sample_kwargs)

    return_dict = convert_samples(samples_cmdstanpy, 'dict')

    if remove_sample_files:
        _remove_sampling_files(samples_cmdstanpy.runset)

    return return_dict


def inferential_calibration(**kwargs):
    """DEPRECATED."""
    raise RuntimeError(
        "`inferential_calibration()` is deprecated. Use `inferential_calibration_cmdstanpy()`."
    )


def write_prior_predictive_samples_to_csv(samples, csv_file):
    """Write samples to a CSV file in CmdStan format.

    Parameters
    ----------
    samples : dict
        Dictionary of prior predictive samples. Each key is the
        variable name and each value is either a scalar or a 1D array
        of data. Arrays are unpacked to separate columns with dot
        notation for indices (e.g., 'b[1,2]' becomes 'b.1.2' for the
        second element).
    csv_file : str
        Name of a CSV file to write samples.

    Notes
    -----
    The output follows CmdStan's CSV format specification at
    https://mc-stan.org/docs/cmdstan-guide/stan_csv_apdx.html
    """
    if not isinstance(samples, dict):
        raise ValueError("`samples` must be a dictionary.")

    if not samples:
        raise ValueError("`samples` dictionary cannot be empty.")

    def _convert_to_dot_notation(key):
        """Convert bracket notation to dot notation for Stan CSV format."""
        # Replace '[' with '.', remove ']', and replace ',' with '.'
        # e.g., 'c[1,2]' -> 'c.1.2', 'b[1]' -> 'b.1'
        result = key.replace('[', '.').replace(']', '').replace(',', '.')
        return result

    # Build list of (column_name, value) tuples
    columns = []
    for key in sorted(samples.keys()):
        value = samples[key]
        arr = np.asarray(value).astype(np.float64)

        if arr.ndim == 0:
            # Scalar value
            col_name = _convert_to_dot_notation(key)
            columns.append((col_name, arr.item()))
        elif arr.ndim == 1:
            # 1D array - unpack to separate columns
            base_name = _convert_to_dot_notation(key)
            for i, val in enumerate(arr, start=1):
                col_name = f"{base_name}.{i}"
                columns.append((col_name, val))
        else:
            raise ValueError(
                f"Sample values must be scalar or 1D array, got {arr.ndim}D array for key '{key}'."
            )

    # Write CSV file with Stan format header
    with open(csv_file, "w") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"# {timestamp}\n")

        # Write column names
        column_names = [name for name, _ in columns]
        f.write(",".join(column_names) + "\n")

        # Write single data row
        row_values = [str(val) for _, val in columns]
        f.write(",".join(row_values) + "\n")


def inferential_calibration_cmdstanpy(
    prior_predictive_fun=None,
    prior_predictive_kwargs={},
    prior_sds_fun=None,
    prior_sds_kwargs={},
    posterior_stan_file=None,
    measured_data=None,
    additional_posterior_model_data={},
    var_names=None,
    ignored_var_names=(),
    sample_kwargs=None,
    diagnostic_check_kwargs=None,
    cores=1,
    N=400,
    samples_dir="inferential_calibration_samples",
    remove_sample_files=True,
    df_package="polars",
    progress_bar=False,
    output_csv="inferential_calibration_results.csv",
    overwrite=False,
):
    """Perform simulation-based calibration for models using the 
    cmdstanpy interface.

    Parameters
    ----------
    prior_predictive_fun : function
        Function with call signature 
        prior_predictive_fun(seed=None, **prior_predictive_kwargs).
        This function is used to draw prior predictive samples of
        data sets from the posterior. The function need not necessarily
        use a provided seed, but must have a keyword argument allowing
        for seed specification. The function must still run, even if
        `seed` is None. The function must return a dictionary
        of samples, where each key is the variable name and each
        value is either a parameter drawn from the prior or a data
        set drawn from the prior predictive distribution. Note that
        any parameter must be a scalar. E.g., for a matrix-valued
        parameter A, the keys must be 'A[1,1]', 'A[1,2]', and so on.
        Note the 1-indexing.
    prior_predictive_kwargs : dict, default {}
        Kwargs to be passed to prior_predictive_fun().
    prior_sds_fun : function
        Function with call signature 
        prior_sds_fun(**prior_sds_kwargs).
        This function is used to compute the standard deviations of
        the marginal prior for each parameter. The function need not 
        necessarily use a provided seed, but must have a keyword 
        argument allowing for seed specification. The function must 
        return a dictionary where each key is the variable
        name and each value is the standard deviation of the 
        corresponding marginal prior distribution. Note that
        any parameter must be a scalar. E.g., for a matrix-valued
        parameter A, the keys must be 'A[1,1]', 'A[1,2]', and so on.
        Note the 1-indexing.
    prior_sds_kwargs : dict, default {}
        Kwargs to be passed to prior_sds_fun().
    posterior_model : str
        Stan file for the posterior that allows sampling.
    measured_data : dictionary
        A dictionary where each key is the variable name of measured
        data. The value of each entry in the `measured_data` dictionary
        is the type of the corresponding data, almost always either 
        `int` or `float`. Upon generation of a prior predictive data 
        set, a new dictionary is created with keys given by the keys of
        `measured_data` and value given by the generated prior 
        predictive data set.
    additional_posterior_model_data : dict, default {}
        Dictionary with entries specified for the data block of the 
        posterior Stan model. These are all entries in the data block
        of the posterior model that are not measured data generated 
        by `prior_predictive_fun()`.
    var_names : None, or tuple or list of strings
        A list of strings containing parameter names to be considered
        in the inferential calibration analysis. Not all parameters of 
        the model need be considered; only those in `var_names` have 
        rank statistics calculated. Note that for multidimensional
        variables, `var_names` only has the root name. E.g.,
        `var_names=['x', 'y']`, *not* something like
        `var_names=['x[0]', 'x[1]', 'y[0,0]']`. If None, names of
        variables are inferred from output of `prior_predictive_fun()`.
    ignored_var_names : tuple or list of strings, default ()
        Variable names to ignore in the inferential calibration
        procedure. These variables have no rank statistics computed and
        are omitted from the diagnostic checks. This is typically used
        for posterior retrodictive variables and log likelihood 
        variables. As with `var_names`, only the root name is given for
        multidimensional variables, e.g. `ignored_var_names=['x_prc']`,
        *not* `ignored_var_names=['x_prc[0]', 'x_prc[1]']`.
    sample_kwargs : dict, default None
        kwargs to be passed to `sm.sample()`.
    diagnostic_check_kwargs : dict, default None
        kwargs to pass to `check_all_diagnostics()`. If `verbosity` and/or
        `return_diagnostics` are given, they are ignored.
        `max_treedepth` is inferred from `sample_kwargs`.
    cores : int, default 1
        Number of cores to use in the inferential calibration calculation.
    N : int, 400
        Number of simulations to run.
    n_prior_draws_for_sd : int, default 1000
        Number of prior draws to compute the prior standard deviation
        for a parameter in the prior distribution. This standard
        deviation is used in the shrinkage calculation.
    samples_dir : str, default "inferential_calibration_samples"
        Path to directory to store .csv and .txt files generated by
        CmdStan. The directory specified here will NOT be destroyed
        after a calculation, though the files within it may be,
        depending on the `remove_sample_files` kwarg.
    remove_sample_files : bool, default True
        If True, remove .csv and .txt files generated by CmdStan.
    df_package : str, either 'polars' (default) or 'pandas'
        Which package to use for output data frame
    progress_bar : bool, default False
        If True, display a progress bar for the calculation using tqdm.
    output_csv : str, default 'inferential_calibration_results.csv'
        Path to CSV file where results will be written. Results are
        written directly to this file as workers complete (instead of
        collecting in memory). This reduces memory usage and allows
        monitoring progress by examining the growing CSV file. The file
        includes metadata comments at the top with run parameters and
        start time.
    overwrite : bool, default False
        If False, raises RuntimeError if output_csv already exists.
        If True, overwrites existing file with new results.

    Returns
    -------
    output : Polars or Pandas DataFrame
        Data frame with the output of the inferential calibration analysis. It has
        the following columns.

            - trial : Unique trial number for the simulation.
            - warning_code : Warning code based on diagnostic checks outputted by `check_all_diagnostics()`.
            - parameter: The name of the scalar parameter.
            - prior: Value of the parameter used in the simulation. This value was drawn out of the prior distribution.
            - mean : mean parameter value based on sampling out of the posterior in the simulation.
            - sd : standard deviation of the parameter value based on sampling out of the posterior in the simulation.
            - L : The number of bins used in computing the rank statistic. The rank statistic should be uniform on the integers [0, L].
            - rank_statistic : Value of the rank statistic for the parameter for the trial.
            - shrinkage : The shrinkage for the parameter for the given trial. This is computed as 1 - sd / sd_prior, where sd_prior is the standard deviation of the parameters as determined from drawing out of the prior.
            - z_score : The z-score for the parameter for the given trial. This is computed as abs(mean - prior) / sd.
            - tail_xi_hat_left : Left tail xi_hat for the parameter, computed over all chains pooled together.
            - tail_xi_hat_right : Right tail xi_hat for the parameter, computed over all chains pooled together.
            - inc_tau_hat : Incremental empirical integrated autocorrelation time for the parameter, computed over all chains pooled together.
            - variance : Empirical variance of the parameter's posterior samples, computed over all chains pooled together.
            - ESS_hat : Total empirical effective sample size for the parameter, summed over all chains.
            - ESS_hat_per_iter : `ESS_hat` divided by the total number of posterior samples (all chains and draws).
            - ESS_hat_min_per_chain : Minimum per-chain empirical effective sample size for the parameter. This is the quantity the ESS_hat diagnostic warning is based on: a warning is raised (and the corresponding bit set in `warning_code`) when it falls below `min_ess_hat_per_chain`. Because `ESS_hat` is a sum over chains, it can be large even when this per-chain minimum, and hence the warning, indicates poor mixing in a single chain.
            - average_accept_proxy : Average proxy acceptance statistic for the trial, taken over all chains.
            - prior_predictive_seed : Random number seed used by Stan to generate the prior predictive data set for the trial.
            - posterior_seed : Random number seed used by Stan for the posterior MCMC sampling for the trial.
            - prior_predictive_prefix : Full-path prefix of the sampling files (.csv and .txt) generated by Stan when drawing the prior predictive data set for the trial. If `remove_sample_files` is True (so the files are deleted), every entry is null (Polars output) or np.nan (Pandas output).
            - samples_prefix : Full-path prefix of the sampling files (.csv and .txt) generated by Stan for the posterior MCMC sampling for the trial. If `remove_sample_files` is True (so the files are deleted), every entry is null (Polars output) or np.nan (Pandas output).

    Notes
    -----
    .. Each simulation is done by sampling a parameter set out of the
    prior distribution, using those parameters to generate data from
    the likelihood, and then performing posterior sampling based on
    the generated data. A rank statistic for each simulation is
    computed. This rank statistic should be uniformly distributed
    over its L possible values. See https://arxiv.org/abs/1804.06788,
    by Talts, et al., for details.
    """
    if N > 100_000_000:
        raise RuntimeError("Maximum number of simulations is 100 million.")

    if sample_kwargs is None:
        sample_kwargs = {}

    if diagnostic_check_kwargs is None:
        diagnostic_check_kwargs = {}

    if prior_predictive_fun is None:
        raise RuntimeError("`prior_predictive_fun` must be specified.")
    if prior_sds_fun is None:
        raise RuntimeError("`prior_sds_fun` must be specified.")
    if posterior_stan_file is None:
        raise RuntimeError("`posterior_stan_file` must be specified.")
    if measured_data is None:
        raise RuntimeError("`measured_data` must be specified.")
    if measured_data_dtypes is None:
        raise RuntimeError("`measured_data_dtypes` must be specified.")

    if "output_dir" in sample_kwargs:
        raise RuntimeError(
            "The 'output_dir' kwarg is not allowed in `sample_kwargs` because"
            " unambiguous naming is not possible if new sampling is done more"
            " than once per minute, given CmdStanPy's naming convention."
        )

    # Normalize ignored variable names.
    ignored_var_names = (
        (ignored_var_names,)
        if isinstance(ignored_var_names, str)
        else tuple(ignored_var_names)
    )

    # Defaults for diagnostic checks
    diagnostic_check_kwargs["verbosity"] = 0
    diagnostic_check_kwargs["return_diagnostics"] = True
    if "max_treedepth" in diagnostic_check_kwargs:
        if (
            "max_treedepth" in sample_kwargs 
            and sample_kwargs["max_treedepth"] != sample_kwargs["max_treedepth"]
        ):
            raise RuntimeError(
                "Disagreeing values of 'max_treedepth' in `sample_kwargs"
                " and `diagnostic_check_kwargs."
            )
    elif "max_treedepth" in sample_kwargs:
        diagnostic_check_kwargs["max_treedepth"] = sample_kwargs["max_treedepth"]

    # Also omit ignored variables from the diagnostic checks.
    existing_omit = diagnostic_check_kwargs.get("omit", ())
    if isinstance(existing_omit, str):
        existing_omit = (existing_omit,)
    diagnostic_check_kwargs["omit"] = tuple(ignored_var_names) + tuple(existing_omit)

    # We parallelize simulations, not chains within each simulation
    if "n_jobs" in sample_kwargs:
        del sample_kwargs["n_jobs"]
    if "parallel_chains" in sample_kwargs:
        del sample_kwargs["parallel_chains"]

    # Shut off CmdStanPy's progress bar by default
    if "show_progress" not in sample_kwargs:
        sample_kwargs["show_progress"] = False

    # Take a prior sample to check to make sure all variables in 
    # `measured_data` are produced by `prior_predictive_fun()`
    prior_sample = prior_predictive_fun(**prior_predictive_kwargs)
    prior_expectands, _ = _split_dict(prior_sample)

    # Make sure we can extract all pertinent measured data
    for data in measured_data:
        try:
            _ = np.squeeze(extract_variable(prior_sample, data))
        except:
            raise RuntimeError(
                f"Unable to extract data for {data} from prior predictive sample."
            )

    # Parameter names are the unique base names of the expectands that are
    # not measured data. Array-valued parameters (e.g. 'beta[1]',
    # 'beta[2]') collapse to a single base name ('beta').
    prior_param_names = []
    for key in prior_expectands:
        name = _get_var_name(key)
        if name not in measured_data and name not in prior_param_names:
            prior_param_names.append(name)

    # Check parameters
    if var_names is None:
        var_names = prior_param_names
    else:
        for param in var_names:
            if param not in prior_param_names:
                raise RuntimeError(
                    f"parameter `{param}` not in prior sample generated from `prior_predictive_model`."
                )

    # Ignored variables have no rank statistics computed.
    var_names = [name for name in var_names if name not in ignored_var_names]

    # Determine prior SDs for parameters of interest
    prior_sd = prior_sds_fun(**prior_sds_kwargs)

    # Determine number of iterations
    thin = sample_kwargs["thin"] if "thin" in sample_kwargs else 1
    chains = sample_kwargs["chains"] if "chains" in sample_kwargs else 4
    if "iter_sampling" in sample_kwargs:
        iter_sampling = sample_kwargs["iter_sampling"]
    else:
        iter_sampling = 1000
    L = iter_sampling * chains // thin

    # Check if output file exists and handle overwrite
    if os.path.exists(output_csv) and not overwrite:
        raise RuntimeError(
            f"Output file '{output_csv}' already exists. "
            "Set overwrite=True to overwrite the existing file."
        )

    # Create manager for shared lock
    manager = multiprocessing.Manager()
    write_lock = manager.Lock()

    # Initialize CSV file with metadata comments
    start_time = datetime.datetime.now()
    with open(output_csv, "w") as f:
        f.write(f"# Inferential calibration Analysis Results\n")
        f.write(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#\n")
        f.write(f"# Parameters:\n")
        f.write(f"#    N: {N}\n")
        f.write(f"#    cores: {cores}\n")
        f.write(f"#    prior_predictive_fun name: {prior_predictive_fun.__name__}\n")
        f.write("#    prior_predictive_fun source:\n")
        for line in inspect.getsourcelines(prior_predictive_fun)[0]:
            f.write(f"#        {line}\n")
        f.write(f"#   posterior_stan_file: {posterior_stan_file}\n")
        f.write(f"#   var_names: {var_names}\n")
        f.write(f"#   measured_data: {measured_data}\n")
        f.write(f"#   sample_kwargs: {sample_kwargs}\n")
        f.write(f"#   diagnostic_check_kwargs: {diagnostic_check_kwargs}\n")
        f.write(f"#   samples_dir: {samples_dir}\n")
        f.write(f"#   remove_sample_files: {remove_sample_files}\n")
        f.write(f"#\n")
        # Write CSV header
        f.write(
            "trial,parameter,ground_truth,rank_statistic,mean,sd,shrinkage,z_score,Rhat,ESS_hat,ESS_hat_per_iter,ESS_hat_min_per_chain,tail_xi_hat_left,tail_xi_hat_right,inc_tau_hat,variance,n_divergences,n_bad_ebfmi,n_max_treedepth,average_accept_proxy,warning_code,prior_predictive_seed,posterior_seed,L,prior_predictive_prefix,samples_prefix,error\n"
        )

    def arg_input_generator():
        counter = 0
        while counter < N:
            yield (
                prior_predictive_fun,
                prior_predictive_kwargs,
                posterior_stan_file,
                measured_data,
                additional_posterior_model_data,
                var_names,
                sample_kwargs,
                diagnostic_check_kwargs,
                prior_sd,
                samples_dir,
                remove_sample_files,
                output_csv,
                write_lock,
                L,
                counter,  # trial number
            )
            counter += 1

    # Use multiprocessing with spawn context
    mp_context = _multiprocessing_spawn_context()
    with mp_context.Pool(cores) as pool:
        iterator = pool.imap_unordered(
            _perform_inferential_calibration, arg_input_generator(), chunksize=1
        )

        # Consume the iterator (results written to CSV by workers)
        if progress_bar == "notebook":
            for _ in tqdm.tqdm_notebook(iterator, total=N):
                pass
        elif progress_bar == True:
            for _ in tqdm.tqdm(iterator, total=N):
                pass
        elif progress_bar == False:
            for _ in iterator:
                pass
        else:
            raise RuntimeError("Invalid `progress_bar`.")

    # Read back the CSV and return as DataFrame
    if type(df_package) == str and df_package.lower() == "polars":
        return pl.read_csv(output_csv, comment_prefix="#")
    else:
        return pd.read_csv(output_csv, comment="#")


def _perform_inferential_calibration(args):
    """Perform an inferential calibration analysis"""
    (
        prior_predictive_fun,
        prior_predictive_kwargs,
        posterior_stan_file,
        measured_data,
        additional_posterior_model_data,
        var_names,
        sample_kwargs,
        diagnostic_check_kwargs,
        prior_sd,
        samples_dir,
        remove_sample_files,
        output_csv,
        write_lock,
        L,
        trial,
    ) = args

    posterior_model_data = copy.deepcopy(additional_posterior_model_data)
    output_prefix = _get_output_prefix(trial)

    # Seeds for the prior predictive draw and the posterior sampling. If a
    # seed was supplied in `sample_kwargs` it is used for the posterior
    # sampling; otherwise a random seed is drawn. Both seeds are recorded in
    # the output so that each trial can be reproduced.
    prior_predictive_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    posterior_seed = sample_kwargs.get(
        "seed", int(np.random.default_rng().integers(1, 2**31 - 1))
    )

    with disable_logging():
        output_dir = os.path.join(samples_dir, output_prefix + '-prior_pred')

        # Build model locally within worker (should default to pre-build exe file)
        posterior_model = cmdstanpy.CmdStanModel(stan_file=posterior_stan_file)

        prior_sample = prior_predictive_fun(
            seed=prior_predictive_seed, 
            **prior_predictive_kwargs
        )

    prior_sample = convert_samples(prior_sample, 'dict')
    prior_expectands, _ = _split_dict(prior_sample)

    # Record the full-path prefix of the sampling files (.csv and .txt)
    # generated by Stan for the prior predictive draw. If the files are
    # removed, there is nothing to point to, so the prefix is recorded as
    # NaN (which becomes null in Polars output and np.nan in Pandas output).
    if remove_sample_files:
        prior_predictive_prefix = np.nan
    else:
        prior_predictive_prefix = prior_sample_cmdstanpy.runset.file_path("")

    # Default samples prefix; set below if posterior sampling succeeds and
    # the sampling files are retained.
    samples_prefix = np.nan

    # Extract data generated from the prior predictive calculation
    for data in measured_data:
        ar = np.squeeze(extract_variable(prior_sample, data))
        if ar.ndim == 0:
            posterior_model_data[data] = ar.astype(measured_data_dtypes[data]).item()
        else:
            posterior_model_data[data] = ar.astype(measured_data_dtypes[data])

    # Store what the parameters were to generate prior predictive data. There
    # is a single prior draw, so each entry is a single value.
    param_priors = {
        name: val.item()
        for name, val in prior_expectands.items()
        if _get_var_name(name) not in measured_data
    }

    # Generate posterior samples
    try:
        with disable_logging():
            output_dir = os.path.join(samples_dir, output_prefix + '-posterior')
            posterior_samples_cmdstanpy = posterior_model.sample(
                data=posterior_model_data,
                parallel_chains=1,
                output_dir=output_dir,
                **dict(sample_kwargs, seed=posterior_seed),
            )

        posterior_samples = convert_samples(posterior_samples_cmdstanpy, 'dict')

        # Record the full-path prefix of the sampling files generated by
        # Stan for the MCMC. If the files are removed, the prefix is left as
        # NaN. Clean out samples to save disk space when requested.
        if remove_sample_files:
            _remove_sampling_files(posterior_samples_cmdstanpy.runset)
        else:
            samples_prefix = posterior_samples_cmdstanpy.runset.file_path("")

        # Check diagnostics
        warning_code, diagnostics = check_all_diagnostics(
            posterior_samples, **diagnostic_check_kwargs
        )
        err_str = "no error"
        success = True


    except Exception as exception:
        err_str = exception.__str__()
        warnings.warn(f"Trial failure with error message: {err_str}")
        success = False

    if success:
        # Extract posterior draws for the parameters of interest as flat
        # arrays (all chains and draws combined).
        posterior_expectands, _ = _split_dict(posterior_samples)
        names = [
            name
            for name in posterior_expectands
            if _get_var_name(name) in var_names
        ]
        vals = [posterior_expectands[name].ravel() for name in names]

        # Generate output dictionary
        output = {
            name + "_rank_statistic": (vals[i] < param_priors[name]).sum()
            for i, name in enumerate(names)
        }
        for name, p_prior in param_priors.items():
            output[name + "_ground_truth"] = p_prior

        # E-BFMI is flagged per chain when it falls below the rule of thumb.
        efmi_threshold = diagnostic_check_kwargs.get("efmi_rule_of_thumb", 0.2)

        # Compute posterior sensitivities. The tail xi_hat, incremental
        # tau_hat, and variance are computed over all chains pooled together
        # (`val` is the flattened set of draws from every chain).
        for name, val in zip(names, vals):
            output[name + "_mean"] = np.mean(val)
            output[name + "_sd"] = np.std(val)
            output[name + "_z_score"] = (
                output[name + "_mean"] - output[name + "_ground_truth"]
            ) / output[name + "_sd"]
            output[name + "_shrinkage"] = (
                1 - (output[name + "_sd"] / prior_sd[name]) ** 2
            )
            # ESS_hat is the total empirical effective sample size across chains.
            ess_hat = float(np.sum(diagnostics["ess_hat"][name]))
            output[name + "_ESS_hat"] = ess_hat
            output[name + "_ESS_hat_per_iter"] = (
                ess_hat / posterior_expectands[name].size
            )
            # Minimum per-chain ESS_hat. This is the quantity the ESS_hat
            # warning is based on (a warning is raised when it falls below
            # `min_ess_hat_per_chain`), so it explains a fired ESS_hat flag
            # even when the total ESS_hat is large.
            output[name + "_ESS_hat_min_per_chain"] = float(
                np.min(diagnostics["ess_hat"][name])
            )
            output[name + "_Rhat"] = diagnostics["rhat"][name]

            # Left and right tail xi_hats over all chains.
            tail_xi_hat_left, tail_xi_hat_right = _compute_tail_xi_hats(val)
            output[name + "_tail_xi_hat_left"] = tail_xi_hat_left
            output[name + "_tail_xi_hat_right"] = tail_xi_hat_right

            # Incremental integrated autocorrelation time over all chains.
            output[name + "_inc_tau_hat"] = _compute_tau_hat(val) / len(val)

            output[name + "_variance"] = np.var(val)

        output["average_accept_proxy"] = float(
            np.mean(diagnostics["average_accept_proxy"])
        )
        output["n_bad_ebfmi"] = int(np.sum(diagnostics["efmi"] < efmi_threshold))
        output["n_divergences"] = int(np.sum(diagnostics["divergences"]))
        output["n_max_treedepth"] = int(np.sum(diagnostics["treedepth"]))
        output["warning_code"] = warning_code
    else:
        # Parameters of interest, from the prior draw.
        names = [
            name
            for name in prior_expectands
            if _get_var_name(name) in var_names
        ]

        # Generate output dictionary
        output = {name + "_rank_statistic": np.nan for i, name in enumerate(names)}
        for name, p_prior in param_priors.items():
            output[name + "_ground_truth"] = p_prior

        # Posterior sensitivities
        for name in names:
            output[name + "_mean"] = np.nan
            output[name + "_sd"] = np.nan
            output[name + "_z_score"] = np.nan
            output[name + "_shrinkage"] = np.nan
            output[name + "_ESS_hat"] = np.nan
            output[name + "_ESS_hat_per_iter"] = np.nan
            output[name + "_ESS_hat_min_per_chain"] = np.nan
            output[name + "_Rhat"] = np.nan
            output[name + "_tail_xi_hat_left"] = np.nan
            output[name + "_tail_xi_hat_right"] = np.nan
            output[name + "_inc_tau_hat"] = np.nan
            output[name + "_variance"] = np.nan

        output["average_accept_proxy"] = np.nan
        output["n_bad_ebfmi"] = np.nan
        output["n_divergences"] = np.nan
        output["n_max_treedepth"] = np.nan
        output["warning_code"] = np.nan
    output["prior_predictive_seed"] = prior_predictive_seed
    output["posterior_seed"] = posterior_seed
    output["error"] = err_str
    output["L"] = L
    output["prior_predictive_prefix"] = prior_predictive_prefix
    output["samples_prefix"] = samples_prefix

    # Convert output dict to tidied format (multiple rows, one per parameter)
    tidy_df = _tidy_single_inferential_calibration_result(output, trial)

    # Use lock for thread-safe file writing
    with write_lock:
        # Append tidied rows to CSV (headers already written at start)
        tidy_df.to_csv(output_csv, mode="a", header=False, index=False)

    return None


def _get_prior_sds(
    prior_predictive_fun,
    prior_predictive_kwargs,
    var_names,
    n_prior_draws_for_sd,
):
    """Compute standard deviations of prior parameters."""
    # Draw a sample
    prior_sample = prior_predictive_fun(
        seed=int(np.random.default_rng().integers(1, 2**31 - 1)), 
        **prior_predictive_kwargs
    )

    # Initialize dictionary of parameter samples
    param_samples = {
        name: [val] + [None for _ in range(n_prior_draws_for_sd - 1)] 
        for name, val in prior_sample.items()
        if name in var_names or _base_name(name) in var_names
    }

    # Get many samples
    for i in range(1, n_prior_draws_for_sd):
        prior_sample = prior_predictive_fun(
            seed=int(np.random.default_rng().integers(1, 2**31 - 1)), 
            **prior_predictive_kwargs
        ) 
        for name in param_samples:
            param_samples[name][i] = prior_sample[name]

    # Compute standard deviation of all
    return {name: np.std(x) for name, x in param_samples.items()}


def _base_name(name):
    if "[" not in name or name[-1] != "]":
        return name

    ind = name.rfind("[")
    comma_inds = [ind + i for i, char in enumerate(name[ind:]) if char == ","]

    if len(comma_inds) == 0:
        if ind == len(name) - 2 or not name[ind + 1 : -1].isnumeric():
            return name
        else:
            return name[:ind]
    elif comma_inds[0] == ind + 1 or comma_inds[-1] == len(name) - 2:
        return name
    else:
        if name[ind + 1 : -1].replace(",", "").isnumeric():
            return name[:ind]
        else:
            return name


def _tidy_single_inferential_calibration_result(output_dict, trial_number):
    """Tidy a single inferential calibration result dict into multiple rows (one per parameter).

    Parameters
    ----------
    output_dict : dict
        Single result dictionary from _perform_inferential_calibration
    trial_number : int
        Trial number for this result

    Returns
    -------
    df : pandas DataFrame
        Tidied output with one row per parameter
    """
    # Extract parameter names from rank_statistic keys
    rank_stat_keys = [k for k in output_dict.keys() if k.endswith("_rank_statistic")]
    params = [k[: -len("_rank_statistic")] for k in rank_stat_keys]

    stats = [
        "ground_truth",
        "rank_statistic",
        "mean",
        "sd",
        "shrinkage",
        "z_score",
        "Rhat",
        "ESS_hat",
        "ESS_hat_per_iter",
        "ESS_hat_min_per_chain",
        "tail_xi_hat_left",
        "tail_xi_hat_right",
        "inc_tau_hat",
        "variance",
    ]

    aux_keys = [
        "n_divergences",
        "n_bad_ebfmi",
        "n_max_treedepth",
        "average_accept_proxy",
        "warning_code",
        "prior_predictive_seed",
        "posterior_seed",
        "L",
        "prior_predictive_prefix",
        "samples_prefix",
        "error",
    ]

    rows = []
    for param in params:
        row = {"trial": trial_number, "parameter": param}

        # Add statistics for this parameter
        for stat in stats:
            key = f"{param}_{stat}"
            row[stat] = output_dict.get(key, np.nan)

        # Add auxiliary columns
        for aux_key in aux_keys:
            row[aux_key] = output_dict.get(aux_key, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def _tidy_inferential_calibration_output(inferential_calibration_output):
    """Tidy output from inferential_calibration().

    Returns
    -------
    output : DataFrame
        Tidy data frame with inferential calibration results.

    """
    df = inferential_calibration_output.copy()
    df["trial"] = df.index.values

    rank_stat_cols = list(df.columns[df.columns.str.contains("_rank_statistic")])
    params = [col[: col.rfind("_rank_statistic")] for col in rank_stat_cols]

    dfs = []
    stats = [
        "ground_truth",
        "rank_statistic",
        "mean",
        "sd",
        "shrinkage",
        "z_score",
        "Rhat",
        "ESS_hat",
        "ESS_hat_per_iter",
        "ESS_hat_min_per_chain",
        "tail_xi_hat_left",
        "tail_xi_hat_right",
        "inc_tau_hat",
        "variance",
    ]

    aux_cols = [
        "n_divergences",
        "n_bad_ebfmi",
        "n_max_treedepth",
        "average_accept_proxy",
        "warning_code",
        "prior_predictive_seed",
        "posterior_seed",
        "L",
        "prior_predictive_prefix",
        "samples_prefix",
        "trial",
        "error",
    ]
    for param in params:
        cols = [param + "_" + stat for stat in stats]
        sub_df = df[cols + aux_cols].rename(
            columns={old_col: new_col for old_col, new_col in zip(cols, stats)}
        )
        sub_df["parameter"] = param
        dfs.append(sub_df)

    return pd.concat(dfs, ignore_index=True)


def _efmi(energy):
    """Compute energy-fraction of missing information"""
    return np.sum(np.diff(energy) ** 2) / (len(energy) - 1) / np.var(energy)


def _get_var_name(parameter):
    """Convert a parameter name to a var_name. Example: 'alpha[1,2]'
    returns 'alpha'.
    """
    if parameter[-1] != "]":
        return parameter

    ind = parameter.rfind("[")
    if ind == 0 or ind == len(parameter) - 1:
        return parameter

    substr = parameter[ind + 1 : -1]
    if len(substr) == 0:
        return parameter

    if not substr[0].isdigit():
        return parameter

    if not substr[-1].isdigit():
        return parameter

    for char in substr:
        if not (char.isdigit() or char == ","):
            return parameter

    if ",," in substr:
        return parameter

    return parameter[:ind]


def _parse_omit(omit=(), omit_array_entry=()):
    """Normalize and validate `omit` and `omit_array_entry`.

    A single string is wrapped into a one-element tuple. Each
    `omit_array_entry` string is validated to name a specific array
    element with bracketed, comma-separated integer indices (e.g.
    'y_pred[1]' or 'beta[1,2]').

    Parameters
    ----------
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against base variable names.
    omit_array_entry : str or iterable of str
        Specific array entries to omit.

    Returns
    -------
    omit : tuple
        Normalized `omit`.
    omit_array_entry : tuple
        Normalized `omit_array_entry`.
    """
    omit = (omit,) if isinstance(omit, (str, re.Pattern)) else tuple(omit)
    omit_array_entry = (
        (omit_array_entry,)
        if isinstance(omit_array_entry, str)
        else tuple(omit_array_entry)
    )

    for entry in omit_array_entry:
        if re.fullmatch(r".+\[\d+(,\d+)*\]", entry.replace(" ", "")) is None:
            raise ValueError(
                f"Invalid `omit_array_entry` entry '{entry}'. Each entry must "
                "name a specific array element with bracketed, comma-separated "
                "integer indices, e.g. 'y_pred[1]' or 'beta[1,2]'."
            )

    return omit, omit_array_entry


def _screen_param(param, omit=(), omit_array_entry=()):
    """Determine whether a parameter should be kept or omitted.

    Parameters
    ----------
    param : str
        Name of the parameter, possibly indexed (e.g. 'beta[1,2]').
    omit : str, re.Pattern, or iterable thereof
        Glob pattern(s) matched against the base (non-indexed) name of
        `param`. If the base name matches any entry, the parameter is
        omitted, whether it is scalar or array valued. A compiled
        `re.Pattern` entry is matched as a regular expression instead.
    omit_array_entry : str or iterable of str
        Specific array entries (e.g. 'beta[1,2]') to omit. An entry omits
        exactly the matching array element.

    Returns
    -------
    output : bool
        True if the parameter should be kept, False if it should be
        omitted.
    """
    omit = (omit,) if isinstance(omit, (str, re.Pattern)) else omit
    omit_array_entry = (
        (omit_array_entry,)
        if isinstance(omit_array_entry, str)
        else omit_array_entry
    )

    # Omit specific array entries by exact (whitespace-insensitive) match.
    stripped = param.replace(" ", "")
    for entry in omit_array_entry:
        if stripped == entry.replace(" ", ""):
            return False

    # Omit by glob pattern matched against the base variable name.
    base_name = _get_var_name(param)
    for pattern in omit:
        if isinstance(pattern, re.Pattern):
            if pattern.fullmatch(base_name) is not None:
                return False
        elif fnmatch.fnmatchcase(base_name, pattern):
            return False

    return True


def _get_output_prefix(trial):
    """Hand-specify directory for sampling results. This is necessary
    so that CmdStanPy doesn't try to access files with the same name
    while doing multiprocessing."""
    # Set up time stamp + random nine-digit number
    now_ns = time.time_ns()
    dt = datetime.datetime.fromtimestamp(now_ns // 1000000000)
    s = (
        str(trial).zfill(9) 
        + '-'
        + dt.strftime("%Y%m%d%H%M%S") 
        + "." 
        + str(int(now_ns % 1000000000)).zfill(9)
        + "-" 
        + str(random.randint(0, 1000000000)).zfill(9)
    )

    return s


def _remove_sampling_files(runset):
    """Remove files laying around from sampling run."""

    #### Alternatively, we could delete files one-by-one, in case other
    #### stuff is in the directory we're deleting from. Here, we
    #### obliterate everything in the directory.

    # files = []
    # try:
    #     files.append(runset._csv_files)
    # except:
    #     pass

    # try:
    #     files.append(runset._stdout_files)
    # except:
    #     pass

    # try:
    #     files.append(runset._stderr_files)
    # except:
    #     pass

    # try:
    #     files.append(runset._diagnostic_files)
    # except:
    #     pass

    # try:
    #     files.append(runset._profile_files)
    # except:
    #     pass

    # for flist in files:
    #     for fname in flist:
    #         try:
    #             os.remove(fname)
    #         except:
    #             pass
    #
    # try:
    #     os.rmdir(runset._output_dir)
    # except:
    #     pass

    try:
        shutil.rmtree(runset._output_dir)
    except:
        pass


@contextlib.contextmanager
def disable_logging(level=logging.CRITICAL):
    """Context manager for disabling logging when doing MCMC sampling.

    Parameters
    ----------
    level : int, default logging.CRITICAL

    """
    previous_level = logging.root.manager.disable

    logging.disable(level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def parse_warning_code(warning_code, quiet=False, return_dict=False):
    """Deprecated. Use `decode_warning_code()`.
    """
    raise DeprecationWarning('Deprecated. Use `decode_warning_code()`.')

