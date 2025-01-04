import re
import datetime
import time
import random
import contextlib
import copy
import os
import shutil
import sys
import urllib
import glob
import logging
import warnings

try:
    import multiprocess
except:
    import multiprocessing as multiprocess

import tqdm

import numpy as np
import pandas as pd
import polars as pl
import xarray

try:
    import arviz as az
    import arviz.sel_utils
except:
    raise RuntimeError("Could not import ArviZ. Perhaps it is not installed.")

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
    raise RuntimeError('bebi103.stan.StanModel is deprecated.')


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
        raise RuntimeError('This is only meant for installation using Google Colab.')

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
                print(f'CmdStan version {cmdstan_version} successfully installed.')

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
                print(f'CmdStan version {cmdstan_version} successfully installed.')

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
                print(f'CmdStan version {cmdstan_version} successfully installed.')

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
                print(f'CmdStan version {cmdstan_version} successfully installed.')

            return None
        except:
            pass

        # The only way we get here is if we never successfully completed a try
        raise RuntimeError('Unable to install CmdStan, most likely because the URL for the most recent version of CmdStan could not be found. Try finding at URL for a downloadable Colab-compatible version of CmdStan here: https://github.com/stan-dev/cmdstan/releases. Then call `install_cmdstan_colab(cmdstan_colab_url)`.')
    else:
        try:
            urllib.request.urlretrieve(cmdstan_colab_url)
            shutil.unpack_archive(fname)
            os.environ["CMDSTAN"] = f"./cmdstan-{cmdstan_version}"

            if not quiet:
                print('CmdStan successfully installed. You should nonetheless test the installation.')

            return None
        except:
            raise RuntimeError('Unable to install CmdStan, most likely because the inputted URL for CmdStan was invalid. You can run without providing a URL, and the latest version of CmdStan willbe installed; `install_cmdstan_colab()`.')


def include_path():
    """Return path to include files for Stan.

    Returns
    -------
    output : str
        Absolute path to directory containing include files.
    """
    return os.path.join(os.path.dirname(__file__), 'stan_include')


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
    >>> import pandas as pd
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
    >>> data, df = bebi103.stan.df_to_datadict_hier(df,
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
    if 'polars.dataframe.frame.DataFrame' in str(type(df)):
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


def arviz_to_dataframe(
    data, var_names=None, diagnostics=("diverging",), df_package="polars"
):
    """Convert ArviZ InferenceData to a Pandas or Polars data frame.

    Any multi-dimensional parameters are converted to one-dimensional
    equivalents. For example, a 2x2 matrix A is converted to columns
    in the data frame with headings `'A[0,0]'`, `'A[0,1]'`, `'A[1,0]'`,
    and `'A[1,1]'`.

    Only divergence information, chain ID, and draw number (not
    additional sampling stats) are stored in the data frame.

    Parameters
    ----------
    data : ArviZ InferenceData instance
        Results from MCMC sampling to convert to a data frame. It must
        have a `posterior` attribute, and it should have a
        `sample_stats` attribute if divergence information is to be
        stored.
    var_names : list of strings or None
        List of variables present in `data.posterior.data_vars` to store
        in the data frame.  If None, all variables present in
        `data.posterior.data_vars` are stored. Note that *only*
        variables in `data.posterior.data_vars` may be stored.
    diagnostics : list or tuple of strings
        Diagnostic data to be stored in the data frame. The elements of
        the list may include: 'lp', 'acceptance_rate', 'step_size',
        'tree_depth', 'n_steps', 'diverging', and 'energy'.
    df_package : str, one of 'polars' (default) or 'pandas'
        What type of data frame to output

    Returns
    -------
    output : Polars or Pandas DataFrame
        DataFrame with posterior samples and diagnostics. The column
        names of all diagnostics, as well as 'chain' and 'draw', are
        appended with a double-underscore (`__`) to signify that they
        are not samples out of the posterior, but rather metadata about
        a sample.

    """
    if type(data) == xarray.Dataset:
        raise RuntimeError(
            "You need to pass in an ArviZ InferenceData instance, not an xarray Dataset. Maybe you passed in the posterior attribute of the InferenceData instance?"
        )

    if diagnostics is not None:
        if type(diagnostics) == str:
            diagnostics = (diagnostics,)

    var_names = _parameters_to_arviz_var_names(data, var_names)

    if not hasattr(data, "posterior"):
        raise RuntimeError("No posterior contained in `data`.")

    diag_dict = {}
    if diagnostics is not None and len(diagnostics) > 0:
        if not hasattr(data, "sample_stats"):
            raise RuntimeError(
                "Asking for diagnostics, but input has not attribute sample_stats"
            )
        for diag in diagnostics:
            if hasattr(data.sample_stats, diag):
                diag_dict[diag + "__"] = np.ravel(data.sample_stats[diag])
            else:
                raise RuntimeError(f"{diag} not in data.sample_stats.")

    cols, data_as_ndarray = _xarray_to_ndarray(data.posterior, var_names=var_names)

    chain = np.concatenate(
        [
            [i] * data.posterior.sizes["draw"]
            for i in range(data.posterior.sizes["chain"])
        ]
    )

    draw = np.concatenate(
        [data.posterior["draw"].values for i in range(data.posterior.sizes["chain"])]
    )

    if type(df_package) == str and df_package.lower() == "pandas":
        df = pd.DataFrame(data=data_as_ndarray.T, columns=cols)
        df["chain__"] = chain
        df["draw__"] = draw
        for diag, res in diag_dict.items():
            df[diag] = res
    elif type(df_package) == str and df_package.lower() == "polars":
        df = pl.DataFrame(data=data_as_ndarray.T, schema=cols)
        df = df.with_columns(
            pl.Series(chain).alias("chain__"),
            pl.Series(draw).alias("draw__"),
            *[pl.Series(res).alias(diag) for diag, res in diag_dict.items()],
        )
    else:
        raise RuntimeError("Invalid `df_package`. Must be either 'polars' or 'pandas'.")

    return df


def check_divergences(samples, quiet=False, return_diagnostics=False):
    """Check transitions that ended with a divergence.

    Parameters
    ----------
    samples : ArviZ InferenceData instance
        Contains samples to be checked. Must contain both `posterior`
        and `sample_stats`.
    quiet : bool, default False
        If True, do not print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and the number of samples where the tree depth was too
        deep. Otherwise, only return Boolean if the test passed.

    Returns
    -------
    passed : bool
        Return True if tree depth test passed. Return False otherwise.
    n_divergent : int, optional
        Number of divergent samples.
    """
    n_divergent = samples.sample_stats["diverging"].values.sum()
    n_total = samples.sample_stats.sizes["chain"] * samples.sample_stats.sizes["draw"]

    if not quiet:
        msg = "{} of {} ({}%) iterations ended with a divergence.".format(
            n_divergent, n_total, 100 * n_divergent / n_total
        )
        print(msg)

    pass_check = n_divergent == 0

    if not pass_check and not quiet:
        print("  Try running with larger adapt_delta to remove divergences.")

    if return_diagnostics:
        return pass_check, n_divergent
    return pass_check


def check_treedepth(samples, max_treedepth=10, quiet=False, return_diagnostics=False):
    """Check transitions that ended prematurely due to maximum tree depth limit.

    Parameters
    ----------
    samples : ArviZ InferenceData instance
        Contains samples to be checked. Must contain both `posterior`
        and `sample_stats`.
    max_treedepth: int, default 10
        Maximum tree depth used in the calculation.
    quiet : bool, default False
        If True, do not print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and the number of samples where the tree depth was too
        deep. Otherwise, only return Boolean if the test passed.

    Returns
    -------
    passed : bool
        Return True if tree depth test passed. Return False otherwise.
    n_too_deep : int, optional
        Number of samplers wherein the tree depth was greater than
        `max_treedepth`.
    """
    # ArviZ v. 0.11.2 updated treedepth to be tree_depth and v. 0.11.4 reverted
    try:
        n_too_deep = (samples.sample_stats.tree_depth.values >= max_treedepth).sum()
    except:
        n_too_deep = (samples.sample_stats.treedepth.values >= max_treedepth).sum()

    n_total = samples.sample_stats.sizes["chain"] * samples.sample_stats.sizes["draw"]

    if not quiet:
        msg = "{} of {} ({}%) iterations saturated".format(
            n_too_deep, n_total, 100 * n_too_deep / n_total
        )
        msg += " the maximum tree depth of {}.".format(max_treedepth)
        print(msg)

    pass_check = n_too_deep == 0

    if not pass_check and not quiet:
        print(
            "  Try running again with max_treedepth set to a larger value"
            + " to avoid saturation."
        )

    if return_diagnostics:
        return pass_check, n_too_deep
    return pass_check


def check_energy(
    samples, e_bfmi_rule_of_thumb=0.3, quiet=False, return_diagnostics=False
):
    """Checks the energy-Bayes fraction of missing information (E-BFMI)

    Parameters
    ----------
    samples : ArviZ InferenceData instance
        Contains samples to be checked. Must contain both `posterior`
        and `sample_stats`.
    e_bfmi_rule_of_thumb : float, default 0.3 (as per cmdstan)
        Rule of thumb value for E-BFMI. If below this value, there may
        be cause for concern.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and a data frame containing results about the E-BMFI
        tests. Otherwise, only return Boolean if the test passed.

    Returns
    -------
    passed : bool
        Return True if test passed. Return False otherwise.
    e_bfmi_diagnostics : DataFrame
        DataFrame with information about which chains had problematic
        E-BFMIs.
    """
    # Alternatively:
    # samples.sample_stats["energy"].groupby("chain").apply(_ebfmi).values
    ebfmi = az.bfmi(samples)

    problematic = ebfmi < e_bfmi_rule_of_thumb

    pass_check = (~problematic).all()

    if not quiet:
        if pass_check:
            print("E-BFMI indicated no pathological behavior.")
        else:
            for i, (problem, ebfmi_val) in enumerate(zip(problematic, ebfmi)):
                print("Chain {}: E-BFMI = {}".format(i, ebfmi_val))
            print(
                "  E-BFMI below 0.3 indicates you may need to "
                + "reparametrize your model."
            )

    if return_diagnostics:
        return (
            pass_check,
            pd.DataFrame(
                {
                    "chain": np.arange(len(ebfmi)),
                    "E-BFMI": ebfmi,
                    "problematic": ebfmi < e_bfmi_rule_of_thumb,
                }
            ),
        )
    return pass_check


def check_ess(
    samples,
    parameters=None,
    total_ess_rule_of_thumb=100,
    fractional_ess_rule_of_thumb=0.001,
    quiet=False,
    return_diagnostics=False,
):
    """Checks the effective sample size (ESS).

    Parameters
    ----------
    samples : ArviZ InferenceData instance
        Contains samples to be checked. Must contain both `posterior`
        and `sample_stats`.
    parameters : list of str, or None (default)
        Names of parameters to use. If None, use all parameters. For
        multidimensional parameters, each entry must be given
        separately, e.g., `['alpha[0]', 'alpha[1]', 'beta[0,1]']`.
    quiet : bool, default False
        If True, do not print diagnostic result to the screen.
    total_ess_rule_of_thumb : int, default 100
        Rule of thumb for number of effective samples per chain. The
        default of 100 is based on the suggestion of Vehtari, et al., to
        have 50 effective samples per split chain.
    fractional_ess_rule_of_thumb : float, default 0.001
        Rule of thumb value for fractional number of effective samples.
        The default of 0.001 is based on CmdStan's defaults.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and a data frame containing results about the number
        of effective samples tests. Otherwise, only return Boolean if
        the test passed.

    Returns
    -------
    passed : bool
        Return True if test passed. Return False otherwise.
    ess_diagnostics : DataFrame
        Data frame with information about problematic ESS.

    Notes
    -----
    Parameters with ESS given as NaN are not checked.

    References
    ----------
    Vehtari, et al., 2019, https://arxiv.org/abs/1903.08008.
    """
    # For convenience
    N = samples.posterior.sizes["draw"]
    M = samples.posterior.sizes["chain"]
    total_ess_rule_of_thumb *= M

    var_names_arviz = _parameters_to_arviz_var_names(samples, parameters)

    ess = az.ess(samples, var_names=var_names_arviz, method="bulk")
    tail_ess = az.ess(samples, var_names=var_names_arviz, method="tail")

    # Convert to list of names and numpy arrays
    names, ess = _xarray_to_ndarray(ess)
    _, tail_ess = _xarray_to_ndarray(tail_ess)
    if ess.shape == (1, 1):
        ess = np.array([ess[0, 0]])
        tail_ess = np.array([tail_ess[0, 0]])
    else:
        ess = ess.squeeze()
        tail_ess = tail_ess.squeeze()

    # Fractional ESS
    frac_ess = ess / M / N
    frac_tail_ess = tail_ess / M / N

    pass_check = (
        (ess[~np.isnan(ess)] > total_ess_rule_of_thumb).all()
        and (frac_ess[~np.isnan(ess)] > fractional_ess_rule_of_thumb).all()
        and (tail_ess[~np.isnan(tail_ess)] > total_ess_rule_of_thumb).all()
        and (frac_tail_ess[~np.isnan(tail_ess)] > fractional_ess_rule_of_thumb).all()
    )

    if not quiet:
        if not pass_check:
            n_e = 0
            n_f = 0
            for name, e, f, te, tf in zip(
                names, ess, frac_ess, tail_ess, frac_tail_ess
            ):
                if e < total_ess_rule_of_thumb:
                    print("ESS for parameter {} is {}.".format(name, e))
                    n_e += 1
                if f < fractional_ess_rule_of_thumb:
                    print("ESS / iter for parameter {} is {}.".format(name, f))
                    n_f += 1
                if te < total_ess_rule_of_thumb:
                    print("tail-ESS for parameter {} is {}.".format(name, te))
                    n_e += 1
                if tf < fractional_ess_rule_of_thumb:
                    print("ESS / iter for parameter {} is {}.".format(name, tf))
                    n_f += 1
            if n_e > 0:
                print(
                    """  ESS or tail-ESS below 100 per chain indicates that expectation values
  computed from samples are unlikely to be good approximations of the
  true expectation values."""
                )
            if n_f > 0:
                print(
                    """  ESS / iter or tail-ESS / iter below 0.001 indicates that the effective
  sample size has likely been overestimated."""
                )
        else:
            print("Effective sample size looks reasonable for all parameters.")

    if return_diagnostics:
        return (
            pass_check,
            pd.DataFrame(
                data={
                    "parameter": names,
                    "ESS": ess,
                    "ESS_per_iter": frac_ess,
                    "tail_ESS": tail_ess,
                    "tail_ESS_per_iter": frac_tail_ess,
                }
            ),
        )
    return pass_check


def check_rhat(
    samples,
    parameters=None,
    rhat_rule_of_thumb=1.01,
    omit=(),
    quiet=False,
    return_diagnostics=False,
):
    """Checks the potential issues with scale reduction factors. Rhat
    is computed with rank-normalization and folding.

    Parameters
    ----------
    samples : ArviZ InferenceData instance
        Contains samples to be checked. Must contain both `posterior`
        and `sample_stats`.
    parameters : list of str, or None (default)
        Names of parameters to use. If None, use all parameters. For
        multidimensional parameters, each entry must be given
        separately, e.g., `['alpha[0]', 'alpha[1]', 'beta[0,1]']`.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    rhat_rule_of_thumb : float, default 1.01
        Rule of thumb value for maximum allowed R-hat, as per Vehtari,
        et al.
    omit : str, re.Pattern, or list or tuple of str and re.Pattern
        If `parameters` is not provided, all sampled parameters are
        checked for Rhat. We often want to ignore samples of variables
        that a transformed parameters, as their Rhats are irrelevant and
        often NaNs. For each string entry in `omit`, the variable given
        by the string is omitted. For each entry that is a compiled
        regular expression patters (`re.Pattern`), any variable name
        matching the pattern is omitted. By default, no variables are
        omitted.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and a data frame containing results about the number
        of effective samples tests. Otherwise, only return Boolean if
        the test passed.

    Returns
    -------
    passed : bool
        Return True if test passed. Return False otherwise.
    rhat_diagnostics : DataFrame
        Data frame with information about problematic R-hat values.

    References
    ----------
    Vehtari, et al., 2019, https://arxiv.org/abs/1903.08008.

    """
    if omit is None:
        omit = []
    elif type(omit) not in [tuple, list]:
        omit = [omit]

    var_names_arviz = _parameters_to_arviz_var_names(samples, parameters)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rhat = az.rhat(samples, var_names=var_names_arviz, method="rank")

    # Convert to list of names and numpy arrays
    names, rhat = _xarray_to_ndarray(rhat)

    if rhat.shape == (1, 1):
        rhat = np.array([rhat[0, 0]])
    else:
        rhat = rhat.squeeze()

    if parameters is None:
        rhat = np.array(
            [r for name, r in zip(names, rhat) if _screen_param(name, omit)]
        )
        names = [name for name in names if _screen_param(name, omit)]
    else:
        rhat = np.array([r for name, r in zip(names, rhat) if name in parameters])
        names = [name for name, r in zip(names, rhat) if name in parameters]

    nans = np.isnan(rhat)
    non_nan_pass = np.sum(np.array(rhat[~nans]) > rhat_rule_of_thumb) == 0
    nan_pass = nans.sum() == 0

    pass_check = non_nan_pass and nan_pass

    if not quiet:
        if not pass_check:
            if not nan_pass:
                for name, nan in zip(names, nans):
                    if nan:
                        print("Rhat for parameter {} is NaN.".format(name))

            if not non_nan_pass:
                for name, r in zip(names, rhat):
                    if r > rhat_rule_of_thumb:
                        print("Rhat for parameter {} is {}.".format(name, r))
                print(
                    "  Rank-normalized Rhat above 1.01 indicates that the chains very"
                    " likely have not mixed."
                )
        else:
            print("Rhat looks reasonable for all parameters.")

    if return_diagnostics:
        return pass_check, pd.DataFrame(data={"parameter": names, "Rhat": rhat})

    return pass_check


def check_all_diagnostics(
    samples,
    parameters=None,
    e_bfmi_rule_of_thumb=0.3,
    total_ess_rule_of_thumb=100,
    fractional_ess_rule_of_thumb=0.001,
    rhat_rule_of_thumb=1.01,
    omit=(),
    max_treedepth=10,
    quiet=False,
    return_diagnostics=False,
):
    """Checks all MCMC diagnostics

    Parameters
    ----------
    samples : ArviZ InferenceData instance
        Contains samples to be checked. Must contain both `posterior`
        and `sample_stats`.
    parameters : list of str, or None (default)
        Names of parameters to use in checking Rhat and ESS. If None,
        use all parameters. For multidimensional parameters, each entry
        must be given separately, e.g.,
        `['alpha[0]', 'alpha[1]', 'beta[0,1]']`.
    e_bfmi_rule_of_thumb : float, default 0.3
        Rule of thumb value for E-BFMI. If below this value, there may
        be cause for concern.
    total_ess_rule_of_thumb : int, default 100
        Rule of thumb for number of effective samples per chain. The
        default of 100 is based on the suggestion of Vehtari, et al., to
        have 50 effective samples per split chain.
    fractional_ess_rule_of_thumb : float, default 0.001
        Rule of thumb value for fractional number of effective samples.
        The default of 0.001 is based on CmdStan's defaults.
    rhat_rule_of_thumb : float, default 1.1
        Rule of thumb value for maximum allowed R-hat.
    omit : str, re.Pattern, or list or tuple of str and re.Pattern
        If `parameters` is not provided, all sampled parameters are
        checked for Rhat. We often want to ignore samples of variables
        that a transformed parameters, as their Rhats are irrelevant and
        often NaNs. For each string entry in `omit`, the variable given
        by the string is omitted. For each entry that is a compiled
        regular expression patters (`re.Pattern`), any variable name
        matching the pattern is omitted. By default, no variables are
        omitted.
    max_treedepth: int, default 'infer'
        If int, specification of maximum treedepth allowed. If 'infer',
        inferred from `fit`.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return a dictionary containing the results of each test.

    Returns
    -------
    warning_code : int
        When converted to binary, each digit in the code stands for
        whether or not a test passed. A digit of zero indicates the test
        passed. The ordering of the tests goes:

        - ess
        - r_hat
        - divergences
        - tree depth
        - E-BFMI

        For example, a warning code of 12 has a binary representation
        of 01100, which means that R-hat and divergences tests failed.
    return_dict : dict
        Returned if `return_dict` is True. A dictionary with the result
        of each diagnostic test.
    """
    warning_code = 0
    diag_results = {}

    pass_check, diag_results["ess"] = check_ess(
        samples,
        parameters=parameters,
        total_ess_rule_of_thumb=total_ess_rule_of_thumb,
        fractional_ess_rule_of_thumb=fractional_ess_rule_of_thumb,
        quiet=quiet,
        return_diagnostics=True,
    )
    if not pass_check:
        warning_code = warning_code | (1 << 0)

    if not quiet:
        print("")

    pass_check, diag_results["rhat"] = check_rhat(
        samples,
        parameters=parameters,
        rhat_rule_of_thumb=rhat_rule_of_thumb,
        omit=omit,
        quiet=quiet,
        return_diagnostics=True,
    )
    if not pass_check:
        warning_code = warning_code | (1 << 1)

    if not quiet:
        print("")

    pass_check, diag_results["n_divergences"] = check_divergences(
        samples, quiet=quiet, return_diagnostics=True
    )
    if not pass_check:
        warning_code = warning_code | (1 << 2)

    if not quiet:
        print("")

    pass_check, diag_results["n_max_treedepth"] = check_treedepth(
        samples, max_treedepth=max_treedepth, quiet=quiet, return_diagnostics=True
    )
    if not pass_check:
        warning_code = warning_code | (1 << 3)

    if not quiet:
        print("")

    pass_check, diag_results["e_bfmi"] = check_energy(
        samples,
        e_bfmi_rule_of_thumb=e_bfmi_rule_of_thumb,
        quiet=quiet,
        return_diagnostics=True,
    )
    if not pass_check:
        warning_code = warning_code | (1 << 4)

    if return_diagnostics:
        return warning_code, diag_results

    return warning_code


def parse_warning_code(warning_code, quiet=False, return_dict=False):
    """Parses warning code from `check_all_diagnostics()` into
    individual failures and prints results.

    Parameters
    ----------
    warning_code : int
        When converted to binary, each digit in the code stands for
        whether or not a test passed. A digit of zero indicates the test
        passed. The ordering of the tests goes:

        - ESS
        - r_hat
        - divergences
        - tree depth
        - E-BFMI

        For example, a warning code of 12 has a binary representation
        of 01100, which means that R-hat and divergences tests failed.
    quiet : bool, default False
        If True, suppress print results to the screen.
    return_dict : bool, default False
        If True, return a dictionary of containing test passage info.

    Returns
    -------
    output : dict
        If `return_dict` is True, returns a dictionary where each entry
        is True is the respective diagnostic check was passed and False
        if it was not.
    """

    if quiet and not return_dict:
        raise RuntimeError(
            "`quiet` is True and `return_dict` is False, "
            + "so there is nothing to do."
        )

    passed_tests = dict(
        neff=True, rhat=True, divergence=True, treedepth=True, energy=True
    )

    if warning_code & (1 << 0):
        passed_tests["neff"] = False
        if not quiet:
            print("ESS warning")
    if warning_code & (1 << 1):
        passed_tests["rhat"] = False
        if not quiet:
            print("Rhat warning")
    if warning_code & (1 << 2):
        passed_tests["divergence"] = False
        if not quiet:
            print("divergence warning")
    if warning_code & (1 << 3):
        passed_tests["treedepth"] = False
        if not quiet:
            print("treedepth warning")
    if warning_code & (1 << 4):
        passed_tests["energy"] = False
        if not quiet:
            print("energy warning")
    if warning_code == 0:
        if not quiet:
            print("No diagnostic warnings")

    if return_dict:
        return passed_tests


def sbc(
    prior_predictive_model=None,
    posterior_model=None,
    prior_predictive_model_data=None,
    posterior_model_data=None,
    measured_data=None,
    var_names=None,
    measured_data_dtypes=None,
    posterior_predictive_var_names=None,
    log_likelihood_var_name=None,
    sampling_kwargs=None,
    diagnostic_check_kwargs=None,
    cores=1,
    N=400,
    n_prior_draws_for_sd=1000,
    samples_dir="sbc_samples",
    remove_sample_files=True,
    df_package='polars',
    progress_bar=False,
):
    """Perform simulation-based calibration on a Stan Model.

    Parameters
    ----------
    prior_predictive_model : cmdstanpy.model.CmdStanModel
        A Stan model for generating prior predictive data sets.
    posterior_model : cmdstanpy.model.CmdStanModel
        A Stan model of the posterior that allows sampling.
    prior_predictive_model_data : dict
        Dictionary with entries specified by the data block of the prior
        predictive Stan model.
    posterior_model_data : dict
        Dictionary with entries specified by the data block of the prior
        predictive Stan model. Measured data in this dictionary will be
        replaced in each simulation by what was generated by the prior
        predictive model.
    measured_data : list
        A list of strings containing the variable names of measured
        data. Each entry in `measured_data` must be a key in
        `posterior_model_data`.
    var_names : list of strings
        A list of strings containing parameter names to be considered
        in the SBC analysis. Not all parameters of the model need be
        considered; only those in `var_names` have rank statistics
        calculated. Note that for multidimensional
        variables, `var_names` only has the root name. E.g.,
        `var_names=['x', 'y']`, *not* something like
        `var_names=['x[0]', 'x[1]', 'y[0,0]']`.
    posterior_predictive_var_names : list of strings, default None
        List of variables that are posterior predictive. These are
        ignored in the SBC analysis. Note that for multidimensional
        variables, `var_names` only has the root name. E.g.,
        `var_names=['x_ppc', 'y_ppc']`, *not* something like
        `var_names=['x_ppc[0]', 'x_ppc[1]', 'y_ppc[0,0]']`.
    log_likelihood_var_name : string, default None
        Name of variable in the Stan model that stores the log
        likelihood. This is ignored in the SBC analysis.
    measured_data_dtypes : dict, default None
        The key in the dtypes dict is a string representing the data
        name, and the corresponding item is its dtype, almost always
        either `int` or `float`.
    sampling_kwargs : dict, default None
        kwargs to be passed to `sm.sample()`.
    diagnostic_check_kwargs : dict, default None
        kwargs to pass to `check_all_diagnostics()`. If `quiet` and/or
        `return_diagnostics` are given, they are ignored.
        `max_treedepth` is inferred from `sampling_kwargs`.
    cores : int, default 1
        Number of cores to use in the SBC calculation.
    N : int, 400
        Number of simulations to run.
    n_prior_draws_for_sd : int, default 1000
        Number of prior draws to compute the prior standard deviation
        for a parameter in the prior distribution. This standard
        deviation is used in the shrinkage calculation.
    samples_dir : str, default "sbc_samples"
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

    Returns
    -------
    output : Polars or Pandas DataFrame
        Data frame with the output of the SBC analysis. It has
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

    Notes
    -----
    Each simulation is done by sampling a parameter set out of the
    prior distribution, using those parameters to generate data from
    the likelihood, and then performing posterior sampling based on
    the generated data. A rank statistic for each simulation is
    computed. This rank statistic should be uniformly distributed
    over its L possible values. See https://arxiv.org/abs/1804.06788,
    by Talts, et al., for details.

    """
    if measured_data_dtypes is None:
        measured_data_dtypes = {}

    if sampling_kwargs is None:
        sampling_kwargs = {}

    if diagnostic_check_kwargs is None:
        diagnostic_check_kwargs = {}

    if prior_predictive_model is None:
        raise RuntimeError("`prior_predictive_model` must be specified.")
    if posterior_model is None:
        raise RuntimeError("`posterior_model` must be specified.")
    if prior_predictive_model_data is None:
        raise RuntimeError("`prior_predictive_model_data` must be specified.")
    if posterior_model_data is None:
        raise RuntimeError("`posterior_model_data` must be specified.")
    if measured_data is None:
        raise RuntimeError("`measured_data` must be specified.")

    if "output_dir" in sampling_kwargs:
        raise RuntimeError(
            "The 'output_dir' kwarg is not allowed because unambiguous naming is not possible if new sampling is done more than once per minute, given CmdStanPy's naming convention."
        )

    # Defaults for diagnostic checks
    diagnostic_check_kwargs["quiet"] = True
    diagnostic_check_kwargs["return_diagnostics"] = True
    if (
        "max_treedepth" not in diagnostic_check_kwargs
        and "max_treedepth" in sampling_kwargs
    ):
        diagnostic_check_kwargs["max_treedepth"] = sampling_kwargs["max_treedepth"]

    # We parallelize simulations, not chains within each simulation
    if "n_jobs" in sampling_kwargs:
        del sampling_kwargs["n_jobs"]
    if "parallel_chains" in sampling_kwargs:
        del sampling_kwargs["parallel_chains"]

    # Shut off CmdStanPy's progress bar by default
    if "show_progress" not in sampling_kwargs:
        sampling_kwargs["show_progress"] = False

    # Take a prior sample to infer data types
    with disable_logging():
        prior_sample = prior_predictive_model.sample(
            data=prior_predictive_model_data,
            fixed_param=True,
            chains=1,
            iter_sampling=1,
            show_progress=False,
        )
    prior_sample = az.from_cmdstanpy(
        prior=prior_sample, prior_predictive=measured_data
    )

    # Infer dtypes of measured data
    for data in measured_data:
        ar = prior_sample.prior_predictive[data].squeeze()
        if data not in measured_data_dtypes:
            if np.sum(ar != ar.astype(int)) == 0:
                warnings.warn(f"Inferring int dtype for {data}.")
                measured_data_dtypes[data] = int
            else:
                measured_data_dtypes[data] = float

    # Check parameters
    if var_names is None:
        var_names = [
            param
            for param in prior_sample.prior.data_vars
            if len(param) < 2 or param[-2:] != "__"
        ]
    else:
        for param in var_names:
            if param not in prior_sample.prior.data_vars:
                raise RuntimeError(
                    f"parameter `{param}` not in prior sample generated from `prior_predictive_model`."
                )

    # Determine prior SDs for parameters of interest
    prior_sd = _get_prior_sds(
        prior_predictive_model,
        prior_predictive_model_data,
        var_names,
        measured_data,
        n_prior_draws_for_sd,
        samples_dir,
        remove_sample_files,
    )

    def arg_input_generator():
        counter = 0
        while counter < N:
            counter += 1
            yield (
                prior_predictive_model,
                posterior_model,
                prior_predictive_model_data,
                posterior_model_data,
                measured_data,
                var_names,
                measured_data_dtypes,
                sampling_kwargs,
                diagnostic_check_kwargs,
                posterior_predictive_var_names,
                log_likelihood_var_name,
                prior_sd,
                samples_dir,
                remove_sample_files,
            )

    with multiprocess.Pool(cores) as pool:
        if progress_bar == "notebook":
            output = list(
                tqdm.tqdm_notebook(
                    pool.imap(_perform_sbc, arg_input_generator()), total=N
                )
            )
        elif progress_bar == True:
            output = list(
                tqdm.tqdm(pool.imap(_perform_sbc, arg_input_generator()), total=N)
            )
        elif progress_bar == False:
            output = pool.map(_perform_sbc, arg_input_generator())
        else:
            raise RuntimeError("Invalid `progress_bar`.")

    output = pd.DataFrame(output)

    # Determine number of iterations
    thin = sampling_kwargs["thin"] if "thin" in sampling_kwargs else 1
    chains = sampling_kwargs["chains"] if "chains" in sampling_kwargs else 4
    if "iter_sampling" in sampling_kwargs:
        iter_sampling = sampling_kwargs["iter_sampling"]
    else:
        iter_sampling = 1000

    output["L"] = iter_sampling * chains // thin

    if type(df_package) == str and df_package.lower() == 'polars':
        return pl.from_pandas(_tidy_sbc_output(output))
    else:
        return _tidy_sbc_output(output)


def _perform_sbc(args):
    """Perform an SBC analysis"""
    (
        prior_predictive_model,
        posterior_model,
        prior_predictive_model_data,
        posterior_model_data,
        measured_data,
        var_names,
        measured_data_dtypes,
        sampling_kwargs,
        diagnostic_check_kwargs,
        posterior_predictive_var_names,
        log_likelihood_var_name,
        prior_sd,
        samples_dir,
        remove_sample_files,
    ) = args

    posterior_model_data = copy.deepcopy(posterior_model_data)

    with disable_logging():
        output_dir = _get_output_dir(samples_dir, "prior_pred")
        prior_sample_cmdstanpy = prior_predictive_model.sample(
            data=prior_predictive_model_data,
            fixed_param=True,
            chains=1,
            iter_sampling=1,
            show_progress=False,
            output_dir=output_dir,
        )
    prior_sample = az.from_cmdstanpy(
        prior=prior_sample_cmdstanpy, prior_predictive=measured_data
    )
    if remove_sample_files:
        _remove_sampling_files(prior_sample_cmdstanpy.runset)

    # Extract data generated from the prior predictive calculation
    for data in measured_data:
        ar = prior_sample.prior_predictive[data].squeeze().values
        if len(ar.shape) == 0:
            posterior_model_data[data] = ar.astype(measured_data_dtypes[data]).item()
        else:
            posterior_model_data[data] = ar.astype(measured_data_dtypes[data])

    # Store what the parameters were to generate prior predictive data
    names, vals = _xarray_to_ndarray(prior_sample.prior)
    param_priors = {name: val.item() for name, val in zip(names, vals)}

    # Generate posterior samples
    try:
        with disable_logging():
            output_dir = _get_output_dir(samples_dir, "model")
            posterior_samples_cmdstanpy = posterior_model.sample(
                data=posterior_model_data,
                parallel_chains=1,
                output_dir=output_dir,
                **sampling_kwargs,
            )
        posterior_samples = az.from_cmdstanpy(
            posterior=posterior_samples_cmdstanpy,
            posterior_predictive=posterior_predictive_var_names,
            log_likelihood=log_likelihood_var_name,
        )

        # Clean out samples to save disk space
        if remove_sample_files:
            _remove_sampling_files(posterior_samples_cmdstanpy.runset)

        # Check diagnostics
        warning_code, diagnostics = check_all_diagnostics(
            posterior_samples, **diagnostic_check_kwargs
        )
        err_str = 'no error'
        success = True
    except Exception as exception:
        err_str = exception.__str__()
        warnings.warn(f"Trial failure with error message: {err_str}")
        success = False

    if success:
        # Convert output to Numpy array
        names, vals = _xarray_to_ndarray(posterior_samples.posterior, var_names=var_names)

        # Generate output dictionary
        output = {
            name + "_rank_statistic": (vals[i] < param_priors[name]).sum()
            for i, name in enumerate(names)
        }
        for name, p_prior in param_priors.items():
            output[name + "_ground_truth"] = p_prior

        # Compute posterior sensitivities
        for name, val in zip(names, vals):
            output[name + "_mean"] = np.mean(val)
            output[name + "_sd"] = np.std(val)
            output[name + "_z_score"] = (
                output[name + "_mean"] - output[name + "_ground_truth"]
            ) / output[name + "_sd"]
            output[name + "_shrinkage"] = 1 - (output[name + "_sd"] / prior_sd[name]) ** 2
            output[name + "_ESS"] = (
                diagnostics["ess"]
                .loc[diagnostics["ess"]["parameter"] == name, "ESS"]
                .values[0]
            )
            output[name + "_ESS_per_iter"] = (
                diagnostics["ess"]
                .loc[diagnostics["ess"]["parameter"] == name, "ESS_per_iter"]
                .values[0]
            )
            output[name + "_tail_ESS"] = (
                diagnostics["ess"]
                .loc[diagnostics["ess"]["parameter"] == name, "tail_ESS"]
                .values[0]
            )
            output[name + "_tail_ESS_per_iter"] = (
                diagnostics["ess"]
                .loc[diagnostics["ess"]["parameter"] == name, "tail_ESS_per_iter"]
                .values[0]
            )
            output[name + "_Rhat"] = (
                diagnostics["rhat"]
                .loc[diagnostics["rhat"]["parameter"] == name, "Rhat"]
                .values[0]
            )

        output["n_bad_ebfmi"] = diagnostics["e_bfmi"]["problematic"].sum()
        output["n_divergences"] = int(diagnostics["n_divergences"])
        output["n_max_treedepth"] = int(diagnostics["n_max_treedepth"])
        output["warning_code"] = warning_code
    else:
        # Convert output to Numpy array
        names, _ = _xarray_to_ndarray(prior_sample.prior, var_names=var_names)

        # Generate output dictionary
        output = {
            name + "_rank_statistic": np.nan
            for i, name in enumerate(names)
        }
        for name, p_prior in param_priors.items():
            output[name + "_ground_truth"] = p_prior

        # Posterior sensitivities
        for name, val in zip(names, vals):
            output[name + "_mean"] = np.nan
            output[name + "_sd"] = np.nan
            output[name + "_z_score"] = np.nan
            output[name + "_shrinkage"] = np.nan
            output[name + "_ESS"] = np.nan
            output[name + "_ESS_per_iter"] = np.nan
            output[name + "_tail_ESS"] = np.nan
            output[name + "_tail_ESS_per_iter"] = np.nan
            output[name + "_Rhat"] = np.nan

        output["n_bad_ebfmi"] = np.nan
        output["n_divergences"] = np.nan
        output["n_max_treedepth"] = np.nan
        output["warning_code"] = np.nan
    output['error'] = err_str

    return output


def _get_prior_sds(
    prior_predictive_model,
    prior_predictive_model_data,
    var_names,
    measured_data,
    n_prior_draws_for_sd,
    samples_dir,
    remove_sample_files,
):
    """Compute standard deviations of prior parameters."""
    with disable_logging():
        output_dir = _get_output_dir(samples_dir, "prior_pred")
        prior_samples_cmdstanpy = prior_predictive_model.sample(
            data=prior_predictive_model_data,
            fixed_param=True,
            iter_sampling=n_prior_draws_for_sd,
            chains=1,
            show_progress=False,
            output_dir=output_dir,
        )
    prior_samples = az.from_cmdstanpy(
        prior=prior_samples_cmdstanpy, prior_predictive=measured_data
    )

    if remove_sample_files:
        _remove_sampling_files(prior_samples_cmdstanpy.runset)

    # Compute prior sd's
    names, vals = _xarray_to_ndarray(prior_samples.prior)
    prior_sd = {
        name: np.std(val)
        for name, val in zip(names, vals)
        if name in var_names or _base_name(name) in var_names
    }

    return prior_sd


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


def _tidy_sbc_output(sbc_output):
    """Tidy output from sbc().

    Returns
    -------
    output : DataFrame
        Tidy data frame with SBC results.

    """
    df = sbc_output.copy()
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
        "ESS",
        "ESS_per_iter",
        "tail_ESS",
        "tail_ESS_per_iter",
    ]

    aux_cols = [
        "n_divergences",
        "n_bad_ebfmi",
        "n_max_treedepth",
        "warning_code",
        "L",
        "trial",
        "error"
    ]
    for param in params:
        cols = [param + "_" + stat for stat in stats]
        sub_df = df[cols + aux_cols].rename(
            columns={old_col: new_col for old_col, new_col in zip(cols, stats)}
        )
        sub_df["parameter"] = param
        dfs.append(sub_df)

    return pd.concat(dfs, ignore_index=True)


def _xarray_to_ndarray(ds, var_names=None, omit_dunders=True):
    """Convert xarray data set with coordinates `chain` and `draw` to a
    Numpy array and a list of row labels for the Numpy array.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset stored in an ArviZ InferenceData instance. This is
        usually contained in the `prior` or `posterior` fields of the
        InferenceData instance.
    var_names : list of strings, default None
        Names of variables to include. If None, include all. Note that
        for multidimensional variables, `var_names` only has the root
        name. E.g., `var_names=['x', 'y']`, *not* something like
        `var_names=['x[0]', 'x[1]', 'y[0,0]']`.
    omit_dunders : bool, default True
        If True, omit any variable that ends in '__' unless it is
        explicitly included in `var_names`. These are not variables, but
        sampling information computed by Stan.

    Output
    ------
    names : list of strings
        List of names of headings. For a multidimensional xarray
        DataArray, each entry in the array is the variable name,
        followed indexing information. As an example, if `A` is a 2x2
        matrix, then there will be names 'A[0,0]', 'A[0,1]', 'A[1,0]',
        and 'A[1,1]'.
    vals : 2D Numpy array
        Each row has the samples for a given variable. If combined is
        True, each row is a concatenation of samples from all chains.

    """

    names, vals = arviz.sel_utils.xarray_to_ndarray(
        ds, var_names=var_names, combined=True
    )

    names = [
        name.replace("\n", "[").replace(", ", ",") + "]" if "\n" in name else name
        for name in names
    ]

    if var_names is None:
        var_names = []

    inds = [
        i
        for i, name in enumerate(names)
        if name in var_names or len(name) < 2 or name[-2:] != "__"
    ]

    return [name for i, name in enumerate(names) if i in inds], vals[inds, :]


def _ebfmi(energy):
    """Compute energy-Bayes fraction of missing information"""
    return np.sum(np.diff(energy) ** 2) / (len(energy) - 1) / np.var(energy)


def _get_var_name(parameter):
    """Convert a parameter name to a var_name. Example: 'alpha[0,1]'
    returns 'alpha'."""
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


def _parameters_to_arviz_var_names(samples, parameters):
    """Convert individual parameter names to ArviZ var_names.
    E.g., 'alpha[0]' becomes 'alpha'.
    """
    if parameters is not None:
        if type(parameters) not in (list, tuple):
            raise RuntimeError("`parameters` must be a list or tuple.")

        sample_vars = list(samples.posterior.data_vars)

        var_names = [
            param if param in sample_vars else _get_var_name(param)
            for param in parameters
        ]

        # Drop any duplicates
        var_names_set = set()
        var_names = [
            var_name
            for var_name in var_names
            if not (var_name in var_names_set or var_names_set.add(var_name))
        ]

        for var_name in var_names:
            if var_name not in sample_vars:
                raise RuntimeError(f"variable {var_name} not in the input.")
    else:
        var_names = None

    return var_names


def _samples_parameters_to_df(samples, parameters, omit=[]):
    """Convert ArviZ InferenceData posterior results to a data frame"""
    if parameters is None:
        params = None
    else:
        param_dict = {}
        for param in parameters:
            if type(param) in [tuple, list]:
                param_dict[param[0]] = param[1]
            else:
                param_dict[param] = param

        params = list(param_dict.keys())
        parameters = [(k, v) for k, v in param_dict.items()]

    var_names_arviz = _parameters_to_arviz_var_names(samples, params)

    diagnostics = ('diverging',) if hasattr(samples.sample_stats, 'diverging') else tuple()

    df = arviz_to_dataframe(samples, var_names=var_names_arviz, diagnostics=diagnostics, df_package='pandas')

    if parameters is None:
        parameters = [col for col in df.columns if _screen_param(col, omit)]
        params = copy.copy(parameters)

    if hasattr(samples.sample_stats, 'diverging'):
        cols = list(params) + ["chain__", "draw__", "diverging__"]
    else:
        cols = list(params) + ["chain__", "draw__"]

    return parameters, df[cols].copy()


def _screen_param(param, omit):
    if param in ["chain__", "draw__", "diverging__"]:
        return False

    for pattern in omit:
        if type(pattern) == re.Pattern:
            if bool(pattern.match(param)):
                return False
        elif pattern == param:
            return False

    return True


def _get_output_dir(samples_dir, model_name):
    """Hand-specify directory for sampling results. This is necessary
    so that CmdStanPy doesn't try to access files with the same name
    while doing multiprocessing."""
    # Set up time stamp + random nine-digit number
    now_ns = time.time_ns()
    dt = datetime.datetime.fromtimestamp(now_ns // 1000000000)
    s = dt.strftime("%Y%m%d%H%M%S") + "." + str(int(now_ns % 1000000000)).zfill(9)
    s += "-" + str(random.randint(0, 1000000000)).zfill(9)

    subdir = model_name + "-" + s
    return os.path.join(samples_dir, subdir)


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
