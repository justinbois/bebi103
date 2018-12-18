import contextlib
import copy
import itertools
import os
import re
import pickle
import hashlib
import logging
import multiprocessing
import warnings

import tqdm

import numpy as np
import pandas as pd
import numba
import scipy.stats as st

import pystan

import arviz as az

import bokeh.plotting

from . import viz


def StanModel(file=None, charset='utf-8', model_name='anon_model',
              model_code=None, force_compile=False, **kwargs):
    """"Utility to load/save cached compiled Stan models.

    Parameters
    ----------
    file : str or open file
        File from which Stan model is to be read. Cannot be specified
        if `model_code` is not None.
    charset : str, default utf-8
        Character set to be used to decode model.
    model_name : str, 'default anon_model'
        The name of the model.
    model_code : str, default None
        The Stan code to be compiled. If not None, `file` must be None.
    force_compile : bool, deafult False
        If True, compile, even if a cached version exists.
    kwargs : dict
        And kwargs to pass to pystem.StanModel().

    Returns
    -------
    output : pystan.model.StanModel
        A compiled Stan model.

    Notes
    -----
    .. If a Stan model does not exist in the pwd that matches the code
       provided in either `file` or `model_code`, a new Stan model is
       built and compiled and then pickled and stored in pwd. If such a
       model does exist, the pickled model is loaded.
    """

    logger = logging.getLogger('pystan')

    if file and model_code:
        raise ValueError("Specify stan model with `file` or `model_code`, "
                         "not both.")
    if file is None and model_code is None:
        raise ValueError("Model file missing and empty model_code.")
    if file is not None:
        if isinstance(file, str):
            try:
                with open(file, 'rt', encoding=charset) as f:
                    model_code = f.read()
            except:
                logger.critical("Unable to read file specified by `file`.")
                raise
        else:
            model_code = file.read()

    # Make a code_hash to use for file name
    code_hash = hashlib.md5(model_code.encode('ascii')).hexdigest()

    if model_name is None:
        cache_fname = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fname = 'cached-{}-{}.pkl'.format(model_name, code_hash)

    if force_compile:
        sm = pystan.StanModel(model_code=model_code, **kwargs)

        with open(cache_fname, 'wb') as f:
            pickle.dump(sm, f)
    else:
        try:
            sm = pickle.load(open(cache_fname, 'rb'))
        except:
            sm = pystan.StanModel(model_code=model_code, **kwargs)

            with open(cache_fname, 'wb') as f:
                pickle.dump(sm, f)
        else:
            print("Using cached StanModel.")

    return sm


def to_dataframe(fit, pars=None, permuted=False, dtypes=None,
                 inc_warmup=False, diagnostics=True):
    """
    Convert the results of a fit to a DataFrame.

    Parameters
    ----------
    fit : StanFit4Model instance
        The output of fitting a Stan model.
    pars : str or list of strs
        The parameters to extract into the outputted data frame.
    permuted : bool, default False
        If True, the chains are combined and the samples are permuted.
    dypes : dict
        Each entry key, value pair is param_name, data dtype. If None,
        all data types are floats. Be careful: results are cast into
        whatever you specify.
    inc_warmup : bool, default False
        If True, include warmup samples.
    diagnostics : bool, default True
        If True, also include diagnostic information about the samples.
        Note that if no Monte Carlo sampling is done (e.g., if the
        Fixed_param algorithm is used).

    Returns
    -------
    output : DataFrame
        A Pandas DataFrame containing the samples. Each row consists of
        a single sample, including the parameters and diagnostics if
        `diagnostics` is True.

    Notes
    -----
    .. This functionality is present in PyStan >= 2.18. Due to
       compilation issues with higher versions of PyStan, we include
       this functionality here. The output of this function is meant to
       match that of fit.to_dataframe() of a StanFit4Model instance
       from PyStan 2.18 and above. Because of PyStan's GPL license,
       I have re-written this functionality here to enable permissive
       licensing of this module.
    """
    if permuted and inc_warmup:
        raise RuntimeError(
                'If `permuted` is True, `inc_warmup` must be False.')

    if permuted and diagnostics:
        raise RuntimeError(
                'Diagnostics are not available when `permuted` is True.')

    if dtypes is not None and not permuted and pars is None:
        raise RuntimeError('`dtypes` cannot be specified when `permuted`'
                            + ' is False and `pars` is None.')

    try:
        return fit.to_dataframe(pars=pars,
                                permuted=permuted,
                                dtypes=dtypes,
                                inc_warmup=inc_warmup,
                                diagnostics=diagnostics)
    except:
        pass

    # Diagnostics to pull out
    diags = ['divergent__', 'energy__', 'treedepth__', 'accept_stat__',
             'stepsize__', 'n_leapfrog__']

    # Build parameters if not supplied
    if pars is None:
        pars = tuple(fit.flatnames + ['lp__'])
    if type(pars) not in [list, tuple, pd.core.indexes.base.Index]:
        raise RuntimeError('`pars` must be list or tuple or pandas index.')
    if 'lp__' not in pars:
        pars = tuple(list(pars) + ['lp__'])

    # Build dtypes if not supplied
    if dtypes is None:
        dtypes = {par: float for par in pars}
    else:
        dtype['lp__'] = float

    # Make sure dtypes supplied for every parameter
    for par in pars:
        if par not in dtypes:
            raise RuntimeError(f"'{par}' not in `dtypes`.")

    # Retrieve samples
    samples = fit.extract(pars=pars,
                          permuted=permuted,
                          dtypes=dtypes,
                          inc_warmup=inc_warmup)
    n_chains = len(fit.stan_args)
    thin = fit.stan_args[0]['thin']
    n_iters = fit.stan_args[0]['iter'] // thin
    n_warmup = fit.stan_args[0]['warmup'] // thin
    n_samples = n_iters - n_warmup

    # Dimensions of parameters
    dim_dict = {par: dim for par, dim in zip(fit.model_pars, fit.par_dims)}
    dim_dict['lp__'] = []

    if inc_warmup:
        n = (n_warmup + n_samples)*n_chains
        warmup = np.concatenate([[1]*n_warmup + [0]*n_samples
                                    for _ in range(n_chains)]).astype(int)
        chain = np.concatenate([[i+1]*(n_warmup+n_samples)
                                    for i in range(n_chains)]).astype(int)
        chain_idx = np.concatenate([np.arange(1, n_warmup+n_samples+1)
                                        for _ in range(n_chains)]).astype(int)
    else:
        n = n_samples * n_chains
        warmup = np.array([0]*n, dtype=int)
        chain = np.concatenate([[i+1]*n_samples
                                    for i in range(n_chains)]).astype(int)
        chain_idx = np.concatenate([np.arange(1, n_samples+1)
                                        for _ in range(n_chains)]).astype(int)

    if permuted:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(
                data=dict(chain=chain, chain_idx=chain_idx, warmup=warmup))

    if diagnostics:
        sampler_params = fit.get_sampler_params(inc_warmup=inc_warmup)
        diag_vals = list(sampler_params[0].keys())

        # If they are standard, do same order as PyStan 2.18 to_dataframe
        if (len(diag_vals) == len(diags)
                and sum([val not in diags for val in diag_vals]) == 0):
            diag_vals = diags
        for diag in diag_vals:
            df[diag] = np.concatenate([sampler_params[i][diag]
                                            for i in range(n_chains)])
            if diag in ['treedepth__', 'n_leapfrog__']:
                df[diag] = df[diag].astype(int)

    if isinstance(samples, np.ndarray):
        for k, par in enumerate(pars):
            try:
                indices = re.search('\[(\d+)(,\d+)*\]', par).group()
                base_name = re.split('\[(\d+)(,\d+)*\]', par, maxsplit=1)[0]
                col = (base_name
                    + re.sub('\d+', lambda x: str(int(x.group())+1), indices))
            except AttributeError:
                col = par

            df[col] = samples[:,:,k].flatten(order='F')
    else:
        for par in pars:
            if len(dim_dict[par]) == 0:
                df[par] = samples[par].flatten(order='F')
            else:
                for inds in itertools.product(*[range(dim)
                                                for dim in dim_dict[par]]):
                    col = (par + '[' +
                        ','.join([str(ind+1) for ind in inds[::-1]]) + ']')

                    if permuted:
                        array_slice = tuple([slice(n), *inds[::-1]])
                    else:
                        array_slice = tuple([slice(n), slice(n), *inds[::-1]])

                    df[col] = samples[par][array_slice].flatten(order='F')

    return df


def pickle_dump_samples(fit=None, model=None, pkl_file=None):
    """Dump samples into a pickle file.

    Parameters
    ----------
    fit : StanFit4Model instance
        The output of fitting a Stan model.
    model : StanModel instance
        StanModel instance used to get `fit`.
    pkl_file : str
        Name of pickle file to create and dump result.

    Returns
    -------
    None
    """
    if None in [fit, model, pkl_file]:
        raise RuntimeError(
            '`fit`, `model`, and `pkl_file` must all be specified.')

    if os.path.isfile(pkl_file):
        raise RuntimeError(f'File {pkl_file} already exists.')

    with open(pkl_file, 'wb') as f:
        pickle.dump({'model': model, 'fit': fit}, f, protocol=-1)


def pickle_load_samples(pkl_file):
    """Load samples out of a pickle file.

    Parameters
    ----------
    fit : StanFit4Model instance
        The output of fitting a Stan model.
    model : StanModel instance
        StanModel instance used to get `fit`.
    pkl_file : str
        Name of pickle file to create and dump result.

    Returns
    -------
    None
    """
    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict['fit'], data_dict['model']


def extract_array(samples, name):
    """Extract an array values from a DataFrame containing samples.

    Parameters
    ----------
    samples : StanFit4Model instance or Pandas DataFrame
        Samples from a Stan calculation that contains an array for
        extraction.
    name : str
        The name of the array to extract. For example, if `name` is
        'my_array', and the array is one-dimensional, entries with
        variable names 'my_array[1]', 'my_array[2]', ... are extracted.

    Returns
    -------
    output : Pandas DataFrame
        A tidy DataFrame with the extracted array. For a 1-D array, the
        DataFrame has columns:
            ['chain', 'chain_idx', 'warmup', 'index_1', 'name']
        For a 2D array, there is an additional column named 'index_2'. A
        3D array has a column 'index_3' and so on.
    """
    df = _fit_to_df(samples, diagnostics=False)

    regex_name = name
    for char in '[\^$.|?*+(){}':
        regex_name = regex_name.replace(char, '\\'+char)

    # Extract columns that match the name
    sub_df = df.filter(regex=regex_name+'\[(\d+)(,\d+)*\]')

    if len(sub_df.columns) == 0:
        raise RuntimeError(
                "column '{}' is either absent or scalar-valued.".format(name))

    n_entries = len(sub_df.columns)
    n = len(sub_df)

    df_out = pd.DataFrame(data={name: sub_df.values.flatten(order='F')})
    for col in ['chain', 'chain_idx', 'warmup']:
        if col in df:
            df_out[col] = np.concatenate([df[col].values]*n_entries)

    indices = [re.search('\[(\d+)(,\d+)*\]', col).group()[1:-1].split(',')
                           for col in sub_df.columns]
    indices = np.vstack([np.array([[int(i) for i in ind]]*n)
                            for ind in indices])
    ind_df = pd.DataFrame(columns=['index_{0:d}'.format(i)
                                       for i in range(1, indices.shape[1]+1)],
                          data=indices)

    return pd.concat([ind_df, df_out], axis=1)


def extract_par(fit, par, permuted=False, inc_warmup=False,
                logging_disable_level=logging.ERROR):
    """Extract samples of a parameter out of a StanFit4Model.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    par : str
        The name of the variable to extract. For example, if `name` is
        'my_array', and the array is one-dimensional, entries with
        variable names 'my_array[1]', 'my_array[2]', ... are extracted.
    permuted : bool, default False
        If True, the chains are combined and the samples are permuted.
    inc_warmup : bool, default False
        If True, include warmup samples.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Numpy array
        Numpy array with the samples. The first dimension of
        the array is for the iterations; the second for the number of chains;
        the third for the parameters. Vectors and arrays are expanded to one
        parameter (a scalar) per cell, with names indicating the third dimension.

    Notes
    -----
    .. PyStan 2.17.1.0 has a bug where the `extract()` functionality of
       a StanFit4Model instance does not work as advertised. This
       accommodates that functionality.
    """
    if permuted:
        return fit.extract(pars=par, permuted=True, dtypes=None,
                           inc_warmup=inc_warmup)[par]

    if pystan.__version__ >= '2.18':
        with _disable_logging(level=logging_disable_level):
            return fit.extract(pars=par, permuted=False, dtypes=None,
                               inc_warmup=inc_warmup)

    with _disable_logging(level=logging_disable_level):
        data = fit.extract(permuted=False, inc_warmup=inc_warmup, dtypes=None)

    inds = []
    for k, par_name in enumerate(fit.flatnames):
        try:
            indices = re.search('\[(\d+)(,\d+)*\]', par_name).group()
            base_name = re.split('\[(\d+)(,\d+)*\]', par_name, maxsplit=1)[0]
            if base_name == par:
                inds.append(k)
        except AttributeError:
            if par_name == par:
                inds.append(k)

    if len(inds) == 1:
        return data[:,:,inds[0]]

    return data[:,:,inds]


def to_arviz(fit, log_likelihood=None, logging_disable_level=logging.ERROR):
    """Convert a StanFit4Model to an ArviZ data set.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : ArviZ data object.

    Notes
    -----
    .. This function is only necessary because of problems ArviZ has
       with PyStan < 2.18. You should just directly use
       `arviz.from_pystan()` if you have PyStan >= 2.18. Note that this
       function does not allow other kwargs besides `log_likelihood`
       that are available in `arviz.from_pystan()`. The main purpose
       of this function is to get minimal information necessary for
       doing PSIS-LOO and WAIC calculations using ArviZ.
    """
    with _disable_logging(level=logging_disable_level):
        az_data = az.from_pystan(fit=fit, log_likelihood=log_likelihood)

    if pystan.__version__ < '2.18':
        if log_likelihood is not None:
            # Get the log likelihood
            log_lik = np.swapaxes(
                extract_par(fit, log_likelihood,
                            logging_disable_level=logging_disable_level), 0, 1)

            # dims for xarray
            dims = ['chain', 'draw', 'log_likelihood_dim_0']

            # Properly do the log likelihood
            az_data.sample_stats['log_likelihood'] = (dims, log_lik)

        # Get the lp__
        with _disable_logging(level=logging_disable_level):
            lp = np.swapaxes(fit.extract(permuted=False)[:,:,-1], 0, 1)
        az_data.sample_stats['lp'] = (['chain', 'draw'], lp)

    return az_data


def waic(fit, log_likelihood=None, pointwise=False,
         logging_disable_level=logging.ERROR):
    """Compute the WAIC using ArviZ.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    pointwise : bool, default False
        If True, also return point-wise WAIC.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Pandas data frame
        Pandas DataFrame with columns:
          waic: widely available information criterion
          waic_se: standard error of waic
          p_waic: effective number parameters
          var_warn: 1 if posterior variance of the log predictive
            densities exceeds 0.4
          waic_i: and array of the pointwise predictive accuracy, only
            if `pointwise` True
    """
    if log_likelihood is None:
        raise RuntimeError('Must supply `log_likelihood`.')

    az_data = to_arviz(fit, log_likelihood=log_likelihood,
                       logging_disable_level=logging_disable_level)

    return az.waic(az_data, pointwise=pointwise)


def loo(fit, log_likelihood=None, pointwise=False, reff=None,
        logging_disable_level=logging.ERROR):
    """Compute the PSIS-LOO.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    pointwise : bool, default False
        If True, also return point-wise predictive accuracy.
    reff : float, optional
        Relative MCMC efficiency, `effective_n / n` i.e. number of
        effective samples divided by the number of actual samples.
        Computed from trace by default.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Pandas data frame
        Pandas DataFrame with columns:
          loo: approximated Leave-one-out cross-validation
          loo_se: standard error of loo
          p_loo: effective number of parameters
          shape_warn: 1 if the estimated shape parameter of Pareto
            distribution is greater than 0.7 for one or more samples.
          loo_i: array of pointwise predictive accuracy, only if
            `pointwise` True
    """
    if log_likelihood is None:
        raise RuntimeError('Must supply `log_likelihood`.')

    az_data = to_arviz(fit, log_likelihood=log_likelihood,
                       logging_disable_level=logging_disable_level)

    return az.loo(az_data, pointwise=pointwise, reff=reff)


def compare(fit_dict, log_likelihood=None,
            logging_disable_level=logging.ERROR, **kwargs):
    """Compare models.

    Parameters
    ----------
    fit_dict : StanFit4Model instance
        A dictionary where each key it a name attached to a model and
        the value is a StanFit4Model instance that has the posterior
        samples.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Pandas data frame
        Pandas DataFrame with columns:
          IC : Information Criteria (WAIC or LOO).
          pIC : Estimated effective number of parameters.
          dIC : Relative difference between each IC (WAIC or LOO) and
            the lowest IC (WAIC or LOO). It is always 0 for the
            top-ranked model.
          weight: Relative weight for each model.
            This can be loosely interpreted as the probability of each
            model (among the compared model) given the data. By default
            the uncertainty in the weights estimation is considered using
            Bayesian bootstrap.
          SE : Standard error of the IC estimate. If `method` is
            BB-pseudo-BMA, these values are estimated using Bayesian
            bootstrap.
          dSE : Standard error of the difference in IC between each
            model and the top-ranked model. It is always 0 for the
            top-ranked model.
          warning : A value of 1 indicates that the computation of the
            IC may not be reliable. This could be indication of WAIC/LOO
            starting to fail; see http://arxiv.org/abs/1507.04544.

    Notes
    -----
    .. All kwargs are passed into arviz.stats.compare(). These kwargs
       are given in the ArviZ documentation. Importantly, use
       `ic='waic'` or `ic='loo'` to respectively use WAIC or LOO as the
       information criterion. WAIC is the default. Use
       `method='stacking'` for stacking to compute weights (default) and
       `method='BB-pseudo-BMA'` to use pseudo-Bayesian model averaging
       with Akaike-type weights.

    """
    if log_likelihood is None:
        raise RuntimeError('Must supply `log_likelihood`.')

    arviz_dict = {}
    for key in fit_dict:
        arviz_dict[key] = to_arviz(fit_dict[key], 
                                   log_likelihood=log_likelihood,
                                logging_disable_level=logging_disable_level)

    return az.compare(arviz_dict, **kwargs)


def df_to_datadict_hier(df=None, level_cols=None, data_cols=None,
                        sort_cols=[], cowardly=False):
    """Convert a tidy data frame to a data dictionary for a hierarchical
    Stan model.

    Parameters
    ----------
    df : DataFrame
        A tidy Pandas data frame.
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
        accoring to sort_cols.
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
          'N': Total number of data points
          'J_1': Number of hyper parameters for hierarchical level 1.
          'J_2': Number of hyper parameters for hierarchical level 2.
            ... and so on with 'J_3', 'J_4', ...
          'index_1': Set of `J_2` indices defining which level 1
            parameters condition the level 2 parameters.
          'index_2': Set of `J_3` indices defining which level 2
            parameters condition the level 3 parameters.
            ...and so on for 'index_3', etc.
          'index_k': Set of `N` indices defining which of the level k
            parameters condition the data, for a k-level hierarchical
            model.
          `data_col[0]` : Data from first data_col
          `data_col[1]` : Data from second data_col ...and so on.
    df : DataFrame
        Updated input data frame with added columnes with names given by
        `level_col[0] + '_stan'`, `level_col[1] + '_stan'`, etc. These
        contain the integer indices that correspond to the possibly
        non-integer values in the `level_col`s of the original data
        frame. This enables interpretation of Stan results, which have
        everything integer indexed.

    Notes
    -----
    .. Assumes no missing data.
    .. The ordering of data sets is not guaranteed. So, e.g., if you
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
        raise RuntimeError('`df`, `level_cols`, and `data_cols` must all be specified.')

    if type(sort_cols) != list:
        raise RuntimeError('`sort_cols` must be a list.')

    # Get a copy so we don't overwrite
    new_df = df.copy(deep=True)

    if type(level_cols) not in [list, tuple]:
        level_cols = [level_cols]

    if type(data_cols) not in [list, tuple]:
        data_cols = [data_cols]

    level_cols_stan = [col + '_stan' for col in level_cols]

    if cowardly:
        for col in level_cols_stan:
            if col in df:
                raise RuntimeError('column ' + col + ' already in data frame. Cowardly deciding not to overwrite.')

    for col_ind, col in enumerate(level_cols):
        new_df[str(col)+'_stan'] = df.groupby(
                                        level_cols[:col_ind+1]).ngroup() + 1

    new_df = new_df.sort_values(by=level_cols_stan + sort_cols)

    data = dict()
    data['N'] = len(new_df)
    for i, col in enumerate(level_cols_stan):
        data['J_'+ str(i+1)] = len(new_df[col].unique())
    for i, _ in enumerate(level_cols_stan[1:]):
        data['index_' + str(i+1)] = np.array([key[i] for key in new_df.groupby(level_cols_stan[:i+2]).groups]).astype(int)
    data['index_' + str(len(level_cols_stan))] = new_df[level_cols_stan[-1]].values.astype(int)
    for col in data_cols:
        # Check string naming
        new_col = str(col)
        try:
           bytes(new_col, 'ascii')
        except UnicodeEncodeError:
            raise RuntimeError('Column names must be ASCII.')
        if new_col[0].isdigit():
            raise RuntimeError('Column name cannot start with a number.')
        for char in new_col:
            if char in '`~!@#$%^&*()- =+[]{}\\|:;"\',<.>/?':
                raise RuntimeError('Invalid column name for Stan variable.')

        data[new_col] = new_df[col].values

    return data, new_df


def check_divergences(fit, pars=None, quiet=False, return_diagnostics=False):
    """Check transitions that ended with a divergence.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    pars : list of str, or None (default)
        Names of parameters to use in doing diagnostic checks. If None,
        use all parameters.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and the number of samples where the tree depth was too
        deep. Otherwise, only return Boolean if the test passed.

    Results
    -------
    passed : bool
        Return True if tree depth test passed. Return False otherwise.
    n_divergent : int, optional
        Number of divergent samples.
    """

    df = _fit_to_df(fit, pars=pars)

    n_divergent = df['divergent__'].sum()
    n_total = len(df)

    if not quiet:
        msg = '{} of {} ({}%) iterations ended with a divergence.'.format(
                            n_divergent, n_total, 100 * n_divergent / n_total)
        print(msg)

    pass_check = n_divergent == 0

    if not pass_check and not quiet:
        print('  Try running with larger adapt_delta to remove divergences.')

    if return_diagnostics:
        return pass_check, n_divergent
    return pass_check


def check_treedepth(fit=None, pars=None, max_treedepth='infer', df=None,
                    quiet=False, return_diagnostics=False):
    """Check transitions that ended prematurely due to maximum tree depth limit.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    pars : list of str, or None (default)
        Names of parameters to use in doing diagnostic checks. If None,
        use all parameters.
    max_treedepth: int or 'infer'
        If 'infer', infer the maximum tree depth from the fit. If an
        int, use this a max_treedepth. An int is necessary if `df` is
        not None.
    df : DataFrame, default None
        If not None and `fit` is None, `df` is a DataFrame containing
        the fit results and is used in the check. `max_treedepth` must
        be an int if `df` is not None.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and the number of samples where the tree depth was too
        deep. Otherwise, only return Boolean if the test passed.

    Results
    -------
    passed : bool
        Return True if tree depth test passed. Return False otherwise.
    n_too_deep : int, optional
        Number of samplers wherein the tree depth was greater than
        `max_treedepth`.
    """
    if (fit is None and df is None) or (fit is not None and df is not None):
        raise RuntimeError('Exactly one of `fit` or `df` must be specified.')

    if df is not None and type(max_treedepth) != int:
        raise RuntimeError(
                    'If `df` is specified, `max_treedepth` must be int.')

    if max_treedepth == 'infer':
        max_treedepth = _infer_max_treedepth(fit)

    if df is None:
        df = _fit_to_df(fit, pars=pars)

    n_too_deep = (df['treedepth__'] >= max_treedepth).sum()
    n_total = len(df)

    if not quiet:
        msg = '{} of {} ({}%) iterations saturated'.format(
                            n_too_deep, n_total, 100 * n_too_deep / n_total)
        msg += ' the maximum tree depth of {}.'.format(max_treedepth)
        print(msg)

    pass_check = n_too_deep == 0

    if not pass_check and not quiet:
        print('  Try running again with max_treedepth set to a larger value'
               + ' to avoid saturation.')

    if return_diagnostics:
        return pass_check, n_too_deep
    return pass_check


def check_energy(fit, pars=None, quiet=False, e_bfmi_rule_of_thumb=0.2,
                 return_diagnostics=False):
    """Checks the energy-Bayes fraction of missing information (E-BFMI)

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    pars : list of str, or None (default)
        Names of parameters to use in doing diagnostic checks. If None,
        use all parameters.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    e_bfmi_rule_of_thumb : float, default 0.2
        Rule of thumb value for E-BFMI. If below this value, there may
        be cause for concern.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and a data frame containing results about the E-BMFI
        tests. Otherwise, only return Boolean if the test passed.

    Results
    -------
    passed : bool
        Return True if test passed. Return False otherwise.
    e_bfmi_diagnostics : DataFrame
        DataFrame with information about which chains had problematic
        E-BFMIs.
    """
    df = _fit_to_df(fit, pars=pars)

    result = (df.groupby('chain')['energy__']
                .agg(_ebfmi)
                .reset_index()
                .rename(columns={'energy__': 'E-BFMI'}))
    result['problematic'] = result['E-BFMI'] < e_bfmi_rule_of_thumb

    pass_check = (~result['problematic']).all()

    if not quiet:
        if pass_check:
            print('E-BFMI indicated no pathological behavior.')
        else:
            for _, r in result.iterrows():
                print('Chain {}: E-BFMI = {}'.format(r['chain'], r['E-BFMI']))
            print('  E-BFMI below 0.2 indicates you may need to '
                    + 'reparametrize your model.')

    if return_diagnostics:
        return pass_check, result
    return pass_check


def check_n_eff(fit, pars=None, quiet=False, n_eff_rule_of_thumb=0.001,
                return_diagnostics=False):
    """Checks the effective sample size per iteration.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    pars : list of str, or None (default)
        Names of parameters to use in doing diagnostic checks. If None,
        use all parameters.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    n_eff_rule_of_thumb : float, default 0.001
        Rule of thumb value for fractional number of effective samples.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and a data frame containing results about the number
        of effective samples tests. Otherwise, only return Boolean if
        the test passed.

    Results
    -------
    passed : bool
        Return True if test passed. Return False otherwise.
    n_eff_diagnostics : DataFrame
        Data frame with information about problematic n_eff.

    .. Notes
       Parameters with n_eff given as NaN are not checked.
    """
    if 'StanFit4Model' not in str(type(fit)):
        raise RuntimeError('Must imput a StanFit4Model instance.')

    fit_summary = fit.summary(probs=[], pars=pars)
    n_effs = np.array([x[-2] for x in fit_summary['summary']])
    names = fit_summary['summary_rownames']
    n_iter = len(fit.extract()['lp__'])
    ratio = n_effs / n_iter

    pass_check = (ratio[~np.isnan(ratio)] > n_eff_rule_of_thumb).all()

    if not quiet:
        if not pass_check:
            for name, r in zip(names, ratio):
                if r < n_eff_rule_of_thumb:
                    print('n_eff / iter for parameter {} is {}.'.format(name,
                                                                        r))
            print('  n_eff / iter below 0.001 indicates that the effective'
                  + ' sample size has likely been overestimated.')
        else:
            print('n_eff / iter looks reasonable for all parameters.')

    if return_diagnostics:
        return pass_check, pd.DataFrame(data={'parameter': names,
                                              'n_eff/n_iter': ratio})
    return pass_check


def check_rhat(fit, pars=None, quiet=False, rhat_rule_of_thumb=1.1,
               known_rhat_nans=[], return_diagnostics=False):
    """Checks the potential issues with scale reduction factors.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    pars : list of str, or None (default)
        Names of parameters to use in doing diagnostic checks. If None,
        use all parameters.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    rhat_rule_of_thumb : float, default 1.1
        Rule of thumb value for maximum allowed R-hat.
    known_rhat_nans : list, default []
        List of parameter names which are known to have R-hat be NaN.
        These are typically parameters that are deterministic.
        Parameters in this list are ignored.
    return_diagnostics : bool, default False
        If True, return both a Boolean about whether the diagnostic
        passed and a data frame containing results about the number
        of effective samples tests. Otherwise, only return Boolean if
        the test passed.

    Results
    -------
    passed : bool
        Return True if test passed. Return False otherwise.
    rhat_diagnostics : DataFrame
        Data frame with information about problematic R-hat values.
    """
    if 'StanFit4Model' not in str(type(fit)):
        raise RuntimeError('Must imput a StanFit4Model instance.')

    fit_summary = fit.summary(probs=[], pars=pars)
    names = fit_summary['summary_rownames']
    known_nan = [True if name in known_rhat_nans else False for name in names]

    rhat = np.array([x[-1] for x in fit_summary['summary']])

    pass_check = (np.isnan(rhat[~np.array(known_nan)]).sum() == 0
                  and np.all(rhat[~np.array(known_nan)] < rhat_rule_of_thumb))

    if not quiet:
        if not pass_check:
            for name, r, nan in zip(names, rhat, known_nan):
                if (np.isnan(r) and not nan) or r > rhat_rule_of_thumb:
                    print('Rhat for parameter {} is {}.'.format(name, r))
            print('  Rhat above 1.1 indicates that the chains very likely'
                  + ' have not mixed')
        else:
            print('Rhat looks reasonable for all parameters.')

    if return_diagnostics:
        return pass_check, pd.DataFrame(data={'parameter': names,
                                              'Rhat': rhat})
    return pass_check


def check_all_diagnostics(fit, pars=None, e_bfmi_rule_of_thumb=0.2,
                          n_eff_rule_of_thumb=0.001, rhat_rule_of_thumb=1.1,
                          known_rhat_nans=[], max_treedepth='infer',
                          quiet=False, return_diagnostics=False):
    """Checks all MCMC diagnostics

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    pars : list of str, or None (default)
        Names of parameters to use in doing diagnostic checks. If None,
        use all parameters.
    e_bfmi_rule_of_thumb : float, default 0.2
        Rule of thumb value for E-BFMI. If below this value, there may
        be cause for concern.
    rhat_rule_of_thumb : float, default 1.1
        Rule of thumb value for maximum allowed R-hat.
    known_rhat_nans : list, default []
        List of parameter names which are known to have R-hat be NaN.
        These are typically parameters that are deterministic.
        Parameters in this list are ignored.
    max_treedepth: int, default 'infer'
        If int, specification of maximum treedepth allowed. If 'infer',
        inferred from `fit`.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.
    return_diagnostics : bool, default False
        If True, return a dictionary containing the results of each test.

    Results
    -------
    warning_code : int
        When converted to binary, each digit in the code stands for
        whether or not a test passed. A digit of zero indicates the test
        passed. The ordering of the tests goes:
            n_eff
            r_hat
            divergences
            tree depth
            E-BFMI
        For example, a warning code of 12 has a binary representation
        of 01100, which means that R-hat and divergences tests failed.
    return_dict : dict
        Returned if `return_dict` is True. A dictionary with the result
        of each diagnostic test.
    """
    warning_code = 0
    diag_results = {}

    pass_check, diag_results['neff'] = check_n_eff(
                        fit,
                        pars,
                        n_eff_rule_of_thumb=n_eff_rule_of_thumb,
                        quiet=quiet,
                        return_diagnostics=True)
    if not pass_check:
        warning_code = warning_code | (1 << 0)

    pass_check, diag_results['rhat'] = check_rhat(
                        fit,
                        pars,
                        rhat_rule_of_thumb=rhat_rule_of_thumb,
                        known_rhat_nans=known_rhat_nans,
                        quiet=quiet,
                        return_diagnostics=True)
    if not pass_check:
        warning_code = warning_code | (1 << 1)

    df = _fit_to_df(fit, pars=pars)

    pass_check, diag_results['n_divergences'] = check_divergences(
                        df,
                        quiet=quiet,
                        return_diagnostics=True)
    if not pass_check:
        warning_code = warning_code | (1 << 2)

    if max_treedepth == 'infer':
        max_treedepth = _infer_max_treedepth(fit)
    pass_check, diag_results['n_max_treedepth'] = check_treedepth(
                        df=df,
                        max_treedepth=max_treedepth,
                        quiet=quiet,
                        return_diagnostics=True)
    if not pass_check:
        warning_code = warning_code | (1 << 3)

    pass_check, diag_results['e_bfmi'] = check_energy(
                        df,
                        e_bfmi_rule_of_thumb=e_bfmi_rule_of_thumb,
                        quiet=quiet,
                        return_diagnostics=True)
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
            n_eff
            r_hat
            divergences
            tree depth
            E-BFMI
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
        raise RuntimeError('`quiet` is True and `return_dict` is False, '
            + 'so there is nothing to do.')

    passed_tests = dict(neff=True, rhat=True, divergence=True,
                        treedepth=True, energy=True)

    if warning_code & (1 << 0):
        passed_tests['neff'] = False
        if not quiet:
            print('n_eff / iteration warning')
    if warning_code & (1 << 1):
        passed_tests['rhat'] = False
        if not quiet:
            print('rhat warning')
    if warning_code & (1 << 2):
        passed_tests['divergence'] = False
        if not quiet:
            print('divergence warning')
    if warning_code & (1 << 3):
        passed_tests['treedepth'] = False
        if not quiet:
            print('treedepth warning')
    if warning_code & (1 << 4):
        passed_tests['energy'] = False
        if not quiet:
            print('energy warning')
    if warning_code == 0:
        if not quiet:
            print('No diagnostic warnings')

    if return_dict:
        return passed_tests


def posterior_predictive_ranking(fit=None, ppc_name=None, data=None,
                                 fractional=False):
    """Compute posterior predictive ranking of each data point.

    Parameters
    ----------
    fit : StanFit4Model instance
        A Stan fit of a model that has posterior predictive check values
        for the data.
    ppc_name : str
        The name of the parameter in the `fit` that has the posterior
        predictive checks.
    data : 1D Numpy array
        The data points that were observed.

    Returns
    -------
    output : 1D Numpy array, same shape as `data`
        The ranking among the posterior predictive samples of each data
        points.
    """
    if fit is None or ppc_name is None or data is None:
        raise RuntimeError(
                '`fit`, `ppc_name`, and `data` must all be specified.')

    ppc = fit.extract(ppc_name)[ppc_name]
    rankings = np.empty(ppc.shape[1])

    for i in range(ppc.shape[1]):
        rankings[i] = np.searchsorted(np.sort(ppc[:,i]), data[i])

    if fractional:
        rankings /= ppc.shape[0]

    return rankings


def sbc(prior_predictive_model=None,
        posterior_model=None,
        prior_predictive_model_data=None,
        posterior_model_data=None,
        measured_data=None,
        parameters=None,
        measured_data_dtypes={},
        chains=4,
        warmup=1000,
        iter=2000,
        thin=10,
        init=None,
        control=None,
        n_jobs=1,
        N=400,
        n_prior_draws_for_sd=1000,
        progress_bar=False):
    """Perform simulation-based calibration on a Stan Model.

    Parameters
    ----------
    prior_predictive_model : pystan.model.StanModel
        A Stan model for generating prior predictive data sets.
    posterior_model : pystan.model.StanModel
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
    parameters : list
        A list of strings containing parameter names to be considered
        in the SBC analysis. Not all parameters of the model need be
        considered; only those in `parameters` have rank statistics
        calculated.
    measured_data_dtypes : dict, default {}
        The key in the dtypes dict is a string representing the date
        name, and the corresponding item is its dtype, almost always
        either `int` or `float`.
    chains : int
        Number of chains to use in each simulation.
    warmup : int, default 1000
        Number of posterior samples in each simulation to be considered
        as warmup.
    iter : int, default 2000
        Number of posterior samples, including warmup, to draw in each
        simuation.
    init : {0, '0', 'random', function returning dict, list of dict}
        Specification of initialization in MCMC sampling, as defined
        in the PyStan docs for StanModel.sampling().
    control : dict, default None
        Specification of control parameters for MCMC sampler, as defined
        in the PyStan docs for StanModel.sampling().
    thin : int, default 10
        Thinning parameter for outputted samples.
    n_jobs : int, default 1
        Number of cores to use in the calculation.
    N : int, 400
        Number of simulations to run.
    n_prior_draws_for_sd : int, default 1000
        Number of prior draws to compute the prior standard deviation
        for a parameter in the prior distribution. This standard
        deviation is used in the shrinkage calculation.
    progress_bar : bool, default False
        If True, display a progress bar for the calculation using tqdm.

    Returns
    -------
    output : Pandas DataFrame
        A Pandas DataFrame with the output of the SBC analysis. It has
        the following columns.
        - trial : Unique trial number for the simulation.
        - warning_code : Warning code based on diagnostic checks
            outputted by `check_all_diagnostics()`.
        - parameter: The name of the scalar parameter.
        - prior: Value of the parameter used in the simulation. This
            value was drawn out of the prior distribution.
        - mean : mean parameter value based on sampling out of the
            posterior in the simulation.
        - sd : standard deviation of the parameter value based on
            sampling out of the posterior in the simulation.
        - L : The number of bins used in computing the rank statistic.
            The rank statistic should be uniform on the integers [0, L].
        - rank_statistic : Value of the rank statistic for the parameter
            for the trial.
        - shrinkage : The shrinkage for the parameter for the given
            trial. This is computed as 1 - sd / sd_prior, where sd_prior
            is the standard deviation of the parameters as determined
            from drawing out of the prior.
        - z_score : The z-score for the parameter for the given trial.
            This is computed as |mean - prior| / sd.

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

    if prior_predictive_model is None:
        raise RuntimeError('`prior_predictive_model` must be specified.')
    if posterior_model is None:
        raise RuntimeError('`posterior_model` must be specified.')
    if prior_predictive_model_data is None:
        raise RuntimeError('`prior_predictive_model_data` must be specified.')
    if posterior_model_data is None:
        raise RuntimeError('`posterior_model_data` must be specified.')
    if measured_data is None:
        raise RuntimeError('`measured_data` must be specified.')
    if parameters is None:
        raise RuntimeError('`parameters` must be specified.')

    # Take a prior sample to infer data types
    prior_sample = prior_predictive_model.sampling(
        data=prior_predictive_model_data,
        algorithm='Fixed_param',
        iter=1,
        chains=1,
        warmup=0)

    # Infer dtypes of measured data
    for data in measured_data:
        ar = prior_sample.extract(data)[data]
        if data not in measured_data_dtypes:
            if np.sum(ar != ar.astype(int)) == 0:
                warnings.warn(f'Inferring int dtype for {data}.')
                measured_data_dtypes[data] = int

    # Determine prior SDs for parameters of interest
    prior_sd = _get_prior_sds(prior_predictive_model,
                              prior_predictive_model_data,
                              parameters,
                              n_prior_draws_for_sd)

    def arg_input_generator():
        counter = 0
        while counter < N:
            counter += 1
            yield (prior_predictive_model,
                   posterior_model,
                   prior_predictive_model_data,
                   posterior_model_data,
                   measured_data,
                   parameters,
                   chains,
                   warmup,
                   iter,
                   thin,
                   init,
                   control,
                   prior_sd,
                   measured_data_dtypes)

    with multiprocessing.Pool(n_jobs) as pool:
        if progress_bar == 'notebook':
            output = list(tqdm.tqdm_notebook(pool.imap(_perform_sbc,
                                             arg_input_generator()),
                                             total=N))
        elif progress_bar == True:
            output = list(tqdm.tqdm(pool.imap(_perform_sbc,
                                    arg_input_generator()),
                                    total=N))
        elif progress_bar == False:
            output = pool.map(_perform_sbc, arg_input_generator())
        else:
            raise RuntimeError('Invalid `progress_bar`.')

    output = pd.DataFrame(output)
    output['L'] = (iter - warmup) * chains // thin

    return _tidy_sbc_output(output)


def hpd(x, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    x : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(x))

    # Number of total samples taken
    n = len(x)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])


def _fit_to_df(fit, **kwargs):
    """Convert StanFit4Model to data frame."""
    if 'StanFit4Model' in str(type(fit)):
        df = to_dataframe(fit, inc_warmup=False, **kwargs)
    elif type(fit) != pd.core.frame.DataFrame:
        raise RuntimeError('`fit` must be a StanModel or Pandas data frame.')
    else:
        df = fit.loc[fit['warmup']==0, :]

    return df


def _infer_max_treedepth(fit):
    """Extract max treedepth from a StanFit4Model instance."""
    try:
        return fit.stan_args[0]['ctrl']['sampling']['max_treedepth']
    except:
        warnings.warn('Unable to determine max_treedepth. '
                    + 'Using value of 10.')
        return 10


def _ebfmi(energy):
    """Compute energy-Bayes fraction of missing information"""
    return np.sum(np.diff(energy)**2) / (len(energy) - 1) / np.var(energy)


def _perform_sbc(args):
    """Perform an SBC analysis"""
    logging.getLogger('pystan').setLevel(logging.CRITICAL)

    (prior_predictive_model,
     posterior_model,
     prior_predictive_model_data,
     posterior_model_data,
     measured_data,
     parameters,
     chains,
     warmup,
     iter,
     thin,
     init,
     control,
     prior_sd,
     measured_data_dtypes) = args

    posterior_model_data = copy.deepcopy(posterior_model_data)

    prior_sample = prior_predictive_model.sampling(
        data=prior_predictive_model_data,
        algorithm='Fixed_param',
        iter=1,
        chains=1,
        warmup=0)

    # Extract data generated from the prior predictive calculation
    for data in measured_data:
        ar = prior_sample.extract(data)[data]
        if len(ar.shape) == 0:
            posterior_model_data[data] = np.asscalar(
                                ar.astype(measured_data_dtypes[data]))
        else:
            posterior_model_data[data] = ar[0].astype(
                                            measured_data_dtypes[data])

    # Store what the parameters were to generate prior predictive data
    param_priors = {param: float(prior_sample.extract(param)[param])
                            for param in parameters}

    # Generate posterior samples
    posterior_samples = posterior_model.sampling(
            data=posterior_model_data,
            iter=iter,
            chains=chains,
            warmup=warmup,
            n_jobs=1,
            thin=thin)

    # Extract summary
    summary = posterior_samples.summary(probs=[])
    row_names = tuple(summary['summary_rownames'])
    col_names = tuple(summary['summary_colnames'])

    # Omit Rhat calculations on parameters we are not interested in
    known_rhat_nans = list(set(row_names) - set(parameters))
    warning_code, diagnostics = check_all_diagnostics(
        posterior_samples, quiet=True, known_rhat_nans=known_rhat_nans,
        return_diagnostics=True)

    # Generate output dictionary
    output = {param+'_rank_statistic':
        (posterior_samples.extract(param)[param] < param_priors[param]).sum()
                for param in parameters}
    for param, p_prior in param_priors.items():
        output[param + '_ground_truth'] = p_prior

    # Compute posterior sensitivities
    for param in parameters:
        mean_i = col_names.index('mean')
        sd_i = col_names.index('sd')
        param_i = row_names.index(param)
        output[param+'_mean'] = summary['summary'][param_i, mean_i]
        output[param+'_sd'] = summary['summary'][param_i, sd_i]
        output[param+'_z_score'] = (
                (output[param+'_mean'] - output[param + '_ground_truth'])
                / output[param+'_sd'])
        output[param+'_shrinkage'] = (1 -
                (output[param+'_sd'] / prior_sd[param])**2)
        output[param+'_n_eff/n_iter'] = (
            diagnostics['neff'].loc[diagnostics['neff']['parameter']==param,
                                    'n_eff/n_iter'].values[0])
        output[param+'_rhat'] = (
            diagnostics['rhat'].loc[diagnostics['rhat']['parameter']==param,
                                    'Rhat'].values[0])

    output['n_bad_ebfmi'] = diagnostics['e_bfmi']['problematic'].sum()
    output['n_divergences'] = int(diagnostics['n_divergences'])
    output['n_max_treedepth'] = int(diagnostics['n_max_treedepth'])
    output['warning_code'] = warning_code

    return output


def _get_prior_sds(prior_predictive_model,
               prior_predictive_model_data,
               parameters,
               n_prior_draws_for_sd):
    """Compute standard deviations of prior parameters."""

    prior_samples = prior_predictive_model.sampling(
        data=prior_predictive_model_data,
        algorithm='Fixed_param',
        iter=n_prior_draws_for_sd,
        chains=1,
        warmup=0)

    # Make sure only scalar parameters are being checked
    for param in parameters:
        if param not in prior_samples.model_pars:
            raise RuntimeError(f'Parameter {param} not in the model.')
        ind = prior_samples.model_pars.index(param)
        if prior_samples.par_dims[ind] != []:
            err = """Can only perform SBC checks on scalar parameters.
Parameter {} is not a scalar. If you want to check elements
of this parameter, use an entry in the `generated quantities`
block to store the element as a scalar.""".format(param)
            raise RuntimeError(err)

    # Compute prior sd's
    prior_sd = {}
    summary = prior_samples.summary(probs=[])
    sd_i = tuple(summary['summary_colnames']).index('sd')
    for param in parameters:
        param_i = tuple(summary['summary_rownames']).index(param)
        prior_sd[param] = summary['summary'][param_i, sd_i]

    return prior_sd


def _tidy_sbc_output(sbc_output):
    """Tidy output from sbc().

    Returns
    -------
    output : DataFrame
        Tidy data frame with SBC results.
    """
    df = sbc_output.copy()
    df['trial'] = df.index.values

    rank_stat_cols = list(
                df.columns[df.columns.str.contains('_rank_statistic')])
    params = [col[:col.rfind('_rank_statistic')] for col in rank_stat_cols]

    dfs = []
    stats = ['ground_truth', 'rank_statistic', 'mean', 'sd', 'shrinkage',
             'z_score', 'rhat', 'n_eff/n_iter']

    aux_cols = ['n_divergences', 'n_bad_ebfmi', 'n_max_treedepth',
                'warning_code', 'L', 'trial']
    for param in params:
        cols = [param+'_'+stat for stat in stats]
        sub_df = df[cols+aux_cols].rename(columns={old_col: new_col for old_col, new_col in zip(cols, stats)})
        sub_df['parameter'] = param
        dfs.append(sub_df)

    return pd.concat(dfs, ignore_index=True)


@contextlib.contextmanager
def _disable_logging(level=logging.CRITICAL):
    """Context manager for disabling logging."""
    previous_level = logging.root.manager.disable

    logging.disable(level)

    try:
        yield
    finally:
        logging.disable(previous_level)

