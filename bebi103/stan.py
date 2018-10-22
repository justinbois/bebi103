import copy
import itertools
import re
import pickle
import hashlib
import logging
import multiprocessing

import tqdm

import numpy as np
import pandas as pd
import numba
import scipy.stats as st

import pystan

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

    if pystan.__version__ >= '2.18':
        return fit.to_dataframe(pars=pars, 
                                permuted=permuted, 
                                dtypes=dtypes, 
                                inc_warmup=inc_warmup, 
                                diagnostics=diagnostics)

    # Diagnostics to pull out
    diags = ['divergent__', 'energy__', 'treedepth__', 'accept_stat__', 
             'stepsize__', 'n_leapfrog__']
        
    # Build parameters if not supplied
    if pars is None:
        pars = tuple(fit.model_pars + ['lp__'])
    if isinstance(pars, str):
        pars = tuple([pars, 'lp__'])

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
        for k, par in enumerate(fit.flatnames):
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
    df = _fit_to_df(samples)

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


def df_to_datadict_hier(df=None, level_cols=None, data_cols=None, 
                        cowardly=True):
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
    cowardly : bool, default True
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
        df[str(col)+'_stan'] = df.groupby(level_cols[:col_ind+1]).ngroup() + 1
    
    new_df = df.sort_values(by=level_cols_stan)
    
    data = dict()
    data['N'] = len(new_df)
    for i, col in enumerate(level_cols_stan):
        data['J_'+ str(i+1)] = len(new_df[col].unique())    
    for i, _ in enumerate(level_cols_stan[1:]):
        data['index_' + str(i+1)] = np.array([key[i] for key in new_df.groupby(level_cols_stan[:i+2]).groups]).astype(int)
    data['index_' + str(len(level_cols_stan))] = new_df[level_cols_stan[-1]].values.astype(int)
    for col in data_cols:
        data[str(col)] = new_df[col].values
        
    return data, df


def check_divergences(fit, quiet=False, return_diagnostics=False):
    """Check transitions that ended with a divergence.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
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

    df = _fit_to_df(fit)

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


def check_treedepth(fit, max_treedepth=10, quiet=False,
                    return_diagnostics=False):
    """Check transitions that ended prematurely due to maximum tree depth limit.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    max_treedepth : int, default 10
        The maximum allowed tree depth.
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
    df = _fit_to_df(fit)

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


def check_energy(fit, quiet=False, e_bfmi_rule_of_thumb=0.2, 
                 return_diagnostics=False):
    """Checks the energy-Bayes fraction of missing information (E-BFMI)

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
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
    df = _fit_to_df(fit)

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


def check_n_eff(fit, quiet=False, n_eff_rule_of_thumb=0.001,
                return_diagnostics=False):
    """Checks the effective sample size per iteration.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
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

    fit_summary = fit.summary(probs=[])
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


def check_rhat(fit, quiet=False, rhat_rule_of_thumb=1.1, known_rhat_nans=[], 
               return_diagnostics=False):
    """Checks the potential issues with scale reduction factors.

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
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

    fit_summary = fit.summary(probs=[])
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
            print('Rhat looks reasonable for all parameters')

    if return_diagnostics:
        return pass_check, pd.DataFrame(data={'parameter': names, 
                                              'Rhat': rhat})
    return pass_check


def check_all_diagnostics(fit, max_treedepth=10, e_bfmi_rule_of_thumb=0.2,
                          n_eff_rule_of_thumb=0.001, rhat_rule_of_thumb=1.1,
                          known_rhat_nans=[], quiet=False):
    """Checks all MCMC diagnostics

    Parameters
    ----------
    fit : StanFit4Model instance
        Fit for which diagnostic is to be run.
    max_treedepth : int, default 10
        The maximum allowed tree depth.
    e_bfmi_rule_of_thumb : float, default 0.2
        Rule of thumb value for E-BFMI. If below this value, there may
        be cause for concern.
    rhat_rule_of_thumb : float, default 1.1
        Rule of thumb value for maximum allowed R-hat.
    known_rhat_nans : list, default []
        List of parameter names which are known to have R-hat be NaN.
        These are typically parameters that are deterministic. 
        Parameters in this list are ignored.
    quiet : bool, default False
        If True, do no print diagnostic result to the screen.

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
    """
    warning_code = 0

    if not check_n_eff(fit, 
                       n_eff_rule_of_thumb=n_eff_rule_of_thumb, 
                       quiet=quiet):
        warning_code = warning_code | (1 << 0)

    if not check_rhat(fit, 
                      rhat_rule_of_thumb=rhat_rule_of_thumb,
                      known_rhat_nans=known_rhat_nans,
                      quiet=quiet):
        warning_code = warning_code | (1 << 1)

    df = _fit_to_df(fit)

    if not check_divergences(df, quiet=quiet):
        warning_code = warning_code | (1 << 2)

    if not check_treedepth(df, 
                           max_treedepth=max_treedepth, 
                           quiet=quiet):
        warning_code = warning_code | (1 << 3)

    if not check_energy(df,
                        e_bfmi_rule_of_thumb=e_bfmi_rule_of_thumb, 
                        quiet=quiet):
        warning_code = warning_code | (1 << 4)

    return warning_code


def parse_warning_code(warning_code):
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

    Returns
    -------
    None
        Results are printed to the screen.
    """
    if warning_code & (1 << 0):
        print('n_eff / iteration warning')
    if warning_code & (1 << 1):
        print('rhat warning')
    if warning_code & (1 << 2):
        print('divergence warning')
    if warning_code & (1 << 3):
        print('treedepth warning')
    if warning_code & (1 << 4):
        print('energy warning')
    if warning_code == 0:
        print('No diagnostic warnings')


def sbc(prior_predictive_model=None,
        posterior_model=None, 
        prior_predictive_model_data=None,
        posterior_model_data=None,
        measured_data=None,
        parameters=None,
        chains=4,
        warmup=1000,
        iter=2000,
        thin=10,
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
    chains : int
        Number of chains to use in each simulation.
    warmup : int, default 1000
        Number of posterior samples in each simulation to be considered
        as warmup.
    iter : int, default 2000
        Number of posterior samples, including warmup, to draw in each
        simuation.
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
                   prior_sd)
    
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
     prior_sd) = args

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
            posterior_model_data[data] = np.asscalar(ar)
        else:
            posterior_model_data[data] = ar[0]

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
    warning_code = check_all_diagnostics(posterior_samples, 
                                         quiet=True,
                                         known_rhat_nans=known_rhat_nans)

    # Generate output dictionary
    output = {param+'_rank_statistic': 
        (posterior_samples.extract(param)[param] < param_priors[param]).sum()
                for param in parameters}
    for param, p_prior in param_priors.items():
        output[param + '_prior'] = p_prior

    # Compute posterior sensitivities
    for param in parameters:
        mean_i = col_names.index('mean')
        sd_i = col_names.index('sd')
        param_i = row_names.index(param)
        output[param+'_mean'] = summary['summary'][param_i, mean_i]
        output[param+'_sd'] = summary['summary'][param_i, sd_i]
        output[param+'_z_score'] = np.abs(
                (output[param+'_mean'] - output[param + '_prior']) 
                / output[param+'_sd'])
        output[param+'_shrinkage'] = (1 - 
                (output[param+'_sd'] / prior_sd[param])**2)

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
    stats = ['prior', 'rank_statistic', 'mean', 'sd', 'shrinkage', 'z_score']
    aux_cols = ['warning_code', 'L', 'trial']
    for param in params:
        cols = [param+'_'+stat for stat in stats]
        sub_df = df[cols+aux_cols].rename(columns={old_col: new_col for old_col, new_col in zip(cols, stats)})
        sub_df['parameter'] = param
        dfs.append(sub_df)

    return pd.concat(dfs, ignore_index=True)
    