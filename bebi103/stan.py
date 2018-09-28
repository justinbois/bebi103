import copy
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


def StanModel(file=None, model_name='anon_model', model_code=None,
              charset='utf-8', force_compile=False, quiet=False, **kwargs):
    """"
    Utility to load cached Stan model. Use exactly as pystan.StanModel.
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
            if not quiet:
                print("Using cached StanModel.")

    return sm


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
            ['chain', 'chain_idx', 'warmup', 'index_i', 'name'] 
        For a 2D array, there is an additional column named 'index_j'. A
        3D array has a column 'index_k' and so on through 'index_l',
        'index_m', and 'index_n'.
    """
    df = _fit_to_df(samples)

    ind_names = 'ijklmn'

    regex_name = name
    for char in '[\^$.|?*+(){}':
        regex_name = regex_name.replace(char, '\\'+char)

    # Extract columns that match the name
    sub_df = df.filter(regex=regex_name+'\[(\d+)(,\d+)*\]')

    if len(sub_df.columns) == 0:
        raise RuntimeError(
                "column '{}' is either absent or scalar-valued.".format(name))


    # Set up a multiindex
    multiindex = [None for _ in range(len(sub_df.columns))]
    for i, c_str in enumerate(sub_df.columns):
        c = c_str[c_str.rfind('[')+1:-1]
        if ',' in c:
            c = c.split(',')
            c = (int(char) for char in c)
            multiindex[i] = (c_str[:c_str.rfind('[')], *c)
        else:
            multiindex[i] = (c_str[:c_str.rfind('[')], int(c))

    # Names for Multiindex
    n_dim = len(multiindex[0]) - 1
    if n_dim > 6:
        raise RuntimeError('Can only have maximally six dimensions in array.')
    names = (name,) + tuple('index_'+ind_names[i] for i in range(n_dim))

    # Rename columns with Multiindex
    sub_df.columns = pd.MultiIndex.from_tuples(multiindex, names=names)

    # Stack
    for _ in range(n_dim):
        sub_df = sub_df.stack()
    sub_df = sub_df.reset_index()

    # Add in chain, chain_idx, and warmup
    sub_df['warmup'] = df.loc[sub_df['level_0'], 'warmup'].values
    sub_df['chain'] = df.loc[sub_df['level_0'], 'chain'].values
    sub_df['chain_idx'] = df.loc[sub_df['level_0'], 'chain_idx'].values

    # Clean out level 0
    del sub_df['level_0']

    # No need to have column headings named
    sub_df.columns.name = None

    return sub_df


def check_divergences(fit, quiet=False, return_diagnostics=False):
    """Check transitions that ended with a divergence.
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
        print('  Try running with larger adapt_delta to remove divergences')

    if return_diagnostics:
        return pass_check, n_divergent
    return pass_check


def check_treedepth(fit, max_treedepth=10, quiet=False,
                    return_diagnostics=False):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
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
               + ' to avoid saturation')

    if return_diagnostics:
        return pass_check, n_too_deep
    return pass_check


def check_energy(fit, quiet=False, e_bfmi_rule_of_thumb=0.2, 
                 return_diagnostics=False):
    """Checks the energy-Bayes fraction of missing information (E-BFMI)"""
    df = _fit_to_df(fit)

    result = (df.groupby('chain')['energy__']
                .agg(_ebfmi)
                .reset_index()
                .rename(columns={'energy__': 'E-BFMI'}))
    result['problematic'] = result['E-BFMI'] < e_bfmi_rule_of_thumb

    pass_check = (~result['problematic']).all()

    if not quiet:
        if pass_check:
            print('E-BFMI indicated no pathological behavior')
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
                  + ' sample size has likely been overestimated')
        else:
            print('n_eff / iter looks reasonable for all parameters')

    if return_diagnostics:
        return pass_check, pd.DataFrame(data={'parameter': names, 
                                              'n_eff/n_iter': ratio})
    return pass_check


def check_rhat(fit, quiet=False, rhat_rule_of_thumb=1.1, known_rhat_nans=[], 
               return_diagnostics=False):
    """Checks the potential scale reduction factors"""
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
    """Checks all MCMC diagnostics"""
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
    """Parses warning code into individual failures"""
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
        df = fit.to_dataframe(inc_warmup=False, **kwargs)
    elif type(fit) != pd.core.frame.DataFrame:
        raise RuntimeError('`fit` must be a StanModel or Pandas data frame.')
    else:
        df = fit.loc[fit['warmup']==0, :]

    return df


def _ebfmi(energy):
    return np.sum(np.diff(energy)**2) / (len(energy) - 1) / np.var(energy)


def _perform_sbc(args):

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
    

def to_dataframe(fit, pars=None, permuted=False, dtypes=None, 
                 inc_warmup=False, diagnostics=True):
    """Convert output of a Stan calculation to a Pandas dataframe."""
    raise NotImplementedError('Dataframe conversion not yet implemented. If you are using PyStan version >= 2.18, use the `.dataframe()` method.')