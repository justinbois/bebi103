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

def normal_rng(mu, sigma, size=None):

    model_code = """
data {
  real mu;
  real sigma;
}

generated quantities {
  real output = normal_rng(mu, sigma);
}
    """
    if size is None:
        size = 1

    data = dict(mu=mu, sigma=sigma)
    sm = StanModel(model_code=model_code, model_name='normal_rng', quiet=True)
    fit = sm.sampling(data=data,
                      algorithm='Fixed_param', 
                      iter=size, 
                      chains=1,
                      warmup=0)
    output = fit.extract('output')['output']
    if size == 1:
        return float(output)
    else:
        return output.flatten()

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


def stan_rng(dist, *params, size=1):
    """Draw random samples from a distribution using Stan."""

    stan_int_rng_code = """
data {
  real real_param_1;
  real real_param_2;
  real real_param_3;
  int int_param_1;
  int int_param_2;
  int int_param_3;

  int n;

  int section;
  int subsection;
}

generated quantities {
  int output[n];

  // Bernoulli
  if (section == 50 && subsection == 1) {
    for (i in 1:n) {
      output[i] = bernoulli_rng(real_param_1);
    }
  }

  // Bernoulli logit
  else if (section == 50 && subsection == 2) {
    for (i in 1:n) {
      output[i] = bernoulli_logit_rng(real_param_1);
    }    
  }

}
"""

    stan_real_rng_code = """
data {
  real real_param_1;
  real real_param_2;
  real real_param_3;
  int int_param_1;
  int int_param_2;
  int int_param_3;

  int n;

  int section;
  int subsection;
}

generated quantities {
  real output[n];

  // Normal
  if (section == 54 && subsection == 1) {
    for (i in 1:n) {
      output[i] = normal_rng(real_param_1, real_param_2);
    }    
  }

}
"""

    dist_dict = {'bernoulli': {'section': 50, 
                               'subsection': 1, 
                               'param_types': [float], 
                               'param_bounds': [[0, 1]],
                               'output_type': int},
                 'bernoulli_logit': {'section': 50, 
                                     'subsection': 2,
                                     'param_types': [float],
                                     'param_bounds': [[-np.inf, np.inf]],
                                     'output_type': int},
                 'binomial': {'section': 51, 
                              'subsection': 1,
                              'param_types': [int, float],
                              'param_bounds': [[0, np.inf],
                                               [0, 1]],
                              'output_type': int},
                 'normal': {'section': 54, 
                            'subsection': 1,
                            'param_types': [float, float],
                            'param_bounds': [[-np.inf, np.inf],
                                             [0, np.inf]],
                            'output_type': float},
                }

    if dist not in dist_dict:
        raise RuntimeError('Distribution not found.')

    d = dist_dict[dist]

    if len(params) != len(d['param_types']):
        raise RuntimeError('Incompatible number of parameters.')

    param_iterator = zip(params,
                         d['param_types'],
                         d['param_bounds'])
    int_params = [0, 0, 0]
    real_params = [0.0, 0.0, 0.0]
    for i, (param, param_type, param_bounds) in enumerate(param_iterator):
        if type(param) not in [int, float]:
            raise RuntimeError('All inputted params must be `int` or `float`')
        if type(param) == float and param_type == int:
            raise RuntimeError(f'param {param} should be an `int`.')
        if not (param_bounds[0] <= param <= param_bounds[1]):
            raise RuntimeError(f'param {param} out of bounds.')
        if param_type == int:
            int_params[i] = param
        else:
            real_params[i] = float(param)

    data = data = {'real_param_1': real_params[0], 
                   'real_param_2': real_params[1], 
                   'real_param_3': real_params[1], 
                   'int_param_1': int_params[0], 
                   'int_param_2': int_params[1],
                   'int_param_3': int_params[2], 
                   'n': size, 
                   'section': d['section'], 
                   'subsection': d['subsection']}

    if d['output_type'] == int:
        sm = StanModel(model_code=stan_int_rng_code, 
                       model_name='int_rng_model',
                       quiet=True)
    else:
        sm = StanModel(model_code=stan_real_rng_code, 
                       model_name='real_rng_model',
                       quiet=True)

    fit = sm.sampling(data=data, 
                      algorithm='Fixed_param', 
                      chains=1, 
                      iter=1, 
                      warmup=0)

    samples = fit.extract('output')['output'].flatten()
    if d['output_type'] == int:
        return samples.astype(int)
    return samples


def extract_array(df, name):
    ind_names = 'ijklmn'

    regex_name = name
    for char in '[\^$.|?*+(){}':
        regex_name = regex_name.replace(char, '\\'+char)

    # Extract columns that match the name
    sub_df = df.filter(regex=regex_name+'\[(\d+)(,\d+)*\]')

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


def plot_predictive_ecdf(df, name, plot_width=350, plot_height=200,
                         x_axis_label=None, y_axis_label='ECDF',
                         color='blue', x=None, discrete=False):
    if 'StanFit4Model' in str(type(df)):
        df = df.to_dataframe(diagnostics=False)

    if color not in ['green', 'blue', 'red', 'gray', 
                     'purple', 'orange', 'betancourt']:
        raise RuntimeError("Only allowed colors are 'green', 'blue', 'red', 'gray', 'purple', 'orange'")

    if x_axis_label is None:
        x_axis_label=str(name)

    sub_df = extract_array(df, name)

    if 'index_j' in sub_df:
        raise RuntimeError('Can only plot ECDF for one-dimensional data.')

    colors = {'blue': ['#9ecae1','#6baed6','#4292c6','#2171b5','#084594'],
              'green': ['#a1d99b','#74c476','#41ab5d','#238b45','#005a32'],
              'red': ['#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d'],
              'orange': ['#fdae6b','#fd8d3c','#f16913','#d94801','#8c2d04'],
              'purple': ['#bcbddc','#9e9ac8','#807dba','#6a51a3','#4a1486'],
              'gray': ['#bdbdbd','#969696','#737373','#525252','#252525'],
              'betancourt': ['#DCBCBC', '#C79999', '#B97C7C', 
                             '#A25050', '#8F2727', '#7C0000']}

    data_range = sub_df[name].max() - sub_df[name].min()
    if x is None:
      x = np.linspace(sub_df[name].min() - 0.05*data_range,
                      sub_df[name].max() + 0.05*data_range,
                      400)

    df_ecdf = pd.DataFrame()
    df_ecdf_vals = pd.DataFrame()
    grouped = sub_df.groupby(['chain', 'chain_idx'])
    for i, g in grouped:
        df_ecdf_vals[i] = _ecdf_arbitrary_points(g[name], x)
        
    for ptile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        df_ecdf[ptile] = df_ecdf_vals.quantile(
                            ptile/100, axis=1, interpolation='higher')
    df_ecdf['x'] = x

    p = bokeh.plotting.figure(plot_width=plot_width, 
                              plot_height=plot_height,
                              x_axis_label=x_axis_label, 
                              y_axis_label=y_axis_label)

    for i, ptile in enumerate([10, 20, 30, 40]):
        viz.fill_between(df_ecdf['x'], df_ecdf[ptile],
                         df_ecdf['x'], df_ecdf[100-ptile],
                         p=p,
                         show_line=False,
                         fill_color=colors[color][i])

    p.line(df_ecdf['x'],
           df_ecdf[50], 
           line_width=2, 
           color=colors[color][-1])

    return p


def _ecdf_arbitrary_points(data, x):
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side='right')]


def _fit_to_df(fit):
    """Convert StanFit4Model to data frame."""
    if 'StanFit4Model' in str(type(fit)):
        df = fit.to_dataframe(inc_warmup=False)
    elif type(fit) != pd.core.frame.DataFrame:
        raise RuntimeError('`fit` must be a StanModel or Pandas data frame.')
    else:
        df = fit.loc[fit['warmup']==0, :]

    return df


def check_div(fit, quiet=False, return_diagnostics=False):
    """Check transitions that ended with a divergence"""

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


def _ebfmi(energy):
    return np.sum(np.diff(energy)**2) / (len(energy) - 1) / np.var(energy)


def check_energy(fit, quiet=False, e_bfmi_rule_of_thumb=0.2, 
                 return_diagnostics=False):
    """Checks the energy fraction of missing information (E-FMI)"""
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
                print('Chain {}: E-BFMI = {}'.format(r['chain'], r['E=BFMI']))
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

    if not check_div(df, quiet=quiet):
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
        print("n_eff / iteration warning")
    if warning_code & (1 << 1):
        print("rhat warning")
    if warning_code & (1 << 2):
        print("divergence warning")
    if warning_code & (1 << 3):
        print("treedepth warning")
    if warning_code & (1 << 4):
        print("energy warning")
    if warning_code == 0:
        print("No diagnostic warnings")


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

    return output


@numba.jit(nopython=True)
def _y_ecdf(data, x):
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side='right')]

@numba.jit(nopython=True)
def _draw_ecdf_bootstrap(a, b, n, n_bs_reps=100000):
    x = np.arange(L+1)
    ys = np.empty((n_bs_reps, len(x)))
    for i in range(n_bs_reps):
        draws = np.random.randint(0, L+1, size=n)
        ys[i, :] = _y_ecdf(draws, x)
    return ys



def _sbc_envelope(L, n, ptile=95, diff=True, bootstrap=False, n_bs_reps=None):
    x = np.arange(L+1)
    y = st.randint.cdf(x, 0, L+1)
    std = np.sqrt(y * (1 - y) / n)
        
    if bootstrap:
        if n_bs_reps is None:
            n_bs_reps = int(max(n, max(L+1, 100/(100-ptile))) * 100)
        ys = draw_ecdf_bootstrap(a, b, n, n_bs_reps=n_bs_reps)
        y_low, y_high = np.percentile(ys, 
                                      [50 - ptile/2, 50 + ptile/2], 
                                      axis=0)
    else:
        y_low = np.concatenate(
            (st.norm.ppf((50 - ptile/2)/100, y[:-1], std[:-1]), (1.0,)))
        y_high = np.concatenate(
            (st.norm.ppf((50 + ptile/2)/100, y[:-1], std[:-1]), (1.0,)))
        
    # Ensure that ends are appropriate
    y_low = np.maximum(0, y_low)
    y_high = np.minimum(1, y_high)
    
    # Make "formal" stepped ECDFs
    _, y_low = viz._to_formal(x, y_low)
    x_formal, y_high = viz._to_formal(x, y_high)
        
    if diff:
        _, y = viz._to_formal(x, y)
        y_low -= y
        y_high -= y
        
    return x_formal, y_low, y_high


def _ecdf_diff(data, formal=False):
    x, y = viz._ecdf_vals(data)
    y_uniform = (x + 1)/len(x)
    if formal:
        x, y = viz._to_formal(x, y)
        _, y_uniform = viz._to_formal(np.arange(len(data)), y_uniform)
    y -= y_uniform

    return x, y


def sbc_plot(df, param, diff=True, formal=False):
    L = df['L'].iloc[0]
    x, y_low, y_high = _sbc_envelope(L, len(df), ptile=99, diff=diff, bootstrap=False, n_bs_reps=100000)

    if diff:
        x_data, y_data = _ecdf_diff(df[param+'_rank_statistic'], formal=formal)
    else:
        x_data, y_data = viz._ecdf_vals(df[param+'rank_statistic'])

    p = viz.fill_between(x1=x, x2=x, y1=y_high, y2=y_low, fill_color='gray', 
                         fill_alpha=0.5, show_line=True, line_color='gray')
    if formal:
        p.line(x_data, y_data)
    else:
        p.circle(x_data, y_data)

    return p
    

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
