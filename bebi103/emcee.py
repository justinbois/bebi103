import collections
import warnings

import numpy as np
import pandas as pd

import emcee

def generic_log_posterior(log_prior, log_likelihood, params, logpargs=(),
                          loglargs=()):
    """
    Generic log posterior for MCMC calculations

    Parameters
    ----------
    log_prior : function
        Function to compute the log prior.
        Call signature: log_prior(params, *logpargs)
    log_likelihood : function
        Function to compute the log prior.
        Call signature: log_likelhood(params, *loglargs)
    params : ndarray
        Numpy array containing the parameters of the posterior.
    logpargs : tuple, default ()
        Tuple of parameters to be passed to log_prior.
    loglargs : tuple, default ()
        Tuple of parameters to be passed to log_likelihood.

    Returns
    -------
    output : float
        The logarithm of the posterior evaluated at `params`.
    """
    # Compute log prior
    lp = log_prior(params, *logpargs)

    # If log prior is -inf, return that
    if lp == -np.inf:
        return -np.inf

    # Compute and return posterior
    return lp + log_likelihood(params, *loglargs)


def sampler_to_dataframe(sampler, columns=None):
    """
    Convert output of an emcee sampler to a Pandas DataFrame.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler or emcee.PTSampler instance
        Sampler instance form which MCMC has already been run.

    Returns
    -------
    output : DataFrame
        Pandas DataFrame containing the samples. Each column is
        a variable, except: 'lnprob' and 'chain' for an
        EnsembleSampler, and 'lnlike', 'lnprob', 'beta_ind',
        'beta', and 'chain' for a PTSampler. These contain obvious
        values.
    """
    invalid_column_names = ['lnprob', 'chain', 'lnlike', 'beta',
                            'beta_ind']
    if np.any([x in columns for x in invalid_column_names]):
            raise RuntimeError('You cannot name columns with any of these: '
                                    + '  '.join(invalid_column_names))

    if columns is None:
        columns = list(range(sampler.chain.shape[-1]))

    if isinstance(sampler, emcee.EnsembleSampler):
        n_walkers, n_steps, n_dim = sampler.chain.shape

        df = pd.DataFrame(data=sampler.flatchain, columns=columns)
        df['lnprob'] = sampler.flatlnprobability
        df['chain'] = np.concatenate([i * np.ones(n_steps, dtype=int)
                                                for i in range(n_walkers)])
    elif isinstance(sampler, emcee.PTSampler):
        n_temps, n_walkers, n_steps, n_dim = sampler.chain.shape

        df = pd.DataFrame(
            data=sampler.flatchain.reshape(
                (n_temps * n_walkers * n_steps, n_dim)),
            columns=columns)
        df['lnlike'] = sampler.lnlikelihood.flatten()
        df['lnprob'] = sampler.lnprobability.flatten()

        beta_inds = [i * np.ones(n_steps * n_walkers, dtype=int)
                     for i, _ in enumerate(sampler.betas)]
        df['beta_ind'] = np.concatenate(beta_inds)

        df['beta'] = sampler.betas[df['beta_ind']]

        chain_inds = [j * np.ones(n_steps, dtype=int)
                      for i, _ in enumerate(sampler.betas)
                      for j in range(n_walkers)]
        df['chain'] = np.concatenate(chain_inds)
    else:
        raise RuntimeError('Invalid sample input.')

    return df


def run_ensemble_emcee(log_post=None, n_burn=100, n_steps=100,
                       n_walkers=None, p_dict=None, p0=None, columns=None,
                       args=(), threads=None, thin=1, return_sampler=False,
                       return_pos=False):
    """
    Run emcee.

    Parameters
    ----------
    log_post : function
        The function that computes the log posterior.  Must be of
        the form log_post(p, *args), where p is a NumPy array of
        parameters that are sampled by the MCMC sampler.
    n_burn : int, default 100
        Number of burn steps
    n_steps : int, default 100
        Number of MCMC samples to take
    n_walkers : int
        Number of walkers, ignored if p0 is None
    p_dict : collections.OrderedDict
        Each entry is a tuple with the function used to generate
        starting points for the parameter and the arguments for
        the function.  The starting point function must have the
        call signature f(*args_for_function, n_walkers).  Ignored
        if p0 is not None.
    p0 : array
        n_walkers by n_dim array of initial starting values.
        p0[i,j] is the starting point for walk i along variable j.
        If provided, p_dict is ignored.
    columns : list of strings
        Name of parameters.  These will be the column headings in the
        returned DataFrame.  If None, either inferred from p_dict or
        assigned sequential integers.
    args : tuple
        Arguments passed to log_post
    threads : int
        Number of cores to use in calculation
    thin : int
        The number of iterations to perform between saving the
        state to the internal chain.
    return_sampler : bool, default False
        If True, return sampler as well as DataFrame with results.
    return_pos : bool, default False
        If True, additionally return position of the sampler.

    Returns
    -------
    df : pandas.DataFrame
        First columns give flattened MCMC chains, with columns
        named with the variable being sampled as a string.
        Other columns are:
          'chain':    ID of chain
          'lnprob':   Log posterior probability
    sampler : emcee.EnsembleSampler instance, optional
        The sampler instance.
    pos : ndarray, shape (nwalkers, ndim), optional
        Last position of the walkers.
    """

    if p0 is None and p_dict is None:
        raise RuntimeError('Must supply either p0 or p_dict.')

    # Infer n_dim and n_walkers (and check inputs)
    if p0 is None:
        if n_walkers is None:
            raise RuntimeError('n_walkers must be specified if p0 is None')

        if type(p_dict) is not collections.OrderedDict:
            raise RuntimeError('p_dict must be collections.OrderedDict.')

        n_dim = len(p_dict)
    else:
        n_walkers, n_dim = p0.shape
        if p_dict is not None:
            warnings.RuntimeWarning('p_dict is being ignored.')

    # Infer columns
    if columns is None:
        if p_dict is not None:
            columns = list(p_dict.keys())
        else:
            columns = list(range(n_dim))
    elif len(columns) != n_dim:
        raise RuntimeError('len(columns) must equal number of parameters.')

    # Check for invalid column names
    invalid_column_names = ['lnprob', 'chain', 'lnlike', 'beta',
                            'beta_ind']
    if np.any([x in columns for x in invalid_column_names]):
            raise RuntimeError('You cannot name columns with any of these: '
                                    + '  '.join(invalid_column_names))

    # Build starting points of walkers
    if p0 is None:
        p0 = np.empty((n_walkers, n_dim))
        for i, key in enumerate(p_dict):
            p0[:, i] = p_dict[key][0](*(p_dict[key][1] + (n_walkers,)))

    # Set up the EnsembleSampler instance
    if threads is not None:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                        args=args, threads=threads)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                        args=args)

    # Do burn-in
    if n_burn > 0:
        pos, _, _ = sampler.run_mcmc(p0, n_burn, storechain=False)
    else:
        pos = p0

    # Sample again, starting from end burn-in state
    pos, _, _ = sampler.run_mcmc(pos, n_steps, thin=thin)

    # Make DataFrame for results
    df = sampler_to_dataframe(sampler, columns=columns)

    # Set up return
    return_vals = (df, sampler, pos)
    return_bool = (True, return_sampler, return_pos)
    ret = tuple([rv for rv, rb in zip(return_vals, return_bool) if rb])
    if len(ret) == 1:
        return ret[0]
    return ret


def run_pt_emcee(log_like, log_prior, n_burn, n_steps, n_temps=None,
                 n_walkers=None, p_dict=None, p0=None, columns=None,
                 loglargs=(), logpargs=(), threads=None, thin=1,
                 return_lnZ=False, return_sampler=False, return_pos=False):
    """
    Run emcee.

    Parameters
    ----------
    log_like : function
        The function that computes the log likelihood.  Must be of
        the form log_like(p, *llargs), where p is a NumPy array of
        parameters that are sampled by the MCMC sampler.
    log_prior : function
        The function that computes the log prior.  Must be of
        the form log_post(p, *lpargs), where p is a NumPy array of
        parameters that are sampled by the MCMC sampler.
    n_burn : int
        Number of burn steps
    n_steps : int
        Number of MCMC samples to take
    n_temps : int
        The number of temperatures to use in PT sampling.
    n_walkers : int
        Number of walkers
    p_dict : collections.OrderedDict
        Each entry is a tuple with the function used to generate
        starting points for the parameter and the arguments for
        the function.  The starting point function must have the
        call signature f(*args_for_function, n_walkers).  Ignored
        if p0 is not None.
    p0 : array
        n_walkers by n_dim array of initial starting values.
        p0[k,i,j] is the starting point for walk i along variable j
        for temperature k.  If provided, p_dict is ignored.
    columns : list of strings
        Name of parameters.  These will be the column headings in the
        returned DataFrame.  If None, either inferred from p_dict or
        assigned sequential integers.
    args : tuple
        Arguments passed to log_post
    threads : int
        Number of cores to use in calculation
    thin : int
        The number of iterations to perform between saving the
        state to the internal chain.
    return_lnZ : bool, default False
        If True, additionally return lnZ and dlnZ.
    return_sampler : bool, default False
        If True, additionally return sampler.
    return_pos : bool, default False
        If True, additionally return position of the sampler.

    Returns
    -------
    df : pandas.DataFrame
        First columns give flattened MCMC chains, with columns
        named with the variable being sampled as a string.
        Other columns are:
          'chain':    ID of chain
          'beta':     Inverse temperature
          'beta_ind': Index of beta in list of betas
          'lnlike':   Log likelihood
          'lnprob':   Log posterior probability (with beta multiplying
                      log likelihood)
    lnZ : float, optional
        ln Z(1), which is equal to the evidence of the
        parameter estimation problem.
    dlnZ : float, optional
        The estimated error in the lnZ calculation.
    sampler : emcee.PTSampler instance, optional
        The sampler instance.
    pos : ndarray, shape (ntemps, nwalkers, ndim), optional
        Last position of the walkers.
    """

    if p0 is None and p_dict is None:
        raise RuntimeError('Must supply either p0 or p_dict.')

    # Infer n_dim and n_walkers (and check inputs)
    if p0 is None:
        if n_walkers is None:
            raise RuntimeError('n_walkers must be specified if p0 is None')

        if type(p_dict) is not collections.OrderedDict:
            raise RuntimeError('p_dict must be collections.OrderedDict.')

        n_dim = len(p_dict)
    else:
        n_temps, n_walkers, n_dim = p0.shape
        if p_dict is not None:
            warnings.RuntimeWarning('p_dict is being ignored.')

    # Infer columns
    if columns is None:
        if p_dict is not None:
            columns = list(p_dict.keys())
        else:
            columns = list(range(n_dim))
    elif len(columns) != n_dim:
        raise RuntimeError('len(columns) must equal number of parameters.')

    # Check for invalid column names
    invalid_column_names = ['lnprob', 'chain', 'lnlike', 'beta',
                            'beta_ind']
    if np.any([x in columns for x in invalid_column_names]):
            raise RuntimeError('You cannot name columns with any of these: '
                                    + '  '.join(invalid_column_names))

    # Build starting points of walkers
    if p0 is None:
        p0 = np.empty((n_temps, n_walkers, n_dim))
        for i, key in enumerate(p_dict):
            p0[:, :, i] = p_dict[key][0](
                *(p_dict[key][1] + ((n_temps, n_walkers),)))

    # Set up the PTSampler instance
    if threads is not None:
        sampler = emcee.PTSampler(n_temps, n_walkers, n_dim, log_like,
                                  log_prior, loglargs=loglargs,
                                  logpargs=logpargs, threads=threads)
    else:
        sampler = emcee.PTSampler(n_temps, n_walkers, n_dim, log_like,
                                  log_prior, loglargs=loglargs,
                                  logpargs=logpargs)

    # Do burn-in
    if n_burn > 0:
        pos, _, _ = sampler.run_mcmc(p0, n_burn, storechain=False)
    else:
        pos = p0

    # Sample again, starting from end burn-in state
    pos, _, _ = sampler.run_mcmc(pos, n_steps, thin=thin)

    # Compute thermodynamic integral
    lnZ, dlnZ = sampler.thermodynamic_integration_log_evidence(fburnin=0)

    # Make DataFrame for results
    df = sampler_to_dataframe(sampler, columns=columns)

    # Set up return
    return_vals = (df, lnZ, dlnZ, sampler, pos)
    return_bool = (True, return_lnZ, return_lnZ, return_sampler, return_pos)
    ret = tuple([rv for rv, rb in zip(return_vals, return_bool) if rb])
    if len(ret) == 1:
        return ret[0]
    return ret


def lnZ(df_mcmc):
    """
    Compute log Z(1) from PTMCMC traces stored in DataFrame.

    Parameters
    ----------
    df_mcmc : pandas DataFrame, as outputted from run_ptmcmc.
        DataFrame containing output of a parallel tempering MCMC
        run. Only need to contain columns pertinent to computing
        ln Z, which are 'beta_int', 'lnlike', and 'beta'.

    Returns
    -------
    output : float
        ln Z as computed by thermodynamic integration. This is
        equivalent to what is obtained by calling
        `sampler.thermodynamic_integration_log_evidence(fburnin=0)`
        where `sampler` is an emcee.PTSampler instance.

    Notes
    -----
    .. This is useful when the DataFrame from a PTSampler is too
       large to store in RAM.
    """
    # Average the log likelihood over the samples
    log_mean = np.zeros(len(df_mcmc['beta_ind'].unique()))
    for i, b in enumerate(df_mcmc['beta_ind'].unique()):
        log_mean[i] = df_mcmc['lnlike'][df_mcmc['beta_ind']==b].mean()

    # Set of betas (temperatures)
    betas = np.concatenate((np.array(df_mcmc['beta'].unique()), (0,)))

    # Approximate quadrature
    return np.dot(log_mean, -np.diff(betas))