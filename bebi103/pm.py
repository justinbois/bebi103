import warnings

import numpy as np
import pandas as pd

import pymc3 as pm
import pymc3.stats
import theano.tensor as tt

def trace_to_dataframe(trace, model=None, varnames=None, 
                       include_transformed=False, log_post=False):
    """
    Convert a PyMC3 trace to a Pandas DataFrame

    To add: Compute logp for each point using model.logp().
    """
    df = pm.trace_to_dataframe(trace, chains=[0])
    for stat in trace.stat_names:
        if stat in df.columns:
            warnings.warn('`' + stat + '` is in the variable names.`'
                          + ' Not adding this statistic.')
        else:
            df[stat] = trace.get_sampler_stats(stat, chains=[0])
    if 'chain' in df.columns:
        warnings.warn('`chain` is in the variable name.`'
                          + ' Not adding this statistic.')
    else:
        df['chain'] = np.array([0]*len(df), dtype=int)

    if len(trace.chains) == 1:
        return df      

    for chain in trace.chains[1:]:
        df_app = pm.trace_to_dataframe(trace, chains=[chain])
        for stat in trace.stat_names:
            if stat not in df_app.columns:
                df_app[stat] = trace.get_sampler_stats(stat, chains=[chain])
        if 'chain' not in df_app.columns:
            df_app['chain'] = np.array([chain]*len(df_app))

        df = df.append(df_app, ignore_index=True)

    if log_post:
        # Extract the model from context if necessary
        model = pm.modelcontext(model)

        logp = pymc3.stats._log_post_trace(trace, model).sum(axis=1)
        df['log_posterior'] = logp

    return df


class Jeffreys(pm.Continuous):
    """
    Jeffreys prior for a scale parameter.

    Parameters
    ----------
    lower : float, > 0
        Minimum value the variable can take.
    upper : float, > `lower`
        Maximum value the variable can take.

    Returns
    -------
    output : pymc3 distribution
        Distribution for Jeffreys prior.
    """
    def __init__(self, lower=None, upper=None, transform='interval',
                 *args, **kwargs):
        # Check inputs
        if lower is None or upper is None:
            raise RuntimeError('`lower` and `upper` must be provided.')
        if lower <= 0:
            raise RuntimeError('`lower` must be > 0.')
        if upper <= lower:
            raise RuntimeError('`upper` must be > `lower`.')

        if transform == 'interval':
            transform = pm.distributions.transforms.interval(lower, upper)
        super(Jeffreys, self).__init__(transform=transform, *args, **kwargs)        
        self.lower = lower = pm.theanof.floatX(tt.as_tensor_variable(lower))
        self.upper = upper = pm.theanof.floatX(tt.as_tensor_variable(upper))
        
        self.mean = (upper - lower) / tt.log(upper/lower)
        self.median = tt.sqrt(lower * upper)
        self.mode = lower
        
    def logp(self, value):
        lower = self.lower
        upper = self.upper
        return pm.distributions.dist_math.bound(
                    -tt.log(tt.log(upper/lower)) - tt.log(value),
                    value >= lower, value <= upper)


# def Jeffreys(name, lower=None, upper=None, shape=None):
#     """
#     Create a Jeffreys prior for a scale parameter.

#     Parameters
#     ----------
#     name : str
#         Name of the variable.
#     lower : float, > 0
#         Minimum value the variable can take.
#     upper : float, > `lower`
#         Maximum value the variable can take.
#     shape: int or tuple of ints, default 1
#         Shape of array of variables. If 1, then a single scalar.

#     Returns
#     -------
#     output : pymc3 distribution
#         Distribution for Jeffreys prior.
#     """
#     # Check inputs
#     if type(name) != str:
#         raise RuntimeError('`name` must be a string.')
#     if lower is None or upper is None:
#         raise RuntimeError('`lower` and `upper` must be provided.')
#     if lower <= 0:
#         raise RuntimeError('`lower` must be > 0.')
#     if upper <= lower:
#         raise RuntimeError('`upper` must be > `lower`.')

#     # Set up Jeffreys prior
#     if shape is None:
#         log_var = pm.Uniform('log_' + name, 
#                              lower=np.log(lower), 
#                              upper=np.log(upper))
#     else:
#         log_var = pm.Uniform('log_' + name, 
#                              lower=np.log(lower), 
#                              upper=np.log(upper),
#                              shape=shape)
#     var = pm.Deterministic(name, pm.math.exp(log_var))

#     return var


def ReparametrizedNormal(name, mu=None, sd=None, shape=1):
    """
    Create a reparametrized Normally distributed random variable.

    Parameters
    ----------
    name : str
        Name of the variable.
    mu : float
        Mean of Normal distribution.
    sd : float, > 0
        Standard deviation of Normal distribution.
    shape: int or tuple of ints, default 1
        Shape of array of variables. If 1, then a single scalar.

    Returns
    -------
    output : pymc3 distribution
        Distribution for a reparametrized Normal distribution.

    Notes
    -----
    .. The reparametrization procedure allows the sampler to sample
       a standard normal distribution, and then do a deterministic
       reparametrization to achieve sampling of the original desired 
       Normal distribution.
    """
    # Check inputs
    if type(name) != str:
        raise RuntimeError('`name` must be a string.')
    if mu is None or sd is None:
        raise RuntimeError('`mu` and `sd` must be provided.')

    var_reparam = pm.Normal(name + '_reparam', mu=0, sd=1, shape=shape)
    var = pm.Deterministic(name, mu + var_reparam * sd)

    return var


def ReparametrizedCauchy(name, alpha=None, beta=None, shape=1):
    """
    Create a reparametrized Cauchy distributed random variable.

    Parameters
    ----------
    name : str
        Name of the variable.
    alpha : float
        Mode of Cauchy distribution.
    beta : float, > 0
        Scale parameter of Cauchy distribution
    shape: int or tuple of ints, default 1
        Shape of array of variables. If 1, then a single scalar.

    Returns
    -------
    output : pymc3 distribution
        Reparametrized Cauchy distribution.

    Notes
    -----
    .. The reparametrization procedure allows the sampler to sample
       a Cauchy distribution with alpha = 0 and beta = 1, and then do a
       deterministic reparametrization to achieve sampling of the 
       original desired Cauchy distribution.
    """
    # Check inputs
    if type(name) != str:
        raise RuntimeError('`name` must be a string.')
    if alpha is None or beta is None:
        raise RuntimeError('`alpha` and `beta` must be provided.')

    var_reparam = pm.Cauchy(name + '_reparam', alpha=0, beta=1, shape=shape)
    var = pm.Deterministic(name, alpha + var_reparam * beta)

    return var


def chol_to_cov(chol, cov_prefix):
    """
    Convert flattened Cholesky matrix to covariance.
    """
    chol = np.array(chol)

    n = int(np.round((-1 + np.sqrt(8*chol.shape[1] + 1)) / 2))
    sigma = np.zeros_like(chol)
    inds = np.tril_indices(n)
    for i, r in enumerate(chol):
        L =  np.zeros((n, n))
        L[inds] = r
        sig = np.dot(L, L.T)
        sigma[i] = sig[inds]

    cols = ['{0:s}__{1:d}__{2:d}'.format(cov_prefix, i, j) 
                for i, j in zip(*inds)]
    return pd.DataFrame(columns=cols, data=sigma)
