import warnings

import joblib

import numpy as np
import pandas as pd

import pymc3 as pm
import pymc3.stats
import pymc3.model
import theano.tensor as tt
import tqdm

from .hotdists import *

def _log_like_trace(trace, model, progressbar=False):
    """Calculate the elementwise log-likelihood for the sampled trace.
    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion
    Returns
    -------
    logp : array of shape (n_samples, n_observations)
        The contribution of the observations to the log likelihood of 
        the whole model.

    Notes
    -----
    .. This is a copy of the pymc3.stats._log_post_trace() function for
       PyMC3 version 3.2. That is a misnomer, since it is not the log
       posterior for the trace, but rather the contributions to the 
       log posterior of each observation in the likelihood.
    """
    cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

    def logp_vals_point(pt):
        if len(model.observed_RVs) == 0:
            return floatX(np.array([], dtype='d'))

        logp_vals = []
        for var, logp in cached:
            logp = logp(pt)
            if var.missing_values:
                logp = logp[~var.observations.mask]
            logp_vals.append(logp.ravel())

        return np.concatenate(logp_vals)

    try:
        points = trace.points()
    except AttributeError:
        points = trace

    points = tqdm.tqdm(points) if progressbar else points

    try:
        logp = (logp_vals_point(pt) for pt in points)
        return np.stack(logp)
    finally:
        if progressbar:
            points.close()


def _log_prior_trace(trace, model):
    """Calculate the elementwise log-prior for the sampled trace.
    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.

    Returns
    -------
    logp : array of shape (n_samples, n_observations)
        The contribution of the log prior.
    """
    cached = [var.logp for var in model.unobserved_RVs 
                                if type(var) == pymc3.model.FreeRV]

    def logp_vals_point(pt):
        if len(model.unobserved_RVs) == 0:
            return floatX(np.array([], dtype='d'))

        return np.array([logp(pt) for logp in cached])

    try:
        points = trace.points()
    except AttributeError:
        points = trace

    logp = (logp_vals_point(pt) for pt in points)
    return np.stack(logp)


def _log_posterior_trace(trace, model):
    """
    Log posterior of each point in a trace.
    """
    return (_log_like_trace(trace, model).sum(axis=1) 
            + _log_prior_trace(trace, model).sum(axis=1))


def trace_to_dataframe(trace, model=None, log_post=False):
    """
    Convert a PyMC3 trace to a Pandas DataFrame

    Parameters
    ----------
    trace : PyMC3 trace
        Trace returned from pm.sample()
    model : PyMC3 model, default None
        Model returned from pm.Model()
    log_post : bool, default False
        If True, also compute the log posterior.

    Returns
    -------
    output : Pandas DataFrame
        DataFrame with samples and various sampling statistics.
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

    if trace.nchains > 1:
        for chain in trace.chains[1:]:
            df_app = pm.trace_to_dataframe(trace, chains=[chain])
            for stat in trace.stat_names:
                if stat not in df_app.columns:
                    df_app[stat] = trace.get_sampler_stats(stat,
                                                           chains=[chain])
            if 'chain' not in df_app.columns:
                df_app['chain'] = np.array([chain]*len(df_app))

            df = df.append(df_app, ignore_index=True)

    if log_post:
        # Extract the model from context if necessary
        model = pm.modelcontext(model)

        df['log_likelihood'] = _log_like_trace(trace, model).sum(axis=1)
        df['log_prior'] = _log_prior_trace(trace, model).sum(axis=1)
        df['log_posterior'] = df['log_likelihood'] + df['log_prior']

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


class MarginalizedHomoscedasticNormal(pm.Continuous):
    """
    Likelihood generated by marginalizing out a homoscedastic variance
    from a Normal distribution.

    Parameters
    ----------
    mu : array
        Mean of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Distribution for a multivariate Gaussian with homoscedastic
        error, normalized over sigma.
    """
    def __init__(self, mu, *args, **kwargs):
        super(MarginalizedHomoscedasticNormal, self).__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.mean = mu
        self.mode = mu


    def logp(self, value):
        n = value.shape[-1]
        prefactor = (  pm.distributions.dist_math.gammaln(n/2)
                     - tt.log(2)
                     - 0.5 * n * tt.log(np.pi)) 
        return prefactor - 0.5 * n * tt.log(tt.sum((value - self.mu)**2))


class GoodBad(pm.Continuous):
    """
    Likelihood for the good-bad data model, in which each data point
    is either "good" with a small variance or "bad" with a large 
    variance.

    Parameters
    ----------
    w : float
        Probability that a data point is "good."
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation for "good" data points.
    sigma_bad : float
        Standard deviation for "bad" data points.

    Returns
    -------
    output : pymc3 distribution
        Distribution for the good-bad data model.
    """
    def __init__(self, mu, sigma, sigma_bad, w, *args, **kwargs):
        super(GoodBad, self).__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.sigma = tt.as_tensor_variable(sigma)
        self.sigma_bad = tt.as_tensor_variable(sigma_bad)
        self.w = tt.as_tensor_variable(w)
        self.mean = mu
        self.median = mu
        self.mode = mu

    def logp(self, value):
        prefactor = -tt.log(2.0 * np.pi) / 2.0
        ll_good = (  tt.log(self.w / self.sigma) 
                   - ((value - self.mu) / self.sigma)**2 / 2.0)
        ll_bad = (  tt.log((1.0 - self.w) / self.sigma_bad)
                  - ((value - self.mu) / self.sigma_bad)**2 / 2.0)
        term = tt.switch(tt.gt(ll_good, ll_bad),
                         ll_good + tt.log(1 + tt.exp(ll_bad - ll_good)),
                         ll_bad + tt.log(1 + tt.exp(ll_good - ll_bad)))
        return prefactor + term
    

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


class Ordered(pm.distributions.transforms.ElemwiseTransform):
    """
    Class defining transform to order entries in an array.

    Code from Adrian Seyboldt from PyMC3 discourse: https://discourse.pymc.io/t/mixture-models-and-breaking-class-symmetry/208/4
    """
    name = 'ordered'

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out
    
    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])


class Composed(pm.distributions.transforms.Transform):    
    """
    Class to build a transform out of an elementwise transform.

    Code from Adrian Seyboldt from PyMC3 discourse: https://discourse.pymc.io/t/mixture-models-and-breaking-class-symmetry/208/4   
    """
    def __init__(self, trafo1, trafo2):
        self._trafo1 = trafo1
        self._trafo2 = trafo2
        self.name = '_'.join([trafo1.name, trafo2.name])

    def forward(self, x):
        return self._trafo2.forward(self._trafo1.forward(x))
    
    def forward_val(self, x, point=None):
        return self.forward(x)

    def backward(self, y):
        return self._trafo1.backward(self._trafo2.backward(y))

    def jacobian_det(self, y):
        y2 = self._trafo2.backward(y)
        det1 = self._trafo1.jacobian_det(y2)
        det2 = self._trafo2.jacobian_det(y)
        return det1 + det2


def ordered_transform():
    """
    Make an ordered transform.

    Returns
    -------
    output : pm.distirbutions.transforms.Transform subclass instance
        Transform to order entries in tensor.

    Example
    -------
    To insist on ordering probabilities, p1 <= p2 <= p3,
    >>> p = pymc3.Beta('p',
                       alpha=1,
                       beta=1,
                       shape=3,
                       transform=ordered_transform())
    """
    return Composed(pm.distributions.transforms.LogOdds(), Ordered())


def hotdist(dist, name, beta_temp, *args, **kwargs):
    """
    Instantiate a "hot" distribution. The "hot" distribution takes the
    value returned by the logp method of `dist` and returns beta * logp.

    Parameters
    ----------
    dist : PyMC3 distribution
        The name of a distribution you want to make hot. Examples:
        pm.Normal, pm.Binomial, pm.MvNormal, pm.Dirichlet.
    name : str
        Name of the random variable.
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot distribution. 
    """
    class HotDistribution(dist):
        def __init__(self, beta_temp, *args, **kwargs):
            super(HotDistribution, self).__init__(*args, **kwargs)
            if not (0 <= beta_temp <= 1):
                raise RuntimeError('Must have 0 ≤ beta_temp ≤ 1.')
            self.beta_temp = beta_temp

        def logp(self, value):
            return self.beta_temp * dist.logp(self, value)
        
    return HotDistribution(name, beta_temp, *args, **kwargs)


def beta_ladder(n=20, beta_min=1e-8):
    return np.logspace(np.log10(beta_min), 0, n)


def _sample_beta(beta, model_fun, args, kwargs):
    print(f'Sampling beta = {beta}....')

    model = model_fun(beta, *args)

    with model:
        trace = pm.sample(**kwargs)

    return trace, model, beta


def sample_beta_ladder(model_fun, betas, args=(), njobs=1, draws=500,
                       tune=500, progressbar=True, **kwargs):
    """
    Draw MCMC samples for a distribution for various values of beta.

    Parameters
    ----------
    model_fun : function
        Function that returns a PyMC3 model. Must have call signature
        model_fun(beta_temp, *args), where `beta_temp` is the inverse
        temperature.
    betas : array_like
        Array of beta values to sample.
    args : tuple, default ()
        Any additional arguments passed into `model_fun`.
    njobs : int, default 1
        Number of temperatures to run in parallel. This is *not* the
        number of samplers to run in parallel for each temperature.
        Each temperature is only sampled with a single walker.
    draws : int, default 500
        Number of samples to generate for each temperature.
    tune : int, default 500
        Number of tuning steps to take for each temperature.
    progressbar : bool, default True
        If True, show progress bars of samlers.
    kwargs 
        All additional kwargs are passed to pm.sample().

    Returns
    -------
    traces : list of PyMC3 traces
        Traces for each value in betas
    models : list of PyMC3 models
        List of "hot" models corresponding to each value of beta.

    Example
    -------
    .. Draw samples out of a Normal distribution with a flat prior
       on `mu` and a HalfCauchy prior on `sigma`.

       x = np.random.normal(0, 1, size=100)

       def norm_model(beta_temp, beta_cauchy, x):
           with pm.Model() as model:
               mu = pm.Flat('mu')
               sigma = pm.HalfCauchy('sigma', beta=beta_cauchy)
               x_obs = HotNormal('x_obs', beta_temp=beta_temp, mu=mu, 
                                 sd=sigma, observed=x)
           return model

       betas = np.logspace(-3, 0, 10)
       samples, models = sample_beta_ladder(
                           norm_model, betas, args=(beta_cauchy, x))


    """
    # Insert code here to pop draws, tune, and progressbar out of kwargs

    if np.any(betas < 0) or np.any(betas > 1):
        raise RuntimeError('All beta values must be on interval (0, 1].')

    if not np.any(betas == 1):
        warnings.warn(
            'You probably want to sample beta = 1, the cold distribution.')

    if np.any(betas == 0):
        raise RuntimeError("Sampling beta = 0 not allowed;"
                           + " you're just sampling the prior in that case.")

    if len(betas) != len(np.unique(betas)):
        raise RuntimeError('Repeated beta entry.')

    kwargs['draws'] = draws
    kwargs['tune'] = tune
    kwargs['progressbar'] = progressbar

    if njobs == 1:
        return [_sample_beta(beta, model_fun, args, kwargs) for beta in betas]
    else:
        jobs = (joblib.delayed(_sample_beta)(beta, model_fun, args, kwargs)
                     for beta in betas)
        return joblib.Parallel(n_jobs=njobs)(jobs)


def log_evidence_estimate(trace_model_beta):
    """
    Compute an estimate of the log evidence.

    Parameters
    ----------
    trace_model_beta : list of (trace, model, beta) tuples
        List of (trace, model, beta) tuples as would be returned by
        sample_beta_ladder().

    Returns
    -------
    output : float
        Approximate negative log evidence.
    """

    # Extract traces, models, and betas
    betas = []
    traces = []
    models = []
    for tmb in trace_model_beta:
        traces.append(tmb[0])
        models.append(tmb[1])
        betas.append(tmb[2])

    betas = np.array(betas)

    if np.any(betas <= 0) or np.any(betas > 1):
        raise RuntimeError('All beta values must be between zero and one.')

    if len(betas) != len(np.unique(betas)):
        raise RuntimeError('Repeated beta entry.')

    # Sort
    inds = np.argsort(betas)
    betas = betas = [betas[i] for i in inds]
    traces =traces = [traces[i] for i in inds]
    models = models = [models[i] for i in inds]

    # Compute average log likelihood
    mean_log_like = []
    for beta, trace, model in zip(betas, traces, models):
        mean_log_like.append(_log_like_trace(trace, model).sum(axis=1).mean()
                                 / beta)

    # Add zero value
    betas = np.concatenate(((0,), betas))
    mean_log_like = np.concatenate(((mean_log_like[0],), mean_log_like))

    # Perform integral
    return np.trapz(mean_log_like, x=betas)


def chol_to_cov(chol, cov_prefix):
    """
    Convert flattened Cholesky matrix to covariance.

    Parameters
    ----------
    chol : array_like
        Lexicographically flattened Cholesky decomposition as returned
        from trace.get_values(chol), where trace is a PyMC3 MultiTrace
        instance.
    chol_prefix : str
        Prefix for the nam e of the covariance variable. Results are
        stored as prefix__i__j, where i and j are the row and column
        indices, respectively.

    Returns
    -------
    output : Pandas DataFrame
        DataFrame with values of samples of the components of the
        covariance matrix.
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
