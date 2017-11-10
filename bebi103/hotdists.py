import theano.tensor as tt
import pymc3 as pm

class HotBinomial(pm.Binomial):
    """
    A "hot" Binomial distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Binomial distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotBinomial, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Binomial.logp(self, value)


class HotBetaBinomial(pm.BetaBinomial):
    """
    A "hot" BetaBinomial distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot BetaBinomial distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotBetaBinomial, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.BetaBinomial.logp(self, value)


class HotBernoulli(pm.Bernoulli):
    """
    A "hot" Bernoulli distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Bernoulli distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotBernoulli, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Bernoulli.logp(self, value)


class HotDiscreteWeibull(pm.DiscreteWeibull):
    """
    A "hot" DiscreteWeibull distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot DiscreteWeibull distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotDiscreteWeibull, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.DiscreteWeibull.logp(self, value)


class HotPoisson(pm.Poisson):
    """
    A "hot" Poisson distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Poisson distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotPoisson, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Poisson.logp(self, value)


class HotNegativeBinomial(pm.NegativeBinomial):
    """
    A "hot" NegativeBinomial distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot NegativeBinomial distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotNegativeBinomial, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.NegativeBinomial.logp(self, value)


class HotConstantDist(pm.ConstantDist):
    """
    A "hot" ConstantDist distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot ConstantDist distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotConstantDist, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.ConstantDist.logp(self, value)


class HotConstant(pm.Constant):
    """
    A "hot" Constant distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Constant distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotConstant, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Constant.logp(self, value)


class HotZeroInflatedPoisson(pm.ZeroInflatedPoisson):
    """
    A "hot" ZeroInflatedPoisson distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot ZeroInflatedPoisson distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotZeroInflatedPoisson, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.ZeroInflatedPoisson.logp(self, value)


class HotZeroInflatedBinomial(pm.ZeroInflatedBinomial):
    """
    A "hot" ZeroInflatedBinomial distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot ZeroInflatedBinomial distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotZeroInflatedBinomial, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.ZeroInflatedBinomial.logp(self, value)


class HotZeroInflatedNegativeBinomial(pm.ZeroInflatedNegativeBinomial):
    """
    A "hot" ZeroInflatedNegativeBinomial distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot ZeroInflatedNegativeBinomial distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotZeroInflatedNegativeBinomial, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.ZeroInflatedNegativeBinomial.logp(self, value)


class HotDiscreteUniform(pm.DiscreteUniform):
    """
    A "hot" DiscreteUniform distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot DiscreteUniform distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotDiscreteUniform, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.DiscreteUniform.logp(self, value)


class HotGeometric(pm.Geometric):
    """
    A "hot" Geometric distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Geometric distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotGeometric, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Geometric.logp(self, value)


class HotCategorical(pm.Categorical):
    """
    A "hot" Categorical distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Categorical distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotCategorical, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Categorical.logp(self, value)


class HotUniform(pm.Uniform):
    """
    A "hot" Uniform distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Uniform distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotUniform, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Uniform.logp(self, value)


class HotFlat(pm.Flat):
    """
    A "hot" Flat distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Flat distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotFlat, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Flat.logp(self, value)


class HotHalfFlat(pm.HalfFlat):
    """
    A "hot" HalfFlat distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot HalfFlat distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotHalfFlat, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.HalfFlat.logp(self, value)


class HotNormal(pm.Normal):
    """
    A "hot" Normal distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Normal distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotNormal, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Normal.logp(self, value)


class HotBeta(pm.Beta):
    """
    A "hot" Beta distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Beta distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotBeta, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Beta.logp(self, value)


class HotExponential(pm.Exponential):
    """
    A "hot" Exponential distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Exponential distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotExponential, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Exponential.logp(self, value)


class HotLaplace(pm.Laplace):
    """
    A "hot" Laplace distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Laplace distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotLaplace, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Laplace.logp(self, value)


class HotStudentT(pm.StudentT):
    """
    A "hot" StudentT distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot StudentT distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotStudentT, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.StudentT.logp(self, value)


class HotCauchy(pm.Cauchy):
    """
    A "hot" Cauchy distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Cauchy distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotCauchy, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Cauchy.logp(self, value)


class HotHalfCauchy(pm.HalfCauchy):
    """
    A "hot" HalfCauchy distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot HalfCauchy distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotHalfCauchy, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.HalfCauchy.logp(self, value)


class HotGamma(pm.Gamma):
    """
    A "hot" Gamma distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Gamma distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotGamma, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Gamma.logp(self, value)


class HotWeibull(pm.Weibull):
    """
    A "hot" Weibull distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Weibull distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotWeibull, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Weibull.logp(self, value)


HotHalfStudentT = pm.Bound(pm.StudentT, lower=0)


class HotLognormal(pm.Lognormal):
    """
    A "hot" Lognormal distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Lognormal distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotLognormal, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Lognormal.logp(self, value)


class HotChiSquared(pm.ChiSquared):
    """
    A "hot" ChiSquared distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot ChiSquared distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotChiSquared, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.ChiSquared.logp(self, value)


class HotHalfNormal(pm.HalfNormal):
    """
    A "hot" HalfNormal distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot HalfNormal distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotHalfNormal, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.HalfNormal.logp(self, value)


class HotWald(pm.Wald):
    """
    A "hot" Wald distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Wald distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotWald, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Wald.logp(self, value)


class HotPareto(pm.Pareto):
    """
    A "hot" Pareto distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Pareto distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotPareto, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Pareto.logp(self, value)


class HotInverseGamma(pm.InverseGamma):
    """
    A "hot" InverseGamma distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot InverseGamma distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotInverseGamma, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.InverseGamma.logp(self, value)


class HotExGaussian(pm.ExGaussian):
    """
    A "hot" ExGaussian distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot ExGaussian distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotExGaussian, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.ExGaussian.logp(self, value)


class HotVonMises(pm.VonMises):
    """
    A "hot" VonMises distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot VonMises distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotVonMises, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.VonMises.logp(self, value)


class HotSkewNormal(pm.SkewNormal):
    """
    A "hot" SkewNormal distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot SkewNormal distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotSkewNormal, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.SkewNormal.logp(self, value)


class HotLogistic(pm.Logistic):
    """
    A "hot" Logistic distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Logistic distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotLogistic, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Logistic.logp(self, value)


class HotInterpolated(pm.Interpolated):
    """
    A "hot" Interpolated distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Interpolated distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotInterpolated, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Interpolated.logp(self, value)


class HotMvNormal(pm.MvNormal):
    """
    A "hot" MvNormal distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot MvNormal distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotMvNormal, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.MvNormal.logp(self, value)


class HotMvStudentT(pm.MvStudentT):
    """
    A "hot" MvStudentT distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot MvStudentT distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotMvStudentT, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.MvStudentT.logp(self, value)


class HotDirichlet(pm.Dirichlet):
    """
    A "hot" Dirichlet distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Dirichlet distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotDirichlet, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Dirichlet.logp(self, value)


class HotMultinomial(pm.Multinomial):
    """
    A "hot" Multinomial distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Multinomial distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotMultinomial, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Multinomial.logp(self, value)


class HotWishart(pm.Wishart):
    """
    A "hot" Wishart distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Wishart distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotWishart, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Wishart.logp(self, value)


class HotLKJCorr(pm.LKJCorr):
    """
    A "hot" LKJCorr distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot LKJCorr distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotLKJCorr, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.LKJCorr.logp(self, value)


class HotLKJCholeskyCov(pm.LKJCholeskyCov):
    """
    A "hot" LKJCholeskyCov distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot LKJCholeskyCov distribution. 
    """
    def __init__(self, beta_temp, *args, **kwargs):
        super(HotLKJCholeskyCov, self).__init__(*args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.LKJCholeskyCov.logp(self, value)


class HotMixture(pm.Mixture):
    """
    A "hot" Mixture distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot Mixture distribution. 
    """
    def __init__(self, beta_temp, w, comp_dists, *args, **kwargs):

        super(HotMixture, self).__init__(w, comp_dists, *args, **kwargs)

        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.Mixture.logp(self, value)


class HotNormalMixture(pm.NormalMixture):
    """
    A "hot" NormalMixture distribution.

    Parameters
    ----------
    beta_temp : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot NormalMixture distribution. 
    """
    def __init__(self, beta_temp, w, mu, *args, **kwargs):
        super(HotNormalMixture, self).__init__(w, mu, *args, **kwargs)
        if not (0 < beta_temp <= 1):
            raise RuntimeError('Must have 0 < beta_temp ≤ 1.')
        self.beta_temp = beta_temp
        
    def logp(self, value):
        return self.beta_temp * pm.NormalMixture.logp(self, value)

