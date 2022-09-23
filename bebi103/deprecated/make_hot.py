master_str = '''
class Hot{0:s}(pm.{0:s}):
    """
    A "hot" {0:s} distribution.

    Parameters
    ----------
    beta : float on interval [0, 1]
        Beta value (inverse temperature) of the distribution.

    Returns
    -------
    output : pymc3 distribution
        Hot {0:s} distribution. 
    """
    def __init__(self, beta, *args, **kwargs):
        super(Hot{0:s}, self).__init__(*args, **kwargs)
        if not (0 <= beta <= 1):
            raise RuntimeError('Must have 0 ≤ beta ≤ 1.')
        self.beta = beta
        
    def logp(self, value):
        return self.beta * pm.{0:s}.logp(self, value)

'''

dists = ['Binomial',  'BetaBinomial',  'Bernoulli',  'DiscreteWeibull',
           'Poisson', 'NegativeBinomial', 'ConstantDist', 'Constant',
           'ZeroInflatedPoisson', 'ZeroInflatedBinomial', 'ZeroInflatedNegativeBinomial',
           'DiscreteUniform', 'Geometric', 'Categorical'] \
           + ['Uniform', 'Flat', 'HalfFlat', 'Normal', 'Beta', 'Exponential',
           'Laplace', 'StudentT', 'Cauchy', 'HalfCauchy', 'Gamma', 'Weibull',
           'HalfStudentT', 'StudentTpos', 'Lognormal', 'ChiSquared',
           'HalfNormal', 'Wald', 'Pareto', 'InverseGamma', 'ExGaussian',
           'VonMises', 'SkewNormal', 'Logistic', 'Interpolated'] \
           + ['MvNormal', 'MvStudentT', 'Dirichlet',
           'Multinomial', 'Wishart', 'WishartBartlett',
           'LKJCorr', 'LKJCholeskyCov'] \
           + ['Mixture', 'NormalMixture']

with open('hot_dists.py', 'w') as f:
    for dist in dists:
        f.write(master_str.format(dist))
