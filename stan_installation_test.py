import pystan
import cython
import numpy
import scipy
import matplotlib

assert (numpy.__version__[:3] in ['1.7', '1.8', '1.9'] 
        or numpy.__version__[:4] in ['1.10', '1.11', '1.12', 
                                     '1.13', '1.14', '1.15', 
                                     '1.16', '1.17', '1.18'])

assert cython.__version__ >= '0.22' and cython.__version__ != '0.25.1'

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

schools_code = """
data {
  int<lower=0> J; // number of schools
  vector[J] y; // estimated treatment effects
  vector<lower=0>[J] sigma; // s.e. of effect estimates
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;
}

transformed parameters {
  vector[J] theta = mu + tau * eta;
}

model {
  eta ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
"""

sm = pystan.StanModel(model_code=schools_code)
fit = sm.sampling(data=schools_dat, iter=1000, chains=1)