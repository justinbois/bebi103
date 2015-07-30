# As usual, import modules
from __future__ import division, absolute_import, \
                                    print_function, unicode_literals

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd

import emcee
import beeswarm
import triangle
from brewer2mpl import sequential

# Utilities from JB
import jb_utils as jb


plt.clf()


file_name = '../data/reeves_et_al/reeves_gradient_width_various_methods.csv'
df = pd.read_csv(file_name, comment='#')

# Rename key columns
df.rename(columns={'wt cross-sections': 'wt', 
                   'anti-Dorsal dl1/+dl-venus/+': 'dorsal',
                   'anti-Venus dl1/+dl-venus/+': 'venus',
                   'anti-GFP  dl1/+dl-GFP/+': 'gfp'}, inplace=True)

# Generate a beeswarm plot
#bs_plot, ax = beeswarm.beeswarm(
#        [df.wt.dropna(), df.dorsal.dropna(), df.venus.dropna(), 
#         df.gfp.dropna()], 
#        labels=['wt', 'dorsal', 'venus', 'gfp'])
#plt.ylabel('gradient width')
#
#plt.xlim((-0.75, 3.75))


def log_posterior(p, x, x_min, x_max):
    """
    Lots to do; lazy.
    """

    # Unpack parameters
    alpha, beta = p
    
    # Number of data
    n = len(x)
    
    if alpha < x_min or alpha > x_max or beta <= 0:
        return -np.inf
    
    return -(n + 1) * np.log(beta) \
                - np.log(1.0 + ((x - alpha) / beta)**2).sum()

## Set up MCMC parameters
#n_dim = 2
#n_walkers = 10
#n_burn = 500
#n_steps = 5000
#
## Seed random number generator
#np.random.seed(42)
#
## Generate starting positions
#p0 = np.empty((n_walkers, n_dim))
#p0[:,0] = np.random.uniform(0.0, 1.0, n_walkers)
#p0[:,1] = np.random.exponential(1.0, n_walkers)
#
## Instantiate sampler
#sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
#                                args=(df.wt.dropna(), 0.0, 10.0))
#
## Do burn-in
#pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)
#
## Let's go!
#sampler.reset()
#pos, prob, state = sampler.run_mcmc(pos, n_steps)
#

# print('Gaussian parameter: %g' % df.wt.dropna().mean())
# print('Cauchy parameter:   %g' % sampler.flatchain[:,0].mean())


venus_corrupt = np.concatenate((df.venus.dropna().values, (0.45, 0.5, 0.46)))

bs_plot, ax = beeswarm.beeswarm([venus_corrupt], labels=['venus'])

# Set up MCMC parameters
n_dim = 2
n_walkers = 10
n_burn = 500
n_steps = 5000

# Seed random number generator
np.random.seed(42)

# Generate starting positions
p0 = np.empty((n_walkers, n_dim))
p0[:,0] = np.random.uniform(0.0, 1.0, n_walkers)
p0[:,1] = np.random.exponential(1.0, n_walkers)

# Instantiate sampler
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
                                args=(venus_corrupt, 0.0, 10.0))

# Do burn-in
pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)

# Let's go!
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, n_steps)


print('Gaussian parameter: %g' % venus_corrupt.mean())
print('Cauchy parameter:   %g' % sampler.flatchain[:,0].mean())



















plt.draw()
plt.show()
