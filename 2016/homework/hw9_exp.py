
# coding: utf-8


import glob
import os

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.optimize
import scipy.signal
import scipy.stats as st
import numba

# Image processing tools
import skimage
import skimage.io
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation

import bebi103


def area_thresh(im, interpixel_distance):
    """
    Threshold an image and return the area of the largest object.
    """
    # Threshold image
    im_bw = im < skimage.filters.threshold_otsu(im)

    # Clear borders
    im_bw = skimage.segmentation.clear_border(im_bw)

    # Label binary image
    im_labeled = skimage.measure.label(im_bw, background=0, return_num=False)

    # Find areas of all objects
    im_props = skimage.measure.regionprops(im_labeled)

    # Find largest area
    max_area = 0
    for prop in im_props:
        if prop.area > max_area:
            max_area = prop.area

    return max_area * interpixel_distance**2

# Images and metadata
data_dir = '../data/iyer_biswas_et_al'
fname = os.path.join(data_dir, 'bacterium_2.tif')


# Load image collection as 2D NumPy array
ic = skimage.io.imread(fname)



def areas_divisions(t, ic, interpixel_distance, div_thresh, n_below, abs_thresh,
                    median_kernel_size):
    """
    Compute areas of mother bacterium over time
    """
    # Compute the bacterial area for each image
    areas = np.empty(len(ic))
    for i, im in enumerate(ic):
        areas[i] = area_thresh(im, interpixel_distance)

    # Determine which points are good
    good_inds = (areas > abs_thresh[0]) & (areas < abs_thresh[1])

    # Slice out good data points
    t = t[good_inds]
    areas = areas[good_inds]

    # Perform median filter to eliminate outliers
    areas_filt = scipy.signal.medfilt(areas, kernel_size=median_kernel_size)

    # Get downcrossings
    down_inds = np.where((areas_filt[:-1] > div_thresh)
                        & (areas_filt[1:] < div_thresh))[0] + 1

    # Filter out spurious downcrossings
    div_inds = [i for i in down_inds
                        if np.all(areas_filt[i:i+n_below] < div_thresh)]

    return t, areas, div_inds



# Images and metadata
data_dir = '../data/iyer_biswas_et_al'
fname = os.path.join(data_dir, 'bacterium_1.tif')
interpixel_distance = 0.052  # µm
dt = 1 # minutes

# Threshold for division
div_thresh = 2  # µm^2

# Absolute thresholds for big and small cells
abs_thresh = (1, 3.5) # µm^2

# Number of time points below threshold to be considered divided
n_below = 10

# Size of median filter kernel
median_kernel_size = 9

# Load images
ic = skimage.io.imread(fname)

# Determine time points
t_1 = np.arange(len(ic)) / dt

# Get area/time curve with division points
t_1, areas_1, div_inds_1 = areas_divisions(
        t_1, ic, interpixel_distance, div_thresh, n_below, abs_thresh,
        median_kernel_size)


# Images and metadata
fname = os.path.join(data_dir, 'bacterium_2.tif')

# Load images
ic = skimage.io.imread(fname)

# Determine time points
t_2 = np.arange(len(ic)) / dt

# Do the calculation
t_2, areas_2, div_inds_2 = areas_divisions(
        t_2, ic, interpixel_distance, div_thresh, n_below, abs_thresh,
        median_kernel_size)



# Set up DataFrame with results
df = pd.DataFrame(columns=['bacterium', 'division time (min)'])
div_times_1 = np.diff(div_inds_1) * dt
div_times_2 = np.diff(div_inds_2) * dt
df['bacterium'] = np.concatenate((np.ones_like(div_times_1),
                                  np.ones_like(div_times_2) * 2))
df['division time (min)'] = np.concatenate((div_times_1, div_times_2))


def make_dfs(t, areas, div_inds):
    """
    Create a list of DataFrames containing time/area data between successive
    division event.
    """
    # First division (ignore first time point because it's right before division
    data = np.array([t[1:div_inds[0]] - t[1], areas[1:div_inds[0]]]).T
    df_list = [pd.DataFrame(columns=['t', 'area'], data=data)]

    # Add growth curves for subsequent divisions
    for i, ind in enumerate(div_inds[:-1]):
        data = np.array([t[ind:div_inds[i+1]-1] - t[ind],
                         areas[ind:div_inds[i+1]-1]]).T
        df_list.append(pd.DataFrame(columns=['t', 'area'], data=data))

    return df_list

# Make the DataFrames
df_1 = make_dfs(t_1, areas_1, div_inds_1)
df_2 = make_dfs(t_2, areas_2, div_inds_2)




# Delete unreasonable jumps
window = 10
jump_thresh = 0.2

for df_list in [df_1, df_2]:
    for i, df in enumerate(df_list):
        # Compute rolling local median
        local_median = df.area.rolling(center=True, window=window).median()

        # Local median for ends
        local_median[:window//2] = df.area[:window].median()
        local_median[-window//2:] = df.area[-window:].median()

        # Compute fractional difference and
        frac_diff = np.abs((df.area - local_median) / local_median)
        df.loc[frac_diff > jump_thresh, 'area'] = np.nan
        df_list[i] = df.dropna(how='any')



@numba.jit(nopython=True)
def unpack_prior(p, n_divisions):
    """
    Unpack parameters for the prior.

    n_divisions is a list of the number of curves for each bacterium.
    """
    # Total number of divisions, useful to have around.
    n_tot_divisions = n_divisions.sum()

    # Pull out a values
    a_div = p[:n_tot_divisions]

    # Pull out b or kappa values for each division
    bk_div = p[n_tot_divisions:2*n_tot_divisions]

    # Pull out a values for individual bacteria
    ind = 2*n_tot_divisions
    a_indiv = p[ind:ind+len(n_divisions)]

    # Pull out b/kappa values for individual bacteria
    ind += len(n_divisions)
    bk_indiv = p[ind:ind+len(n_divisions)]

    # Pull out sigma_a values for individual bacteria
    ind += len(n_divisions)
    sigma_a_indiv = p[ind:ind+len(n_divisions)]

    # Pull out sigma_b values for individual bacteria
    ind += len(n_divisions)
    sigma_bk_indiv = p[ind:ind+len(n_divisions)]

    # Pull out highest level hyperparameters
    a = p[-5]
    bk = p[-4]
    sigma_a = p[-3]
    sigma_bk = p[-2]

    # Pull out Cauchy scale parameter.
    beta = p[-1]

    return a_div, bk_div, a_indiv, bk_indiv, sigma_a_indiv, sigma_bk_indiv,                 a, bk, sigma_a, sigma_bk, beta


@numba.jit(nopython=True)
def unpack_like(p, n_divisions):
    """
    Unpack parameters for likelihood calculation.
    """
    # Pull out a values
    n_total_divisions = n_divisions.sum()
    a_div = p[:n_total_divisions]
    b_div = p[n_total_divisions:2*n_total_divisions]
    beta = p[-1]

    return a_div, b_div, beta


# Now that we can unpack the parameters, we can compute the log likelihood.  We first define the two theoretical functions we will use for the fits.

# In[21]:

@numba.jit(nopython=True)
def linear_fun(t, a, b):
    return a + b * t

@numba.jit(nopython=True)
def exp_fun(t, a, kappa):
    return a * np.exp(kappa * t)


# We also need to store the `DataFrames` as NumPys array to use Numba.  To do this, we concatenate all of the time points together, as well as the areas.  We store the indices where we have break points between growth curves. It is also convenient to store the number of divisions for each bacterium while we're at it.

# In[22]:

# Specify our data sets
list_of_df_lists = [df_1, df_2]

# Get number of divisions
n_divisions = np.array([len(df_list) for df_list in list_of_df_lists])

# Make set of division times
t_list = [df.t.values for df_list in list_of_df_lists for df in df_list]
t = np.concatenate(t_list)
areas = np.concatenate([df.area.values for df_list in list_of_df_lists
                              for df in df_list])
n_points = np.array([len(t) for t in t_list])
inds = np.concatenate(((0,), n_points.cumsum()))


# Next, we define the log likelihood.  We take the parameter `bk` and other similarly named parameters to mean either the parameter $b_{ij}$ or $\kappa_{ij}$, since they can be used interchangeably, depending on the theoretical function we are using.

# In[23]:

@numba.jit(nopython=True)
def cauchy_logpdf(x, mu, beta):
    """
    Log PDF of the Cauchy distribution.
    """
    return -len(x) * np.log(np.pi * beta)                     - np.sum(np.log(1.0 + ((x - mu) / beta)**2))


@numba.jit(nopython=True)
def log_likelihood_single_curve_linear(a, b, t, area, beta):
    """
    The log likelihood for a single growth curve.
    """
    area_theor = linear_fun(t, a, b)
    return cauchy_logpdf(area, area_theor, beta)


@numba.jit(nopython=True)
def log_likelihood_linear(p, t, areas, inds, n_divisions):
    """
    The log likelihood for all growth curves for all bacteria.
    """
    # Unpack parameters
    a_div, b_div, beta = unpack_like(p, n_divisions)

    log_like = 0.0
    for i in range(n_divisions.sum()):
        t_ = t[inds[i]:inds[i+1]]
        area_ = areas[inds[i]:inds[i+1]]
        log_like += log_likelihood_single_curve_linear(
                                        a_div[i], b_div[i], t_, area_, beta)
    return log_like


@numba.jit(nopython=True)
def log_likelihood_single_curve_exp(a, kappa, t, area, beta):
    """
    The log likelihood for a single growth curve.
    """
    area_theor = exp_fun(t, a, kappa)
    return cauchy_logpdf(area, area_theor, beta)


@numba.jit(nopython=True)
def log_likelihood_exp(p, t, areas, inds, n_divisions):
    """
    The log likelihood for all growth curves for all bacteria.
    """
    # Unpack parameters
    a_div, kappa_div, beta = unpack_like(p, n_divisions)

    log_like = 0.0
    for i in range(n_divisions.sum()):
        t_ = t[inds[i]:inds[i+1]]
        area_ = areas[inds[i]:inds[i+1]]
        log_like += log_likelihood_single_curve_exp(
                                        a_div[i], kappa_div[i], t_, area_, beta)
    return log_like


# We also need to write functions for the log prior.  Because of all the parameters and the informative nature of the hyperparameters on the other parameters, this prior is far messier than the ones we are used to.  Furthermore, since we are using PTMCMC for model selection, we need to ensure that the prior is properly normalized.

# In[24]:

@numba.jit(nopython=True)
def log_prior(p, n_divisions, a_range, bk_range, sigma_a_range, sigma_bk_range,
              beta_range):
    """
    Properly normalized log prior.
    """
    # Everything must be positive
    if (p <= 0).any():
        return -np.inf

    # Unpack parameters
    a_div, bk_div, a_indiv, bk_indiv, sigma_a_indiv, sigma_bk_indiv,                 a, bk, sigma_a, sigma_bk, beta = unpack_prior(p, n_divisions)

    # Check bounds on parameters
    if beta > beta_range[1] or beta < beta_range[0]:
        return -np.inf
    if a > a_range[1] or a < a_range[0]         or (a_indiv > a_range[1]).any() or (a_indiv < a_range[0]).any()         or (a_div > a_range[1]).any() or (a_div < a_range[0]).any():
        return -np.inf
    if bk > bk_range[1] or bk < bk_range[0]         or (bk_indiv > bk_range[1]).any() or (bk_indiv < bk_range[0]).any()         or (bk_div > bk_range[1]).any() or (bk_div < bk_range[0]).any():
        return -np.inf
    if sigma_a > sigma_a_range[1] or sigma_a < sigma_a_range[0]:
        return -np.inf
    if sigma_bk > sigma_bk_range[1] or sigma_bk < sigma_bk_range[0]:
        return -np.inf
    if (sigma_a_indiv > sigma_a_range[1]).any() or             (sigma_a_indiv < sigma_a_range[0]).any():
        return -np.inf
    if (sigma_bk_indiv > sigma_bk_range[1]).any() or             (sigma_bk_indiv < sigma_bk_range[0]).any():
        return -np.inf

    # Total number of divisions
    n_tot_divisions = n_divisions.sum()

    # Total number of cells
    n_cells = len(n_divisions)

    # Compute prior, first with uninformative parts
    log_prior = -np.log(a_range[1] - a_range[0])
    log_prior -= np.log(bk_range[1] - bk_range[0])
    log_prior -= np.log(sigma_a * np.log(sigma_a_range[1] / sigma_a_range[0]))
    log_prior -= np.log(sigma_bk * np.log(sigma_bk_range[1] / sigma_bk_range[0]))
    log_prior -= np.log(beta * np.log(beta_range[1] / beta_range[0]))

    # Compute informative parts from hierarchical model
    log_prior -= (n_cells + n_tot_divisions) * np.log(2 * np.pi)
    log_prior -= n_cells * np.log(sigma_a)
    log_prior -= n_cells * np.log(sigma_bk)
    log_prior -= np.sum((a_indiv - a)**2 / 2 / sigma_a**2)
    log_prior -= np.sum((bk_indiv - bk)**2 / 2 / sigma_bk**2)
    log_prior -= np.sum(n_divisions * np.log(sigma_a_indiv))
    log_prior -= np.sum(n_divisions * np.log(sigma_bk_indiv))

    ind_a = 0
    ind_bk = n_tot_divisions
    for i, n_div in enumerate(n_divisions):
        a_vals = a_div[ind_a:ind_a+n_div]
        log_prior -= np.sum((a_vals - a_indiv[i])**2 / 2 / sigma_a_indiv[i]**2)

        bk_vals = bk_div[ind_bk:ind_bk+n_div]
        log_prior -= np.sum((bk_vals - bk_indiv[i])**2 / 2 / sigma_bk_indiv[i]**2)

        ind_a += n_div
        ind_bk += n_div

    return log_prior


# We now need to specify the ranges of the parameters to input into the prior.  We will take $10^{-3} < \beta < 1$ µm$^2$.  We choose the lower bound to be roughly the area corresponding to a single pixel (the smallest we can resolve), and the upper bound is absurdly high.  For the bounds on the intercept, we take our smallest and largest bacterium that we used in outlier detection, $1 < a < 3.5$ µm$^2$.  We take $0.01 < \sigma_{a}, \sigma_{ai} < 2$ µm$^2$.  For the bounds on the $b$'s and $\sigma_b$'s, we consider the slowest and fastest growth rates we might expect.  The fastest known bacterial growth rate is a division every 12 minutes, which would amount to growth of about 1 µm$^2$ in 12 minutes for *Caulobacter*.  For slow growth, we take the entire duration of the experiment, which was about a week, or 10$^5$ minutes.  So, we take  $10^{-5} < b, \sigma_{b}, \sigma_{bi} < 0.1$ µm$^2$/min.

# In[25]:

# Specify bounds on parameters
a_range = (1, 3.5)
b_range = (1e-5, 0.1)
beta_range = (1e-3, 1)
sigma_a_range = (0.01, 2)
sigma_b_range = (1e-5, 0.1)


# We will start our MCMC runs near the MAP.  To find a starting point near the MAP, we take the following strategy.
#
# 1. Guess the most probable values of the $\sigma_i$, $\sigma$, and $\beta$.
# 2. Fit each individual growth curve ignoring the hyperparameters to get an estimate for the most probable values of $a_{ij}$ and $b_{ij}$.
# 3. Estimate the most probable $b_i$ as the mean of the $b_{ij}$.
# 4. Estimate the most probable $b$ as the mean of the $b_i$.
#
# To use optimization, we first need to define a function that is the negative log likelihood for a single curve.

# In[26]:

@numba.jit(nopython=True)
def neg_log_likelihood_single_curve_linear(p, t, area, beta):
    return -log_likelihood_single_curve_linear(p[0], p[1], t, area, beta)

@numba.jit(nopython=True)
def neg_log_likelihood_single_curve_exp(p, t, area, beta):
    return -log_likelihood_single_curve_exp(p[0], p[1], t, area, beta)


# Now we can do our optimization calculations.

# In[27]:

# Guess beta
beta0 = 0.09   # µm^2

# Guess sigmas
sigma_a0 = 0.1  # µm^2
sigma_b0 = 0.001  # µm^2/min
sigma_a_indiv0 = sigma_a0 * np.ones(len(list_of_df_lists))
sigma_b_indiv0 = sigma_b0 * np.ones(len(list_of_df_lists))

# Perform linear regression on each curve
a_div0 = []
b_div0 = []
for i, df_list in enumerate(list_of_df_lists):
    a_div0.append(np.empty(n_divisions[i]))
    b_div0.append(np.empty(n_divisions[i]))
    for j, df in enumerate(df_list):
        args = (df.t.values, df.area.values, beta0)
        p0 = np.array([1.5, 1 / 100])
        res = scipy.optimize.minimize(neg_log_likelihood_single_curve_linear,
                                      p0, args=args, method='powell')
        a_div0[i][j] = res.x[0]
        b_div0[i][j] = res.x[1]

# Compute the a's for each bacterium
a_indiv0 = np.empty(len(a_div0))
for i, a_ in enumerate(a_div0):
    a_indiv0[i] = a_.mean()

# Compute the b's for each bacterium
b_indiv0 = np.empty(len(b_div0))
for i, b_ in enumerate(b_div0):
    b_indiv0[i] = b_.mean()

# Compute hyperhyper a and b
a0 = a_indiv0.mean()
b0 = b_indiv0.mean()

# Put parameters together as initial guess for big problem
p0 = []
for a_div0_ in a_div0:
    p0 += list(a_div0_)
for b_div0_ in b_div0:
    p0 += list(b_div0_)
p0 += list(a_indiv0)
p0 += list(b_indiv0)
p0 += list(sigma_a_indiv0)
p0 += list(sigma_b_indiv0)
p0 += [a0, b0, sigma_a0, sigma_b0, beta0]
p0 = np.array(p0)



# Sampler setup
n_walkers = 1000
n_burn = 10000
n_steps = 2000
n_temps = 20
n_store = 50

# Arguments for prior and likelihood
logpargs = (n_divisions, a_range, b_range, sigma_a_range, sigma_b_range,
              beta_range)
loglargs = (t, areas, inds, n_divisions)

p0_walkers = np.empty((n_temps, n_walkers, len(p0)))
for i in range(n_temps):
    for j in range(n_walkers):
        p0_walkers[i, j,:] = p0 + 0.05 * p0 * np.random.uniform(-1, 1, len(p0))

cols = ['a_{0:d}_{1:d}'.format(i, j) for i in range(len(list_of_df_lists))
                                        for j in range(n_divisions[i])]
cols += ['b_{0:d}_{1:d}'.format(i, j) for i in range(len(list_of_df_lists))
                                        for j in range(n_divisions[i])]
cols += ['a_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['b_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['sigma_a_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['sigma_b_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['a', 'b', 'sigma_a', 'sigma_b', 'beta_cauchy']




def get_mcmc_results(mcmc_file, lnZ_file, log_like_fun, logpargs, loglargs,
                     p0_walkers, cols, n_walkers=1000, n_burn=1000, n_steps=1000, n_temps=50, n_store=50, thin=10):
    """
    Either load results or perform MCMC.
    """
    try:
        # Load cold chain
        df_mcmc = pd.read_csv(mcmc_file)

        # Load what we need to compute lnZ
        df_lnZ = pd.read_csv(lnZ_file)
        lnZ_val = lnZ(df_lnZ)
    except:
        # Report progress
        print('Computing burn-in and steps 0 through {0:d}...'.format(n_store),
              flush=True)

        # Run burn-in and first set of samples
        df_mcmc, p0_walkers = bebi103.run_pt_emcee(
            log_like_fun, log_prior, n_burn, n_store, n_temps=n_temps,
            p0=p0_walkers, columns=cols, loglargs=loglargs, logpargs=logpargs,
            return_pos=True, thin=thin)

        with open(mcmc_file, 'w') as f, open(lnZ_file, 'w') as fz:
            # Store cold chain
            inds = df_mcmc['beta_ind'] == 0
            df_mcmc[inds].to_csv(f, float_format='%.6e', index=False)

            # Store what we need to compute lnZ
            df_mcmc[['beta_ind', 'beta', 'lnlike']].to_csv(
                fz, float_format='%.6e', index=False)

            # Keep count of how many samples we've gotten
            n_samples = n_store

            # Run the rest
            while n_samples < n_steps:
                print('Computing steps {0:d} through {1:d}...'.format(
                        n_samples, n_samples + n_store), flush=True)

                # Run burn-in and first set of samples
                df_mcmc, p0_walkers = bebi103.run_pt_emcee(
                    log_like_fun, log_prior, 0, n_store, n_temps=n_temps,
                    p0=p0_walkers, columns=cols, loglargs=loglargs,
                    logpargs=logpargs, return_pos=True, thin=thin)

                # Store cold chain
                inds = df_mcmc['beta_ind'] == 0
                df_mcmc[inds].to_csv(f, float_format='%.6e', index=False,
                                     mode='a', header=False)

                # Store what we need to compute lnZ
                df_mcmc[['beta_ind', 'beta', 'lnprob']].to_csv(
                    fz, float_format='%.6e', index=False, mode='a',
                    header=False)

                # Keep count of how many samples we've gotten
                n_samples += n_store

        # Load cold chain
        df_mcmc = pd.read_csv(mcmc_file)

        # Load what we need to compute lnZ
        df_lnZ = pd.read_csv(lnZ_file)
        lnZ_val = bebi103.lnZ(df_lnZ)

    return df_mcmc, lnZ_val


# In[ ]:

# Specify bounds on parameters
kappa_range = np.array(b_range) / np.array(a_range)
sigma_kappa_range = sigma_b_range


# Next, we'll get our guess for the MAP.

# In[ ]:

# Guess sigmas
sigma_kappa0 = 0.001  # µm^2/min
sigma_kappa_indiv0 = sigma_kappa0 * np.ones(len(list_of_df_lists))

# Perform exponential regression on each curve
a0 = []
kappa_div0 = []
for i, df_list in enumerate(list_of_df_lists):
    a0.append(np.empty(n_divisions[i]))
    kappa_div0.append(np.empty(n_divisions[i]))
    for j, df in enumerate(df_list):
        args = (df.t.values, df.area.values, beta0)
        p0 = np.array([1.5, 1 / 100])
        res = scipy.optimize.minimize(neg_log_likelihood_single_curve_exp,
                                      p0, args=args, method='powell')
        a_div0[i][j] = res.x[0]
        kappa_div0[i][j] = res.x[1]

# Compute the a's for each bacterium
a_indiv0 = np.empty(len(a_div0))
for i, a_ in enumerate(a_div0):
    a_indiv0[i] = a_.mean()

# Compute the b's for each bacterium
kappa_indiv0 = np.empty(len(kappa_div0))
for i, kappa_ in enumerate(kappa_div0):
    kappa_indiv0[i] = kappa_.mean()

# Compute hyperhyper a and b
a0 = a_indiv0.mean()
kappa0 = kappa_indiv0.mean()

# Put parameters together as initial guess for big problem
p0 = []
for a_div0_ in a_div0:
    p0 += list(a_div0_)
for kappa_div0_ in kappa_div0:
    p0 += list(kappa_div0_)
p0 += list(a_indiv0)
p0 += list(kappa_indiv0)
p0 += list(sigma_a_indiv0)
p0 += list(sigma_kappa_indiv0)
p0 += [a0, kappa0, sigma_a0, sigma_kappa0, beta0]
p0 = np.array(p0)


# Now let's set up our sampler.

# In[ ]:

# Arguments for prior and likelihood
logpargs = (n_divisions, a_range, kappa_range, sigma_a_range, sigma_kappa_range,
            beta_range)
loglargs = (t, areas, inds, n_divisions)

p0_walkers = np.empty((n_temps, n_walkers, len(p0)))
for i in range(n_temps):
    for j in range(n_walkers):
        p0_walkers[i, j,:] = p0 + 0.05 * p0 * np.random.uniform(-1, 1, len(p0))

cols = ['a_{0:d}_{1:d}'.format(i, j) for i in range(len(list_of_df_lists))
                                        for j in range(n_divisions[i])]
cols += ['kappa_{0:d}_{1:d}'.format(i, j) for i in range(len(list_of_df_lists))
                                        for j in range(n_divisions[i])]
cols += ['a_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['kappa_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['sigma_a_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['sigma_kappa_{0:d}'.format(i) for i in range(len(list_of_df_lists))]
cols += ['a', 'kappa', 'sigma_a', 'sigma_kappa', 'beta_cauchy']


# And now let's let PTMCMC do its thing!

# In[ ]:

# File names
mcmc_file = 'hw9_mcmc_exp.csv'
lnZ_file = 'hw9_mcmc_exp_lnZ.csv'

# Let 'er rip
print('\n\nPerforming nonlinear PTMCMC....')
df_mcmc, lnZ_exp = get_mcmc_results(
    mcmc_file, lnZ_file, log_likelihood_exp, logpargs, loglargs, p0_walkers,
    cols,  n_walkers=n_walkers, n_burn=n_burn, n_steps=n_steps, n_temps=n_temps)
