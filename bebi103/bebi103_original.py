"""
Utilities for Caltech BE/Bi 103.

Author: Justin Bois
"""

import collections
import random
import warnings

import matplotlib.path as path
import numpy as np
import pandas as pd
import scipy.odr
import scipy.stats as st
import statsmodels.tools.numdiff as smnd

import skimage.io
import skimage.measure

import emcee

import bokeh.models
import bokeh.palettes
import bokeh.plotting
import seaborn as sns


# ########################################################################## #
#                     COLOR CONVERSION UTILITIES                             #
# ########################################################################## #
def rgb_frac_to_hex(rgb_frac):
    """
    Convert fractional RGB values to hexidecimal color string.

    Parameters
    ----------
    rgb_frac : array_like, shape (3,)
        Fractional RGB values; each entry is between 0 and 1.

    Returns
    -------
    str
        Hexidecimal string for the given RGB color.

    Examples
    --------
    >>> rgb_frac_to_hex((0.65, 0.23, 1.0))
    '#a53aff'

    >>> rgb_frac_to_hex((1.0, 1.0, 1.0))
    '#ffffff'
    """

    if len(rgb_frac) != 3:
        raise RuntimeError('`rgb_frac` must have exactly three entries.')

    if (np.array(rgb_frac) < 0).any() or (np.array(rgb_frac) > 1).any():
        raise RuntimeError('RGB values must be between 0 and 1.')

    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb_frac[0] * 255),
                                           int(rgb_frac[1] * 255),
                                           int(rgb_frac[2] * 255))


def data_to_hex_color(x, palette, x_range=[0, 1], na_value='#000000'):
    """
    Convert a value to a hexidecimal color according to
    color palette.

    Parameters
    ----------
    x : float or int
        Value to be converted to hexidecimal color.
    palette : list of 3-tuples
        Color palette as returned from seaborn.color_palette().
        List of 3-tuples containing fractional RGB values.
    x_range : array_list, shape (2,), default = [0, 1]
        Low and high value of the range of values `x` may
        assume.

    Returns
    -------
    str
        Hexidecimal string.

    Examples
    --------
    >>> data_to_hex_color(0.7, sns.colorpalette())
    '#ccb974'

    >>> data_to_hex_color(7.1, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [0, 10])
    '#0000ff'
    """
    if x is None or np.isnan(x):
        return na_value
    elif x > x_range[1] or x < x_range[0]:
        raise RuntimeError('data outside of range')
    elif x == x_range[1]:
        return rgb_frac_to_hex(palette[-1])

    # Fractional position of x in x_range
    f = (x - x_range[0]) / (x_range[1] - x_range[0])

    return rgb_frac_to_hex(palette[int(f * len(palette))])


def im_merge_cmy(im_cyan, im_magenta, im_yellow=None):
    """
    Merge channels to make RGB image that has cyan, magenta, and
    yellow.

    Parameters
    ----------
    im_cyan: array_like
        Image represented in cyan channel.  Must be same shape
        as `im_magenta` and `im_yellow`.
    im_magenta: array_like
        Image represented in magenta channel.  Must be same shape
        as `im_yellow` and `im_yellow`.
    im_yellow: array_like
        Image represented in yellow channel.  Must be same shape
        as `im_cyan` and `im_magenta`.

    Returns
    -------
    output : array_like, dtype float, shape (*im_cyan.shape, 3)
        RGB image the give CMY coloring of image

    Notes
    -----
    ..  All input images are streched so that their pixel intensities
        go from 0 to 1.
    """
    im_cyan_scaled = \
        (im_cyan - im_cyan.min()) / (im_cyan.max() - im_cyan.min())
    im_magenta_scaled = \
        (im_magenta - im_magenta.min()) / (im_magenta.max() - im_magenta.min())

    if im_yellow is None:
        im_yellow_scaled = np.zeros_like(im_cyan)
    else:
        im_yellow_scaled = \
            (im_yellow - im_yellow.min()) / (im_yellow.max() - im_yellow.min())

    # Convert images to RGB with magenta, cyan, and yellow channels
    im_cyan_scaled_rgb = np.dstack((np.zeros_like(im_cyan_scaled),
                                    im_cyan_scaled,
                                    im_cyan_scaled))
    im_magenta_scaled_rgb = np.dstack((im_magenta_scaled,
                                       np.zeros_like(im_magenta_scaled),
                                       im_magenta_scaled))
    im_yellow_scaled_rgb = np.dstack((im_yellow_scaled,
                                      im_yellow_scaled,
                                      np.zeros_like(im_yellow_scaled)))

    # Merge together
    merged_image = \
        im_cyan_scaled_rgb + im_magenta_scaled_rgb + im_yellow_scaled_rgb

    # Scale each channel to be between zero and 1
    merged_image[:, :, 0] /= merged_image[:, :, 0].max()
    merged_image[:, :, 1] /= merged_image[:, :, 1].max()
    merged_image[:, :, 2] /= merged_image[:, :, 2].max()

    return merged_image


def bokeh_imshow(im, color_mapper=None, plot_height=400, 
                 length_units='pixels', interpixel_distance=1.0):
    """
    Display an image in a Bokeh figure.
    
    Parameters
    ----------
    im : 2-dimensional Numpy array
        Intensity image to be displayed.
    color_mapper : bokeh.models.LinearColorMapper instance, default None
        Mapping of intensity to color. Default is 256-level Viridis.
    plot_height : int
        Height of the plot in pixels. The width is scaled so that the 
        x and y distance between pixels is the same.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
        
    Returns
    -------
    output : bokeh.plotting.figure instance
        Bokeh plot with image displayed.
    """
    # Get shape, dimensions
    n, m = im.shape
    dw = m * interpixel_distance
    dh = n * interpixel_distance
    
    # Set up figure with appropriate dimensions
    plot_width = int(m/n * plot_height)
    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width, 
                              x_range=[0, dw], y_range=[0, dh], x_axis_label=length_units,
                              y_axis_label=length_units,
                              tools='pan,box_zoom,wheel_zoom,reset,resize')

    # Set color mapper; we'll do Viridis with 256 levels by default
    if color_mapper is None:
        color_mapper = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))

    # Display the image
    im_bokeh = p.image(image=[im[::-1,:]], x=0, y=0, dw=dw, dh=dh, 
                       color_mapper=color_mapper)
    
    return p

# ########################################################################## #
#                            MCMC UTILITIES                                  #
# ########################################################################## #
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


def extract_1d_hist(samples, nbins=100, density=True):
    """
    Compute a 1d histogram with x-values at bin centers.
    Meant to be used with MCMC samples.

    Parameters
    ----------
    samples : array
        1D array of MCMC samples
    nbins : int
        Number of bins in histogram
    density : bool, optional
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

    Returns
    -------
    count : array, shape (nbins,)
        The counts, appropriately weighted depending on the
        `density` kwarg, for the histogram.
    x : array, shape (nbins,)
        The positions of the bin centers.
    """

    # Obtain histogram
    count, bins = np.histogram(trace, bins=nbins, density=density)

    # Make the bins into the bin centers, not the edges
    x = (bins[:-1] + bins[1:]) / 2.0

    return count, x


def extract_2d_hist(samples_x, samples_y, bins=100, density=True,
                    meshgrid=False):
    """
    Compute a 2d histogram with x,y-values at bin centers.
    Meant to be used with MCMC samples.

    Parameters
    ----------
    samples_x : array
        1D array of MCMC samples for x-axis
    samples_y : array
        1D array of MCMC samples for y-axis
    bins : int
        Number of bins in histogram. The same binning is
        used in the x and y directions.
    density : bool, optional
        If False, the result will contain the number of samples
        in each bin.  If True, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
    meshgrid : bool, options
        If True, the returned `x` and `y` arrays are two-dimensional
        as constructed with np.meshgrid().  If False, `x` and `y`
        are returned as 1D arrays.

    Returns
    -------
    count : array, shape (nbins, nbins)
        The counts, appropriately weighted depending on the
        `density` kwarg, for the histogram.
    x : array, shape either (nbins,) or (nbins, nbins)
        The positions of the bin centers on the x-axis.
    y : array, shape either (nbins,) or (nbins, nbins)
        The positions of the bin centers on the y-axis.
    """
    # Obtain histogram
    count, x_bins, y_bins = np.histogram2d(samples_x, samples_y, bins=bins,
                                           normed=density)

    # Make the bins into the bin centers, not the edges
    x = (x_bins[:-1] + x_bins[1:]) / 2.0
    y = (y_bins[:-1] + y_bins[1:]) / 2.0

    # Make mesh grid out of x_bins and y_bins
    if meshgrid:
        y, x = np.meshgrid(x, y)

    return count.transpose(), x, y


def norm_cumsum_2d(sample_x, sample_y, bins=100, meshgrid=False):
    """
    Returns 1 - the normalized cumulative sum of two sets of samples.

    Parameters
    ----------
    samples_x : array
        1D array of MCMC samples for x-axis
    samples_y : array
        1D array of MCMC samples for y-axis
    bins : int
        Number of bins in histogram. The same binning is
        used in the x and y directions.
    meshgrid : bool, options
        If True, the returned `x` and `y` arrays are two-dimensional
        as constructed with np.meshgrid().  If False, `x` and `y`
        are returned as 1D arrays.

    Returns
    -------
    norm_cumsum : array, shape (nbins, nbins)
        1 - the normalized cumulative sum of two sets of samples.
        I.e., an isocontour on this surface at level alpha encompasses
        a fraction alpha of the total probability.
    x : array, shape either (nbins,) or (nbins, nbins)
        The positions of the bin centers on the x-axis.
    y : array, shape either (nbins,) or (nbins, nbins)
        The positions of the bin centers on the y-axis.

    Notes
    -----
    .. To make a contour plot with contour lines drawn to contain
       68.27, 95.45, and 99.73% of the total probability, use the
       output of this function as:
       plt.contourf(x, y, norm_cumsum, levels=(0.6827, 0.9545, 0.9973))
    """

    # Compute the histogram
    count, x, y = extract_2d_hist(sample_x, sample_y, bins=bins,
                                  density=False, meshgrid=meshgrid)
    # Remember the shape
    shape = count.shape
    count = count.ravel()

    # Inverse sort the histogram
    isort = np.argsort(count)[::-1]
    unsort = np.argsort(isort)

    # Compute the cumulative sum and normalize
    count_cumsum = count[isort].cumsum()
    count_cumsum /= count_cumsum[-1]

    # Normalized, reshaped cumulative sum
    return count_cumsum[unsort].reshape(shape), x, y


def hpd(trace, mass_frac):
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
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
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n - n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int + n_samples]])


# ########################################################################## #
#                    IMAGE PROCESSING UTILITIES                              #
# ########################################################################## #

class SimpleImageCollection(object):
    """
    Load a collection of images.

    Parameters
    ----------
    load_pattern : string or list
        If string, uses glob to generate list of files containing
        images. If list, this is the list of files containing images.
    load_func : callable, default skimage.io.imread
        Function to be called to load images.
    conserve_memory : bool, default True
        If True, do not load all images into RAM. If False, load
        all into a list.

    Returns
    -------
    ic : SimpleImageCollection instance
        ic[n] gives image n of the image collection.

    Notes
    -----
    .. Any keyword arguments except those listed above are passed into
    load_func as kwargs.
    .. This is a much simplified (and therefore faster) version of
    skimage.io.ImageCollection.
    """

    def __init__(self, load_pattern, load_func=skimage.io.imread,
                 conserve_memory=True, **load_func_kwargs):
        if isinstance(load_pattern, str):
            self.fnames = glob.glob(load_pattern)
        else:
            self.fnames = load_pattern

        self.conserve_memory = conserve_memory

        if self.conserve_memory:
            self.load_func = load_func
            self.kwargs = load_func_kwargs
        else:
            self.ims = [load_func(f, **load_func_kwargs) for f in self.fnames]



    def __getitem__(self, n):
        """
        Return selected image.
        """
        if self.conserve_memory:
            return self.load_func(self.fnames[n], **self.load_func_kwargs)
        else:
            return self.ims[n]


def simple_image_collection(im_glob, load_func=skimage.io.imread,
                            conserve_memory=True, **load_func_kwargs):
    """
    Load a collection of images.

    Parameters
    ----------
    load_pattern : string or list
        If string, uses glob to generate list of files containing
        images. If list, this is the list of files containing images.
    load_func : callable, default skimage.io.imread
        Function to be called to load images.
    conserve_memory : bool, default True
        If True, do not load all images into RAM. If False, load
        all into a list.

    Returns
    -------
    ic : SimpleImageCollection instance
        ic[n] gives image n of the image collection.

    Notes
    -----
    .. Any keyword arguments except those listed above are passed into
    load_func as kwargs.
    .. This is a much simplified (and therefore faster) version of
    skimage.io.ImageCollection.
    """
    return SimpleImageCollection(im_glob, load_func=load_func,
                                 conserve_memory=conserve_memory,
                                 **load_func_kwargs)

def rgb_to_rgba32(im, flip=True):
    """
    Convert an RGB image to a 32 bit-encoded RGBA image.

    Parameters
    ----------
    im : nd_array, shape (m, n, 3)
        Input m by n RGB image.
    flip : bool, default True
        If True, up-down flit the image. This is useful
        for display with Bokeh.

    Returns
    -------
    output : nd_array, shape (m, n), dtype int32
        RGB image encoded as 32-bit integers.

    Notes
    -----
    .. The input image is converted to 8-bit and then encoded
       as 32-bit. The main use for this function is encoding images
       for display with Bokeh, so this data loss is ok.
    """
    # Ensure it has three channels
    if len(im.shape) != 3 or im.shape[2] !=3:
        raise RuntimeError('Input image is not RGB.')

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    im_8 = skimage.img_as_ubyte(im)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.dstack((im_8, 255*np.ones_like(im_8[:,:,0])))

    # Reshape into 32 bit. Must flip up/down for proper orientation
    return np.flipud(im_rgba.view(dtype=np.int32).reshape(n, m))


def verts_to_roi(verts, size_i, size_j):
    """
    Converts list of vertices to an ROI and ROI bounding box

    Parameters
    ----------
    verts : array_like, shape (n_verts, 2)
        List of vertices of a polygon with no crossing lines.  The units
        describing the positions of the vertices are interpixel spacing.
    size_i : int
        Number of pixels in the i-direction (number of rows) in
        the image
    size_j : int
        Number of pixels in the j-direction (number of columns) in
        the image

    Returns
    -------
    roi : array_like, Boolean, shape (size_i, size_j)
        roi[i,j] is True if pixel (i,j) is in the ROI.
        roi[i,j] is False otherwise
    roi_bbox : tuple of slice objects
        To get a subimage with the bounding box of the ROI, use
        im[roi_bbox].
    roi_box : array_like, shape is size of bounding box or ROI
        A mask for the ROI with the same dimension as the bounding
        box.  The indexing starts at zero at the upper right corner
        of the box.
    """

    # Make list of all points in the image in units of pixels
    i = np.arange(size_i)
    j = np.arange(size_j)
    ii, jj = np.meshgrid(j, i)
    pts = np.array(list(zip(ii.ravel(), jj.ravel())))

    # Make a path object from vertices
    p = path.Path(verts)

    # Get list of points that are in roi
    in_roi = p.contains_points(pts)

    # Convert it to an image
    roi = in_roi.reshape((size_i, size_j)).astype(np.bool)

    # Get bounding box of ROI
    regions = skimage.measure.regionprops(roi.astype(np.int))
    bbox = regions[0].bbox
    roi_bbox = np.s_[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1]

    # Get ROI mask for just within bounding box
    roi_box = roi[roi_bbox]

    # Return boolean in same shape as image
    return (roi, roi_bbox, roi_box)


class CostesColocalization(object):
    """
    Generic class just to store attributes
    """

    def __init__(self, **kw):
        self.__dict__ = kw


def costes_coloc(im_1, im_2, psf_width=3, n_scramble=1000, thresh_r=0.0,
                 roi=None, roi_method='all', do_manders=True):
    """
    Perform Costes colocalization analysis on a pair of images.

    Parameters
    ----------
    im_1: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_1`.
    im_2: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_2`.
    psf_width: int, default 3
        Width, in pixels of the point spread function.
    n_scramble: int, default 1000
        Number of strambled image comparisons to do to get statistics.
    thresh_r: float, default 0.0
        Threshold Pearson r value to be considered colocalized.
    roi: array_like, dtype bool, default None
        Boolean image the same shape as `im_1` and `im_2` that
        is True for pixels within the ROI.
    roi_method: str, default 'all'
        If 'all', all pixels of a given subimage must be within
        the ROI for the subimage itself to be considered part
        of the ROI.  If 'any', if any one pixel is within the ROI,
        the subimage is considered part of the ROI.
    do_manders: bool, default True
        If True, compute the Manders coefficients.

    Returns
    -------
    output: A CostesColocalization instance.
        The CostesColocalization instance has the following attributes.
            im_1, im_2, psf_width, n_scramble, thresh_r, roi,
                roi_method: As in the input parameters.
            a: slope of the regression line I_2 = a * I_1 + b
            b: intercept of regression line I_2 = a * I_1 + b
            M_1: Manders coefficient for image 1
            M_2: Manders coefficient for image 2
            pearson_r: Pearson coerrelaction coefficient of the pixels
                in the two images.
            p_coloc: The probability of colocalization being present
                in the two images.
    """

    # Make mirrored boundaries in preparation for scrambling
    im_1_mirror = mirror_edges(im_1, psf_width)
    im_2_mirror = mirror_edges(im_2, psf_width)

    # Set up ROI
    if roi is None:
        roi = np.ones_like(im_1, dtype='bool')

    # Rename images to be sliced ROI
    im_1 = im_1[roi]
    im_2 = im_2[roi]

    # Mirror ROI at edges
    roi_mirror = mirror_edges(roi, psf_width)

    # Compute the blocks that we'll scramble
    blocks_1 = im_to_blocks(im_1_mirror, psf_width, roi_mirror, roi_method)
    blocks_2 = im_to_blocks(im_2_mirror, psf_width, roi_mirror, roi_method)

    # Flatten second list of blocks for Pearson calculations
    blocks_2_flat = np.array(blocks_2).flatten()

    # Compute the Pearson coefficient
    pearson_r, _ = st.pearsonr(np.array(blocks_1).ravel(), blocks_2_flat)

    # Do image scrambling and r calculations
    r_scr = np.empty(n_scramble)
    for i in range(n_scramble):
        random.shuffle(blocks_1)
        r, _ = scipy.stats.pearsonr(np.array(blocks_1).ravel(), blocks_2_flat)
        r_scr[i] = r

    # Compute percent chance of coloc
    p_coloc = (r_scr < pearson_r).sum() / n_scramble

    # Now do work to compute adjusted Manders's coefficients
    if do_manders:
        # Get the linear relationship between im_2 and im_1
        a, b = odr_linear(im_1.ravel(), im_2.ravel())

        # Perform threshold calculation
        thresh_1 = find_thresh(im_1, im_2, a, b, thresh_r=thresh_r)
        thresh_2 = a * thresh_1 + b

        # Compute Costes's update to the Manders's coefficients
        inds = (im_1 > thresh_1) & (im_2 > thresh_2)
        M_1 = im_1[inds].sum() / im_1.sum()
        M_2 = im_2[inds].sum() / im_2.sum()

        # Toss results into class for returning
        return CostesColocalization(
            im_1=im_1, im_2=im_2, roi=roi, roi_method=roi_method,
            psf_width=psf_width, n_scramble=n_scramble, thresh_r=thresh_r,
            thresh_1=thresh_1, thresh_2=thresh_2, a=a, b=b, M_1=M_1,
            M_2=M_2, r_scr=r_scr, pearson_r=pearson_r, p_coloc=p_coloc)
    else:
        return CostesColocalization(
            im_1=im_1, im_2=im_2, roi=roi, roi_method=roi_method,
            psf_width=psf_width, n_scramble=n_scramble, thresh_r=None,
            thresh_1=None, thresh_2=None, a=None, b=None, M_1=None,
            M_2=None, r_scr=r_scr, pearson_r=pearson_r, p_coloc=p_coloc)


def odr_linear(x, y, intercept=None, beta0=None):
    """
    Performs orthogonal linear regression on x, y data.

    Parameters
    ----------
    x: array_like
        x-data, 1D array.  Must be the same lengths as `y`.
    y: array_like
        y-data, 1D array.  Must be the same lengths as `x`.
    intercept: float, default None
        If not None, fixes the intercept.
    beta0: array_like, shape (2,)
        Guess at the slope and intercept, respectively.

    Returns
    -------
    output: ndarray, shape (2,)
        Array containing slope and intercept of ODR line.
    """

    def linear_fun(p, x):
        return p[0] * x + p[1]

    def linear_fun_fixed(p, x):
        return p[0] * x + intercept

    # Set the model to be used for the ODR fitting
    if intercept is None:
        model = scipy.odr.Model(linear_fun)
        if beta0 is None:
            beta0 = (0.0, 1.0)
    else:
        model = scipy.odr.Model(linear_fun_fixed)
        if beta0 is None:
            beta0 = (1.0,)

    # Make a Data instance
    data = scipy.odr.Data(x, y)

    # Instantiate ODR
    odr = scipy.odr.ODR(data, model, beta0=beta0)

    # Perform ODR fit
    try:
        result = odr.run()
    except scipy.odr.odr_error:
        raise scipy.odr.odr_error('ORD failed.')

    return result.beta


def find_thresh(im_1, im_2, a, b, thresh_r=0.0):
    """
    Find the threshold pixel intensity of `im_1` where
    the Pearson correlation between the images goes below `thresh_r`.

    Parameters
    ----------
    im_1: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_1`.
    im_2: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_2`.
    a: float
        Slope of the ORD regression of `im_2` vs. `im_1`.
    b: float
        Intercept of the ORD regression of `im_2` vs. `im_1`.
    thresh_r: float, default 0.0
        Threshold Pearson correlation

    Returns
    -------
    output: int or float
        The threshold pixel intensity for colocalization
        (see notes below).

    Notes
    -----
    ..  To determine which pixels are colocalized in two images, we
        do the following:
            1. Perform a regression based on all points of to give
               I_2 = a * I_1 + b.
            2. Define T = I_1.max().
            3. Compute the Pearson r value considering all pixels with
               I_1 < T and I_2 < a * T + b.
            4. If r <= thresh_r decrement T and goto 3.  Otherwise,
               save $T_1 = T$ and $T_2 = a * T + b.
            5. Pixels with I_2 > T_2 and I_1 > T_1 are colocalized.
        This function returns T.
    """
    if im_1.dtype not in [np.uint16, np.uint8]:
        incr = (im_1.max() - im_1.min()) / 256.0
    else:
        incr = 1

    thresh_max = im_1.max()
    thresh_min = im_1.min()
    thresh = thresh_max
    r = pearsonr_below_thresh(thresh, im_1, im_2, a, b)
    min_r = r
    min_thresh = thresh
    while thresh > thresh_min and r > thresh_r:
        thresh -= incr
        r = pearsonr_below_thresh(thresh, im_1, im_2, a, b)
        if min_r > r:
            min_r = r
            min_thresh = thresh

    if thresh == thresh_min:
        thresh = min_thresh

    return thresh


def pearsonr_below_thresh(thresh, im_1, im_2, a, b):
    """
    The Pearson r between two images for pixel values below
    threshold.

    Parameters
    ----------
    thresh: float or int
        The threshold value of pixel intensities to consider for
        `im_1`.
    im_1: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_1`.
    im_2: array_like
        Intensity image for colocalization.  Must be the
        same shame as `im_2`.
    a: float
        Slope of the ORD regression of `im_2` vs. `im_1`.
    b: float
        Intercept of the ORD regression of `im_2` vs. `im_1`.
    """
    inds = (im_1 <= thresh) | (im_2 <= a * thresh + b)
    r, _ = st.pearsonr(im_1[inds], im_2[inds])
    return r


def mirror_edges(im, psf_width):
    """
    Given a 2D image pads the boundaries by mirroring so that the
    dimensions of the image are multiples for the width of the
    point spread function.

    Parameters
    ----------
    im: array_like
        Image to mirror edges
    psf_width: int
        The width, in pixels, of the point spread function

    Returns
    -------
    output: array_like
        Image with mirrored edges
    """

    # How much we need to pad
    pad_i = psf_width - (im.shape[0] % psf_width)
    pad_j = psf_width - (im.shape[1] % psf_width)

    # Get widths
    pad_top = pad_i // 2
    pad_bottom = pad_i - pad_top
    pad_left = pad_j // 2
    pad_right = pad_j - pad_left

    # Do the padding
    return np.pad(im, ((pad_top, pad_bottom), (pad_left, pad_right)),
                  mode='reflect')


def im_to_blocks(im, width, roi=None, roi_method='all'):
    """
    Converts image to list of square subimages called "blocks."

    Parameters
    ----------
    im: array_like
        Image to convert to a list of blocks.
    width: int
        Width of square blocks in units of pixels.
    roi: array_like, dtype bool, default None
        Boolean image the same shape as `im_1` and `im_2` that
        is True for pixels within the ROI.
    roi_method: str, default 'all'
        If 'all', all pixels of a given subimage must be within
        the ROI for the subimage itself to be considered part
        of the ROI.  If 'any', if any one pixel is within the ROI,
        the subimage is considered part of the ROI.

    Returns
    -------
    output: list of ndarrays
        Each entry is a `width` by `width` NumPy array containing
        a block.
    """
    # Initialize ROI
    if roi is None:
        roi = np.ones_like(im)

    # Specify method for determining if in ROI or not
    if roi_method == 'all':
        roi_test = np.all
    else:
        roi_test = np.any

    # Construct list of blocks
    return [im[i:i + width, j:j + width]
            for i in range(0, im.shape[0], width)
            for j in range(0, im.shape[1], width)
            if roi_test(roi[i:i + width, j:j + width])]

# ########################################################################## #
#                        GENERAL UTILITIES                                   #
# ########################################################################## #
def ecdf(data, conventional=False, buff=0.1, min_x=None, max_x=None):
    """
    Computes the x and y values for an ECDF of a one-dimensional
    data set.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.
    conventional : bool, default False
        If True, generates x,y values for "conventional" ECDF, which
        give staircase style ECDF when plotted as plt.plot(x, y, '-').
        Otherwise, gives points x,y corresponding to the concave
        corners of the conventional ECDF, plotted as
        plt.plot(x, y, '.').
    buff : float, default 0.1
        How long the tails at y = 0 and y = 1 should extend as a
        fraction of the total range of the data. Ignored if
        `coneventional` is False.
    min_x : float, default -np.inf
        If min_x is greater than extent computed from `buff`, tail at
        y = 0 extends to min_x. Ignored if `coneventional` is False.
    max_x : float, default -np.inf
        If max_x is less than extent computed from `buff`, tail at
        y = 0 extends to max_x. Ignored if `coneventional` is False.

    Returns
    -------
    x : array_like, shape (n_data, )
        The x-values for plotting the ECDF.
    y : array_like, shape (n_data, )
        The y-values for plotting the ECDF.
    """

    # Get x and y values for data points
    x, y = np.sort(data), np.arange(1, len(data)+1) / len(data)

    if conventional:
        # Set defaults for min and max tails
        if min_x is None:
            min_x = -np.inf
        if max_x is None:
            max_x = np.inf

        # Set up output arrays
        x_conv = np.empty(2*(len(x) + 1))
        y_conv = np.empty(2*(len(x) + 1))

        # y-values for steps
        y_conv[:2] = 0
        y_conv[2::2] = y
        y_conv[3::2] = y

        # x- values for steps
        x_conv[0] = max(min_x, x[0] - (x[-1] - x[0])*buff)
        x_conv[1] = x[0]
        x_conv[2::2] = x
        x_conv[3:-1:2] = x[1:]
        x_conv[-1] = min(max_x, x[-1] + (x[-1] - x[0])*buff)

        return x_conv, y_conv

    return x, y


def approx_hess(x, f, epsilon=None, args=(), kwargs={}):
    """
    Parameters
    ----------
    x : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x, `*args`, `**kwargs`)
    epsilon : float or array-like, optional
       Stepsize used, if None, then stepsize is automatically chosen
       according to EPS**(1/4)*x.
    args : tuple
        Arguments for function `f`.
    kwargs : dict
        Keyword arguments for function `f`.


    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian


    Notes
    -----
    Equation (9) in Ridout. Computes the Hessian as::

      1/(4*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j]
                                                     - d[k]*e[k])) -
                 (f(x - d[j]*e[j] + d[k]*e[k]) - f(x - d[j]*e[j]
                                                     - d[k]*e[k]))

    where e[j] is a vector with element j == 1 and the rest are zero and
    d[i] is epsilon[i].

    References
    ----------:

    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74

    Copyright
    ---------
    This is an adaptation of the function approx_hess3() in
    statsmodels.tools.numdiff. That code is BSD (3 clause) licensed as
    follows:

    Copyright (C) 2006, Jonathan E. Taylor
    All rights reserved.

    Copyright (c) 2006-2008 Scipy Developers.
    All rights reserved.

    Copyright (c) 2009-2012 Statsmodels Developers.
    All rights reserved.


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

      a. Redistributions of source code must retain the above copyright notice,
         this list of conditions and the following disclaimer.
      b. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
      c. Neither the name of Statsmodels nor the names of its contributors
         may be used to endorse or promote products derived from this software
         without specific prior written permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    n = len(x)
    h = smnd._get_epsilon(x, 4, epsilon, n)
    ee = np.diag(h)
    hess = np.outer(h,h)

    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (f(*((x + ee[i, :] + ee[j, :],) + args), **kwargs)
                          - f(*((x + ee[i, :] - ee[j, :],) + args), **kwargs)
                          - (f(*((x - ee[i, :] + ee[j, :],) + args), **kwargs)
                          - f(*((x - ee[i, :] - ee[j, :],) + args), **kwargs))
                          )/(4.*hess[i, j])
            hess[j, i] = hess[i, j]
    return hess
