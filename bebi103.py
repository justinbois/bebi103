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

import skimage.io
import skimage.measure

import emcee

import bokeh.models
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


def data_to_hex_color(x, palette, x_range=[0, 1]):
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
    if x > x_range[1] or x < x_range[0]:
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
    im_cyan_scaled = (im_cyan - im_cyan.min()) \
                            / (im_cyan.max() - im_cyan.min())
    im_magenta_scaled = (im_magenta - im_magenta.min()) \
                            / (im_magenta.max() - im_magenta.min())

    if im_yellow is None:
        im_yellow_scaled = np.zeros_like(im_cyan)
    else:
        im_yellow_scaled = (im_yellow - im_yellow.min()) \
                                / (im_yellow.max() - im_yellow.min())

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
    merged_image = im_cyan_scaled_rgb + im_magenta_scaled_rgb \
                        + im_yellow_scaled_rgb

    # Scale each channel to be between zero and 1
    merged_image[:,:,0] /= merged_image[:,:,0].max()
    merged_image[:,:,1] /= merged_image[:,:,1].max()
    merged_image[:,:,2] /= merged_image[:,:,2].max()

    return merged_image


# ########################################################################## #
#                           BOKEH UTILITIES                                  #
# ########################################################################## #
def bokeh_matplot(df, i_col, j_col, data_col, data_range=None, n_colors=21,
                  label_ticks=True, colormap='RdBu_r', plot_width=1000,
                  plot_height=1000):
    """
    Create Bokeh plot of a matrix.

    Parameters
    ----------
    df : Pandas DataFrame
        Tidy DataFrame to be plotted as a matrix.
    i_col : hashable object
        Column in `df` to be used for row indices of matrix.
    j_col : hashable object
        Column in `df` to be used for column indices of matrix.
    data_col : hashable object
        Column containing values to be plotted.  These values
        set which color is displayed in the plot and also are
        displayed in the hover tool.
    data_range : array_like, shape (2,)
        Low and high values that data may take, used for scaling
        the color.  Default is the range of the inputted data.
    n_colors : int, default = 21
        Number of colors to be used in colormap.
    label_ticks : bool, default = True
        If False, do not put tick labels
    colormap : str, default = 'RdBu_r'
        Any of the allowed seaborn colormaps.
    plot_width : int, default 1000
        Width of plot in pixels.
    plot_height : int, default 1000
        Height of plot in pixels.

    Returns
    -------
    Bokeh plotting object


    Examples
    --------
    >>> a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    >>> data = np.array(np.unravel_index(range(9), a.shape) + (a.ravel(),)).T
    >>> df = pd.DataFrame(data, columns=['i', 'j', 'data'])
    >>> bokeh.plotting.output_file('test_matplot.html')
    >>> p = bokeh_matplot(df, i_col, j_col, data_col, n_colors=21,
                          colormap='RdBu_r', plot_width=1000,
                          plot_height=1000)
    >>> bokeh.plotting.show(p)
    """
    # Copy the DataFrame
    df_ = df.copy()

    # Convert i, j to strings so not interpreted as physical space
    df_[i_col] = df_[i_col].astype(str)
    df_[j_col] = df_[j_col].astype(str)

    # Get data range
    if data_range is None:
        data_range = (df[data_col].min(), df[data_col].max())
    elif (data_range[0] > df[data_col].min()) \
            or (data_range[1] < df[data_col].max()):
        raise RuntimeError('Data out of specified range.')

    # Get colors
    palette = sns.color_palette(colormap, n_colors)

    # Compute colors for squares
    df_['color'] = df_[data_col].apply(data_to_hex_color,
                                       args=(palette, data_range))

    # Data source
    source = bokeh.plotting.ColumnDataSource(df_)

    tools = 'reset,resize,hover,save,pan,box_zoom,wheel_zoom'

    # Set up figure; need to reverse y_range to make axis matrix index
    p = bokeh.plotting.figure(
               x_range=list(df_[j_col].unique()),
               y_range=list(reversed(list(df_[i_col].unique()))),
               x_axis_location='above', plot_width=plot_width,
               plot_height=plot_height, toolbar_location='left',
               tools=tools)

    # Populate colored squares
    p.rect(j_col, i_col, 1, 1, source=source, color='color', line_color=None)

    # Set remaining properties
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    if label_ticks:
        p.axis.major_label_text_font_size = '8pt'
    else:
        p.axis.major_label_text_color = None
        p.axis.major_label_text_font_size = '0pt'
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi/3

    # Build hover tool
    hover = p.select(dict(type=bokeh.models.HoverTool))
    hover.tooltips = collections.OrderedDict([
    ('i', '  @' + i_col),
    ('j', '  @' + j_col),
    (data_col, '  @' + data_col)])

    return p


def bokeh_boxplot(df, value, label, ylabel=None, sort=True, plot_width=650,
                  plot_height=450, box_fill_color='medium_purple'):
    """
    Make a Bokeh box plot from a tidy DataFrame.

    Parameters
    ----------
    df : tidy Pandas DataFrame
        DataFrame to be used for plotting
    value : hashable object
        Column of DataFrame containing data to be used.
    label : hashable object
        Column of DataFrame use to categorize.
    ylabel : str, default None
        Text for y-axis label
    sort : Boolean, default True
        If True, sort DataFrame by label so that x-axis labels are
        alphabetical.
    plot_width : int, default 650
        Width of plot in pixels.
    plot_height : int, default 450
        Height of plot in pixels.
    box_fill_color : string
        Fill color of boxes, default = 'medium_purple'

    Returns
    -------
    Bokeh plotting object

    Example
    -------
    >>> cats = list('ABCD')
    >>> values = np.random.randn(200)
    >>> labels = np.random.choice(cats, 200)
    >>> df = pd.DataFrame({'label': labels, 'value': values})
    >>> bokeh.plotting.output_file('test_boxplot.html')
    >>> p = bokeh_boxplot(df, value='value', label='label')
    >>> bokeh.plotting.show(p)

    Notes
    -----
    .. Based largely on example code found here:
     https://github.com/bokeh/bokeh/blob/master/examples/plotting/file/boxplot.py
    """

    # Sort DataFrame by labels for alphabetical x-labeling
    if sort:
        df_sort = df.sort_values(label)
    else:
        df_sort = df.copy()

    # Convert labels to string to allow categorical axis labels
    df_sort[label] = df_sort[label].astype(str)

    # Get the categories
    cats = list(df_sort[label].unique())

    # Group Data frame
    df_gb = df_sort.groupby(label)

    # Compute quartiles for each group
    q1 = df_gb[value].quantile(q=0.25)
    q2 = df_gb[value].quantile(q=0.5)
    q3 = df_gb[value].quantile(q=0.75)

    # Compute interquartile region and upper and lower bounds for outliers
    iqr = q3 - q1
    upper_cutoff = q3 + 1.5*iqr
    lower_cutoff = q1 - 1.5*iqr

    # Find the outliers for each category
    def outliers(group):
        cat = group.name
        outlier_inds = (group[value] > upper_cutoff[cat]) \
                                     | (group[value] < lower_cutoff[cat])
        return group[value][outlier_inds]

    # Apply outlier finder
    out = df_gb.apply(outliers).dropna()

    # Points of outliers for plotting
    outx = []
    outy = []
    if not out.empty:
        for cat in cats:
            if not out[cat].empty:
                for val in out[cat]:
                    outx.append(cat)
                    outy.append(val)

    # Shrink whiskers to smallest and largest non-outlier
    qmin = df_gb[value].min()
    qmax = df_gb[value].max()
    upper = upper_cutoff.combine(qmax, min)
    lower = lower_cutoff.combine(qmin, max)

    # Reindex to make sure ordering is right when plotting
    upper = upper.reindex(cats)
    lower = lower.reindex(cats)
    q1 = q1.reindex(cats)
    q2 = q2.reindex(cats)
    q3 = q3.reindex(cats)

    # Build figure
    p = bokeh.plotting.figure(background_fill='#DFDFE5',
                              plot_width=plot_width,
                              plot_height=plot_height, x_range=cats)
    p.ygrid.grid_line_color = 'white'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_width = 2
    p.yaxis.axis_label = ylabel

    # stems
    p.segment(cats, upper, cats, q3, line_width=2, line_color="black")
    p.segment(cats, lower, cats, q1, line_width=2, line_color="black")

    # boxes
    p.rect(cats, (q3 + q1)/2, 0.5, q3 - q1, fill_color="mediumpurple",
           alpha=0.7, line_width=2, line_color="black")

    # median (almost-0 height rects simpler than segments)
    y_range = qmax.max() - qmin.min()
    p.rect(cats, q2, 0.5, 0.0001 * y_range, line_color="black",
           line_width=2, fill_color='black')

    # whiskers (almost-0 height rects simpler than segments with
    # categorial x-axis)
    p.rect(cats, lower, 0.2, 0.0001 * y_range, line_color='black',
           fill_color='black')
    p.rect(cats, upper, 0.2, 0.0001 * y_range, line_color='black',
           fill_color='black')

    # outliers
    p.circle(outx, outy, size=6, color='black')

    return p



# ########################################################################## #
#                            MCMC UTILITIES                                  #
# ########################################################################## #
def run_ensemble_emcee(log_post, n_burn, n_steps, n_walkers=None, p_dict=None,
                       p0=None, columns=None, args=(), threads=None,
                       return_sampler=False):
    """
    Run emcee.

    Parameters
    ----------
    log_post : function
        The function that computes the log posterior.  Must be of
        the form log_post(p, *args), where p is a NumPy array of
        parameters that are sampled by the MCMC sampler.
    n_burn : int
        Number of burn steps
    n_steps : int
        Number of MCMC samples to take
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
    return_sampler : bool, default False
        If True, return sampler as well as DataFrame with results.

    Returns
    -------
    Pandas DataFrame with columns given by flattened MCMC chains.
    Also has a column 'lnprob' containing the log of the posterior
    and 'chain', which is the chain ID.  Optionally, the sampler is
    returned in addition.
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

    # Build starting points of walkers
    if p0 is None:
        p0 = np.empty((n_walkers, n_dim))
        for i, key in enumerate(p_dict):
            p0[:,i] = p_dict[key][0](*(p_dict[key][1] + (n_walkers,)))

    # Set up the EnsembleSampler instance
    if threads is not None:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                        args=args, threads=threads)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                        args=args)

    # Do burn-in
    pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)

    # Sample again, starting from end burn-in state
    _ = sampler.run_mcmc(pos, n_steps)

    # Make DataFrame for results
    df = pd.DataFrame(data=sampler.flatchain, columns=columns)
    df['lnprob'] = sampler.flatlnprobability
    df['chain'] = np.concatenate([i * np.ones(n_steps, dtype=int)
                                                for i in range(n_walkers)])

    if return_sampler:
        return df, sampler
    else:
        return df


def run_pt_emcee(log_like, log_prior, n_burn, n_steps, n_temps=None,
                 n_walkers=None, p_dict=None, p0=None, columns=None,
                 loglargs=(), logpargs=(), threads=None, return_lnZ=False,
                 return_sampler=False):
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
    return_lnZ : bool, default False
        If True, additionally return lnZ and dlnZ.
    return_sampler : bool, default False
        If True, additionally return sampler.

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
                      log likelihood
    lnZ : float, optional
        ln Z(1), which is equal to the evidence of the
        parameter estimation problem.
    dlnZ : float, optional
        The estimated error in the lnZ calculation.
    sampler : emcee.PTSampler instance, optional
        The sampler instance.
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

    # Build starting points of walkers
    if p0 is None:
        p0 = np.empty((n_temps, n_walkers, n_dim))
        for i, key in enumerate(p_dict):
            p0[:,:,i] = p_dict[key][0](
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
    pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)

    # Sample again, starting from end burn-in state
    _ = sampler.run_mcmc(pos, n_steps)

    # Compute thermodynamic integral
    lnZ, dlnZ = sampler.thermodynamic_integration_log_evidence(fburnin=0)

    # Make DataFrame for results
    df = pd.DataFrame(data=sampler.flatchain.reshape(
                            (n_temps*n_walkers*n_steps, n_dim)),
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

    if return_lnZ:
        if return_sampler:
            return df, lnZ, dlnZ, sampler
        else:
            return df, lnZ, dlnZ
    elif return_sampler:
        return df, sampler
    else:
        return df


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


def extract_2d_hist(samples_x, samples_y, nbins=100, density=True,
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
    nbins : int
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
    count, x_bins, y_bins = np.histogram2d(samples_x, samples_y, bins=nbins,
                                           normed=density)

    # Make the bins into the bin centers, not the edges
    x = (x_bins[:-1] + x_bins[1:]) / 2.0
    y = (y_bins[:-1] + y_bins[1:]) / 2.0

    # Make mesh grid out of x_bins and y_bins
    if meshgrid:
        y, x = np.meshgrid(x, y)

    return count.transpose(), x, y


def norm_cumsum_2d(sample_x, sample_y, nbins=100, meshgrid=False):
    """
    Returns 1 - the normalized cumulative sum of two sets of samples.

    Parameters
    ----------
    samples_x : array
        1D array of MCMC samples for x-axis
    samples_y : array
        1D array of MCMC samples for y-axis
    nbins : int
        Number of bins in histogram. The same binning is
        used in the x and y directions.
    meshgrid : bool, options
        If True, the returned `x` and `y` arrays are two-dimensional
        as constructed with np.meshgrid().  If False, `x` and `y`
        are returned as 1D arrays.

    Returns
    -------
    norm_cumcum : array, shape (nbins, nbins)
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
    count, x, y = extract_2d_hist(sample_x, sample_y, nbins=nbins,
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


def hpd(trace, mass_frac) :
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
    int_width = d[n_samples:] - d[:n-n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])


# ########################################################################## #
#                    IMAGE PROCESSING UTILITIES                              #
# ########################################################################## #
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
    regions = skimage.measure.regionprops(roi)
    bbox = regions[0].bbox
    roi_bbox = np.s_[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]

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
    return [im[i:i+width, j:j+width]
                for i in range(0, im.shape[0], width)
                    for j in range(0, im.shape[1], width)
                        if roi_test(roi[i:i+width, j:j+width])]
