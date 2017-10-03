import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import pymc3 as pm
import scipy.ndimage
import skimage

import matplotlib._cntr

import bokeh.application
import bokeh.application.handlers
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import datashader as ds
import datashader.bokeh_ext


from . import utils

def ecdf(data, p=None, x_axis_label=None, y_axis_label='ECDF', title=None,
         plot_height=300, plot_width=450, formal=False, **kwargs):
    """
    Create a plot of an ECDF.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots.
    kwargs
        Any kwargs to be passed to either p.circle or p.line, for
        `formal` being False or True, respectively.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    # Check data to make sure legit
    data = utils._convert_data(data)

    # Data points on ECDF
    x, y = _ecdf_vals(data, formal)

    # Instantiate Bokeh plot if not already passed in
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)

    if formal:
        # Line of steps
        p.line(x, y, **kwargs)

        # Rays for ends
        p.ray(x[0], 0, None, np.pi, **kwargs)
        p.ray(x[-1], 1, None, 0, **kwargs)      
    else:
        p.circle(x, y, **kwargs)

    return p


def _ecdf_vals(data, formal=False):
    """
    Get x, y, values of an ECDF for plotting.

    Parameters
    ----------
    data : ndarray
        One dimensional Numpay array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)

    if formal:
        # Set up output arrays
        x_formal = np.empty(2*(len(x) + 1))
        y_formal = np.empty(2*(len(x) + 1))

        # y-values for steps
        y_formal[:2] = 0
        y_formal[2::2] = y
        y_formal[3::2] = y

        # x- values for steps
        x_formal[0] = x[0]
        x_formal[1] = x[0]
        x_formal[2::2] = x
        x_formal[3:-1:2] = x[1:]
        x_formal[-1] = x[-1]

        return x_formal, y_formal
    else:
        return x, y


def adjust_range(element, buffer=0.05):
    """
    Adjust soft ranges of dimensions of HoloViews element.

    Parameters
    ----------
    element : holoviews element
        Element which will have the `soft_range` of each kdim and vdim
        recomputed to give a buffer around the glyphs.
    buffer : float, default 0.05
        Buffer, as a fraction of the whole data range, to give around
        data.

    Returns
    -------
    output : holoviews element
        Inputted HoloViews element with updated soft_ranges for its
        dimensions.
    """
    # This only works with DataFrames
    if type(element.data) != pd.core.frame.DataFrame:
        raise RuntimeError(
            'Can only adjust range if data is Pandas DataFrame.')

    # Adjust ranges of kdims
    for i, dim in enumerate(element.kdims):
        if element.data[dim.name].dtype in [float, int]:
            data_range = (element.data[dim.name].min(),
                          element.data[dim.name].max())
            if data_range[1] - data_range[0] > 0:
                buff = buffer * (data_range[1] - data_range[0])
                element.kdims[i].soft_range = (data_range[0] - buff,
                                               data_range[1] + buff)

    # Adjust ranges of vdims
    for i, dim in enumerate(element.vdims):
        if element.data[dim.name].dtype in [float, int]:
            data_range = (element.data[dim.name].min(),
                          element.data[dim.name].max())
            if data_range[1] - data_range[0] > 0:
                buff = buffer * (data_range[1] - data_range[0])
                element.vdims[i].soft_range = (data_range[0] - buff,
                                               data_range[1] + buff)

    return element


def colored_ecdf(df, cats, val, palette, p=None, show_legend=False,
                 x_axis_label=None,
                 y_axis_label='ECDF', title=None, plot_height=300, 
                 plot_width=450, **kwargs):
    """
    """
    df_sorted = df.sort_values(by=val)
    _, df_sorted['__ecdf_y_values'] = _ecdf_vals(df_sorted[val])
    gb = df_sorted.groupby(cats)
    n = len(gb)
    

def _catplot(df, cats, val, kind, p=None, x_axis_label=None,
             y_axis_label=None, title=None, plot_height=300, plot_width=400, 
             palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
             show_legend=False, formal=False, width=0.5, order=None,
             **kwargs):
    """
    Generate a plot with a categorical variable on x-axis.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a kdim
        in HoloViews.
    kind : str, either 'jitter' or 'box'
        Kind of plot to make.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    show_legend : bool, default False
        If True, show legend.
    width : float, default 0.5
        Maximum allowable width of jittered points or boxes. A value of
        1 means that the points or box take the entire space allotted.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots. Only active when `kind` is 'ecdf'.
    show_legend : bool, default False
        If True, show a legend. Only active when `kind` is 'ecdf' or
        'colored_ecdf'.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.circle when making the jitter plot
        or to p.quad when making a box plot..

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot or box plot.
    """
    if order is not None:
        if len(order) > len(set(order)):
            raise RuntimeError('Nonunique entries in `order`.')

    if formal == True and kind != 'ecdf':
        warnings.warn('`formal` kwarg not active for ' + kind + '.')
    if show_legend == True and kind not in ['ecdf', 'colored_ecdf']:
        warnings.warn('`show_legend` kwarg not active for ' + kind + '.')

    if p is None:
        if y_axis_label is None and kind not in ['ecdf', 'colored_ecdf']:
            y_axis_label = val
            
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)

        p_was_None = True
    else:
        p_was_None = False

    # Get GroupBy object, sorted if need be
    if kind == 'colored_ecdf':
        df_sorted = df.sort_values(by=val)
        _, df_sorted['__ecdf_y_values'] = _ecdf_vals(df_sorted[val])
        gb = df_sorted.groupby(cats)
    else:
        gb = df.groupby(cats)

    # Number of categorical variables
    n = len(gb)
        
    # If a single string for palette, set color
    if type(palette) == str:
        if kind  != 'box' and 'color' not in kwargs:
            kwargs['color'] = palette
        elif kind == 'box' and 'fill_color' not in kwargs:
            kwargs['fill_color'] = palette
        palette = None
    elif len(palette) == 1:
        if kind != 'box' and 'color' not in kwargs:
            kwargs['color'] = palette[0]
        elif kind == 'box' and 'fill_color' not in kwargs:
            kwargs['fill_color'] = palette[0]
        palette = None
    else:
        color_cycle = list(range(len(palette))) * (n // len(palette) + 1)

    # Set box line colors
    if kind == 'box' and 'line_color' not in kwargs:
        kwargs['line_color'] = 'black'

    labels = {}
    for i, g in enumerate(gb):
        if kind in ['box', 'jitter']:
            if order is None:
                x = i + 0.5
            elif g[0] not in order:
                raise RuntimeError('Entry ' + g[0], ' not in `order`.')
            else:
                x = order.index(g[0]) + 0.5

            if type(g[0]) == tuple:
                labels[x] = ', '.join([str(c) for c in g[0]])
            else:
                labels[x] = str(g[0])

        if kind == 'box':
            data = g[1][val]
            bottom, middle, top = np.percentile(data, [25, 50, 75])
            iqr = top - bottom
            left = x - width / 2
            right = x + width / 2
            top_whisker = min(top + 1.5*iqr, data.max())
            bottom_whisker = max(bottom - 1.5*iqr, data.min())
            whisk_lr = [x - 0.1, x + 0.1]
            outliers = data[(data > top_whisker) | (data < bottom_whisker)]

            if palette is None:
                p.quad(left, right, top, bottom, **kwargs)
            else:
                p.quad(left, right, top, bottom,
                       fill_color=palette[color_cycle[i]], **kwargs)
            p.line([left, right], [middle]*2, color='black')
            p.line([x, x], [bottom, bottom_whisker], color='black')
            p.line([x, x], [top, top_whisker], color='black')
            p.line(whisk_lr, bottom_whisker, color='black')
            p.line(whisk_lr, top_whisker, color='black')
            p.circle([x]*len(outliers), outliers, color='black')
        elif kind == 'jitter':
            if palette is None:
                p.circle(x={'value': x, 
                            'transform': bokeh.models.Jitter(width=width)},
                         y=g[1][val],
                         **kwargs)
            else:
                p.circle(x={'value': x, 
                            'transform': bokeh.models.Jitter(width=width)},
                         y=g[1][val], 
                         color=palette[color_cycle[i]],
                         **kwargs)
        elif kind in ['ecdf', 'colored_ecdf']:
            if show_legend:
                if type(g[0]) == tuple:
                    legend = ', '.join([str(c) for c in g[0]])
                else:
                    legend = str(g[0])
            else:
                legend = None

            if kind == 'ecdf':
                if palette is None:
                    ecdf(g[1][val],
                         formal=formal,
                         p=p, 
                         legend=legend, 
                         **kwargs)
                else:
                    ecdf(g[1][val],
                         formal=formal,
                         p=p,
                         legend=legend,
                         color=palette[color_cycle[i]],
                         **kwargs)
            elif kind == 'colored_ecdf':
                if palette is None:
                    p.circle(g[1][val],
                             g[1]['__ecdf_y_values'],
                             legend=legend, 
                             **kwargs)
                else:
                    p.circle(g[1][val],
                             g[1]['__ecdf_y_values'],
                             legend=legend, 
                             color=palette[color_cycle[i]],
                             **kwargs)
   
    if kind in ['box', 'jitter']:
        p.xaxis.ticker = np.arange(len(gb)) + 0.5
        p.xaxis.major_label_overrides = labels
        p.xgrid.visible = False
        
    if kind in ['ecdf', 'colored_ecdf']:
        p.legend.location = 'bottom_right'

    return p


def ecdf_collection(
        df, cats, val, p=None, x_axis_label=None, y_axis_label=None,
        title=None, plot_height=300, plot_width=400, 
        palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
        show_legend=True, formal=False, order=None, **kwargs):
    """
    Make a collection of ECDFs from a tidy DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a kdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    show_legend : bool, default False
        If True, show legend.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.circle when making the jitter plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot.
    """
    if x_axis_label is None:
        x_axis_label = val
    if y_axis_label is None:
        y_axis_label = 'ECDF'

    return _catplot(df,
                    cats, 
                    val, 
                    'ecdf', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette,
                    show_legend=show_legend,
                    formal=formal,
                    order=order, 
                    **kwargs)


def colored_ecdf(
        df, cats, val, p=None, x_axis_label=None, y_axis_label=None,
        title=None, plot_height=300, plot_width=400, 
        palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
        show_legend=True, order=None, **kwargs):
    """
    Make an ECDF where points are colored by categorial variables.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a kdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    show_legend : bool, default False
        If True, show legend.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.circle when making the jitter plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot.
    """
    if x_axis_label is None:
        x_axis_label = val
    if y_axis_label is None:
        y_axis_label = 'ECDF'
    if 'formal' in kwargs:
        raise RuntimeError('`formal` kwarg not allowed for colored ECDF.')

    return _catplot(df,
                    cats, 
                    val, 
                    'colored_ecdf', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette,
                    show_legend=show_legend,
                    formal=False,
                    order=order, 
                    **kwargs)


def jitter(df, cats, val, p=None, x_axis_label=None, y_axis_label=None, 
           title=None, plot_height=300, plot_width=400, 
           palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
           jitter_width=0.5, order=None, **kwargs):
    """
    Make a jitter plot from a tidy DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a kdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    jitter_width : float, default 0.5
        Maximum allowable width of jittered points. A value of 1 means
        that the points take the entire space allotted.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.circle when making the jitter plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot.
    """
    return _catplot(df,
                    cats, 
                    val, 
                    'jitter', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette, 
                    width=jitter_width, 
                    show_legend=False,
                    order=order, 
                    **kwargs)


def boxwhisker(df, cats, val, p=None, x_axis_label=None, y_axis_label=None, 
               title=None, plot_height=300, plot_width=400, 
               palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
               box_width=0.5, order=None, **kwargs):
    """
    Make a box-and-whisker plot from a tidy DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a kdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all boxes are colored with that color. Default is
        the default color cycle employed by HoloViews.
    box_width : float, default 0.5
        Maximum allowable width of the boxes. A value of 1 means that
        the boxes take the entire space allotted.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.quad when making the plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with box-and-whisker plot.

    Notes
    -----
    .. Uses the Tukey convention for box plots. The top and bottom of
       the box are respectively the 75th and 25th percentiles of the
       data. The line in the middle of the box is the median. The 
       top whisker extends to the lesser of the largest data point and
       the top of the box plus 1.5 times the interquartile region (the
       height of the box). The bottom whisker extends to the greater of 
       the smallest data point and the bottom of the box minus 1.5 times
       the interquartile region. Data points not between the ends of the
       whiskers are considered outliers and are plotted as individual
       points.
    """
    return _catplot(df,
                    cats, 
                    val, 
                    'box', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette,
                    width=box_width, 
                    show_legend=False,
                    order=order, 
                    **kwargs)


def imshow(im, color_mapper=None, plot_height=400, 
           length_units='pixels', interpixel_distance=1.0,
           no_ticks=False, flip=True, return_im=False):
    """
    Display an image in a Bokeh figure.
    
    Parameters
    ----------
    im : Numpy array
        If 2D, intensity image to be displayed. If 3D, first two
        dimensions are pixel values. Last dimension can be of length
        1, 2, or 3, which specify colors.
    color_mapper : str or bokeh.models.LinearColorMapper, default None
        If `im` is an intensity image, `color_mapper` is a mapping of 
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `color_mapper` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    plot_height : int
        Height of the plot in pixels. The width is scaled so that the 
        x and y distance between pixels is the same.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    return_im : bool, default False
        If True, return the GlyphRenderer instance of the image being
        displayed.
        
    Returns
    -------
    p : bokeh.plotting.figure instance
        Bokeh plot with image displayed.
    im : bokeh.models.renderers.GlyphRenderer instance (optional)
        The GlyphRenderer instance of the image being displayed. This is
        only returned if `return_im` is True. 

    Notes
    -----
    .. The plot area is set to closely approximate square pixels, but
       this is not always possible since Bokeh sets the plotting area
       based on the entire plot, inclusive of ticks and titles. However,
       if you choose `no_ticks` to be True, no tick or axes labels are
       present, and the pixels are displayed as square.
    """
    # If a single channel in 3D image, flatten and check shape
    if im.ndim == 3:
        if im.shape[2] == 1:
            im = im[:,:,0]
        elif im.shape[2] not in [2, 3]:
            raise RuntimeError('Can only display 1, 2, or 3 channels.')

    # Get color mapper
    if im.ndim == 2:
        if color_mapper is None:
            color_mapper = bokeh.models.LinearColorMapper(
                                        bokeh.palettes.viridis(256))
        elif (type(color_mapper) == str 
                and color_mapper.lower() in ['rgb', 'cmy']):
            raise RuntimeError(
                    'Cannot use rgb or cmy colormap for intensity image.')
    elif im.ndim == 3:
        if color_mapper is None or color_mapper.lower() == 'cmy':
            im = im_merge(*np.rollaxis(im, 2), cmy=True)
        elif color_mapper.lower() == 'rgb':
            im = im_merge(*np.rollaxis(im, 2), cmy=False)
        else:
            raise RuntimeError('Invalid color mapper for color image.')
    else:
        raise RuntimeError(
                    'Input image array must have either 2 or 3 dimensions.')

    # Get shape, dimensions
    n, m = im.shape[:2]
    dw = m * interpixel_distance
    dh = n * interpixel_distance
    
    # Set up figure with appropriate dimensions
    plot_width = int(m/n * plot_height)
    if no_ticks:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, x_range=[0, dw],
            y_range=[0, dh], tools='pan,box_zoom,wheel_zoom,reset')
        p.xaxis.major_label_text_font_size = '0pt'
        p.yaxis.major_label_text_font_size = '0pt'
        p.xaxis.major_tick_line_color = None 
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None 
        p.yaxis.minor_tick_line_color = None
    else:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, x_range=[0, dw],
            y_range=[0, dh], x_axis_label=length_units, 
            y_axis_label=length_units, tools='pan,box_zoom,wheel_zoom,reset')

    # Display the image
    if im.ndim == 2:
        if flip:
            im = im[::-1,:]
        im_bokeh = p.image(image=[im], x=0, y=0, dw=dw, dh=dh, 
                           color_mapper=color_mapper)
    else:
        im_bokeh = p.image_rgba(image=[rgb_to_rgba32(im, flip=flip)], 
                                x=0, y=0, dw=dw, dh=dh)
    
    if return_im:
        return p, im_bokeh
    return p


def im_merge(im_0, im_1, im_2=None, im_0_max=None,
             im_1_max=None, im_2_max=None, im_0_min=None,
             im_1_min=None, im_2_min=None, cmy=True):
    """
    Merge channels to make RGB image.

    Parameters
    ----------
    im_0: array_like
        Image represented in first channel.  Must be same shape
        as `im_1` and `im_2` (if not None).
    im_1: array_like
        Image represented in second channel.  Must be same shape
        as `im_1` and `im_2` (if not None).
    im_2: array_like, default None
        Image represented in third channel.  If not None, must be same
        shape as `im_0` and `im_1`.
    im_0_max : float, default max of inputed first channel
        Maximum value to use when scaling the first channel. If None,
        scaled to span entire range.
    im_1_max : float, default max of inputed second channel
        Maximum value to use when scaling the second channel
    im_2_max : float, default max of inputed third channel
        Maximum value to use when scaling the third channel
    im_0_min : float, default min of inputed first channel
        Maximum value to use when scaling the first channel
    im_1_min : float, default min of inputed second channel
        Minimum value to use when scaling the second channel
    im_2_min : float, default min of inputed third channel
        Minimum value to use when scaling the third channel
    cmy : bool, default True
        If True, first channel is cyan, second is magenta, and third is
        yellow. Otherwise, first channel is red, second is green, and 
        third is blue.

    Returns
    -------
    output : array_like, dtype float, shape (*im_0.shape, 3)
        RGB image.
    """

    # Compute max intensities if needed
    if im_0_max is None:
        im_0_max = im_0.max()
    if im_1_max is None:
        im_1_max = im_1.max()
    if im_2 is not None and im_2_max is None:
        im_2_max = im_2.max()

    # Compute min intensities if needed
    if im_0_min is None:
        im_0_min = im_0.min()
    if im_1_min is None:
        im_1_min = im_1.min()
    if im_2 is not None and im_2_min is None:
        im_2_min = im_2.min()

    # Make sure maxes are ok
    if im_0_max < im_0.max() or im_1_max < im_1.max() \
            or (im_2 is not None and im_2_max < im_2.max()):
        raise RuntimeError(
                'Inputted max of channel < max of inputted channel.')

    # Make sure mins are ok
    if im_0_min > im_0.min() or im_1_min > im_1.min() \
            or (im_2 is not None and im_2_min > im_2.min()):
        raise RuntimeError(
                'Inputted min of channel > min of inputted channel.')

    # Scale the images
    if im_0_max > im_0_min:
        im_0 = (im_0 - im_0_min) / (im_0_max - im_0_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    if im_1_max > im_1_min:
        im_1 = (im_1 - im_1_min) / (im_1_max - im_1_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    if im_2 is None:
        im_2 = np.zeros_like(im_0)
    elif im_2_max > im_2_min:
        im_2 = (im_2 - im_2_min) / (im_2_max - im_2_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    # Convert images to RGB
    if cmy:
        im_c = np.stack((np.zeros_like(im_0), im_0, im_0), axis=2)
        im_m = np.stack((im_1, np.zeros_like(im_1), im_1), axis=2)
        im_y = np.stack((im_2, im_2, np.zeros_like(im_2)), axis=2)
        im_rgb = im_c + im_m + im_y
        for i in [0, 1, 2]:
            im_rgb[:,:,i] /= im_rgb[:,:,i].max()
    else:
        im_rgb = np.empty((*im_0.shape, 3))
        im_rgb[:,:,0] = im_0
        im_rgb[:,:,1] = im_1
        im_rgb[:,:,2] = im_2

    return im_rgb


def rgb_to_rgba32(im, flip=True):
    """
    Convert an RGB image to a 32 bit-encoded RGBA image.

    Parameters
    ----------
    im : ndarray, shape (nrows, ncolums, 3)
        Input image. All pixel values must be between 0 and 1.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.

    Returns
    -------
    output : ndarray, shape (nros, ncolumns), dtype np.uint32
        Image decoded as a 32 bit RBGA image.
    """
    # Ensure it has three channels
    if im.ndim != 3 or im.shape[2] !=3:
        raise RuntimeError('Input image is not RGB.')

    # Make sure all entries between zero and one
    if (im < 0).any() or (im > 1).any():
        raise RuntimeError('All pixel values must be between 0 and 1.')

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im_8 = skimage.img_as_ubyte(im)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.stack((*np.rollaxis(im_8, 2),
                        255*np.ones((n, m), dtype=np.uint8)), axis=2)

    # Reshape into 32 bit. Must flip up/down for proper orientation
    if flip:
        return np.flipud(im_rgba.view(dtype=np.int32).reshape((n, m)))
    else:
        return im_rgba.view(dtype=np.int32).reshape((n, m))


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


def corner(trace, datashade=True, vars=None, labels=None, plot_width=150, 
           smooth=2, bins=20, cmap='black', contour_color='black', 
           hist_color='black', alpha=1, bins_2d=50, plot_ecdf=False,
           plot_width_correction=50, plot_height_correction=40, levels=None,
           weights=None):
    """
    Make a corner plot of MCMC results.

    Parameters
    ----------
    trace : PyMC3 Trace or MultiTrace instance or Pandas DataFrame
        Trace of MCMC sampler.
    datashade : bool, default True
        Whether or not to convert sampled points to a raster image using
        Datashader.
    """

    if type(trace) == pd.core.frame.DataFrame:
        df = trace
    else:
        df = pm.trace_to_dataframe(trace) 

    if vars is None:
        vars = df.columns[~(df.columns.isin(
                            ['chain', 'log_like', 'log_post', 'log_prior']))]
    if len (vars) > 6:
        raise RuntimeError(
                    'For space purposes, can show only five variables.')
        
    for col in vars:
        if col not in df.columns:
            raise RuntimeError(
                        'Column ' + col + ' not in the columns of DataFrame.')
            
    if labels is None:
        labels = vars
    elif len(labels) != len(vars):
        raise RuntimeError('len(vars) must equal len(labels)')

    if len(vars) == 1:
        raise NotImplementedError('Single histogram to be implemented.')
        
    if not datashade:
        if len(df) > 1000:
            warnings.warn(
                'Rendering so many points without DataShader is ill-advised.')
        elif len(df) > 10000:
            raise RuntimeError(
                'Cannot render more than 10,000 samples without DataShader.')

    plots = [[None for _ in range(len(vars))] for _ in range(len(vars))]
    
    for i, j in zip(*np.tril_indices(len(vars))):
        pw = plot_width
        ph = plot_width
        if j == 0:
            pw += plot_width_correction
        if i == len(vars) - 1:
            ph += plot_height_correction
            
        x = vars[j]
        if i != j:
            y = vars[i]
            x_range, y_range = _data_range(df, x, y)
            plots[i][j] = bokeh.plotting.figure(
                    x_range=x_range, y_range=y_range,
                    plot_width=pw, plot_height=ph)
            if datashade:
                _ = datashader.bokeh_ext.InteractiveImage(
                    plots[i][j], _create_points_image, df=df, x=x, y=y, 
                    cmap=cmap)
            else:
                plots[i][j].circle(df[x], df[y], size=2, 
                                   alpha=alpha, color=cmap)
            xs, ys = _get_contour_lines(
                df[x].values, df[y].values, bins=bins_2d, smooth=smooth, 
                levels=levels, weights=weights)
            plots[i][j].multi_line(xs, ys, line_color=contour_color, 
                                   line_width=2)
        else:
            if plot_ecdf:
                x_range, _ = _data_range(df, x, x)
                plots[i][i] = bokeh.plotting.figure(
                        x_range=x_range, y_range=[-0.02, 1.02], 
                        plot_width=pw, plot_height=ph)
                if datashade:
                    x_ecdf, y_ecdf = _ecdf_vals(df[x], formal=True)
                    df_ecdf = pd.DataFrame(data={x: x_ecdf, 'ECDF': y_ecdf}) 
                    _ = datashader.bokeh_ext.InteractiveImage(
                            plots[i][i], _create_line_image, df=df_ecdf, 
                            x=x, y='ECDF', cmap=hist_color)
                else:
                    plots[i][i] = ecdf(df[x], p=plots[i][i], formal=True,
                                       line_width=2, line_color=hist_color)
            else:
                x_range, _ = _data_range(df, x, x)
                plots[i][i] = bokeh.plotting.figure(
                            x_range=x_range, plot_width=pw, plot_height=ph)
                f, e = np.histogram(df[x], bins=bins, density=True)
                e0 = np.empty(2*len(e))
                f0 = np.empty(2*len(e))
                e0[::2] = e
                e0[1::2] = e
                f0[0] = 0
                f0[-1] = 0
                f0[1:-1:2] = f
                f0[2:-1:2] = f
                
                plots[i][i].line(e0, f0, line_width=2, color=hist_color)

    # Link axis ranges
    for i in range(1,len(vars)):
        for j in range(i):
            plots[i][j].x_range = plots[j][j].x_range
            plots[i][j].y_range = plots[i][i].x_range

    # Label axes
    for i, label in enumerate(labels):
        plots[-1][i].xaxis.axis_label = label

    for i, label in enumerate(labels[1:]):
        plots[i+1][0].yaxis.axis_label = label

    if plot_ecdf:
        plots[0][0].yaxis.axis_label = 'ECDF'
        
    # Take off tick labels
    for i in range(len(vars)-1):
        for j in range(i+1):
            plots[i][j].xaxis.major_label_text_font_size = '0pt'

    if not plot_ecdf:
        plots[0][0].yaxis.major_label_text_font_size = '0pt'

    for i in range(1, len(vars)):
        for j in range(1, i+1):
            plots[i][j].yaxis.major_label_text_font_size = '0pt'
    
    grid = bokeh.layouts.gridplot(plots, toolbar_location='left',
                                  toolbar_sticky=False)
    return grid


def _data_range(df, x, y, margin=0.02):
    x_range = df[x].max() - df[x].min()
    y_range = df[y].max() - df[y].min()
    return ([df[x].min() - x_range*margin, df[x].max() + x_range*margin],
            [df[y].min() - y_range*margin, df[y].max() + y_range*margin])


def _create_points_image(x_range, y_range, w, h, df, x, y, cmap):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=int(h), 
                    plot_width=int(w))
    agg = cvs.points(df, x, y, agg=ds.reductions.count())
    return ds.transfer_functions.dynspread(ds.transfer_functions.shade(
                                        agg, cmap=cmap, how='linear'))


def _create_line_image(x_range, y_range, w, h, df, x, y, cmap=None):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=int(h), 
                    plot_width=int(w))
    agg = cvs.line(df, x, y)
    return ds.transfer_functions.dynspread(ds.transfer_functions.shade(
                                               agg, cmap=cmap))


def _get_contour_lines(x, y, smooth=4, levels=None, bins=50, weights=None):
    """
    Get lines for contour overlay.

    Based on code from emcee by Dan Forman-Mackey.
    """
    data_range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, data_range)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic data_range. You could try using the "
                         "'data_range' argument.")

    if smooth is not None:
        H = scipy.ndimage.gaussian_filter(H, smooth)
        
    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    # Set up contour object
    X2, Y2 = np.meshgrid(X2, Y2)
    c = matplotlib._cntr.Cntr(X2, Y2, H2.transpose())
    xs = []
    ys = []
    for level in V:
        paths = c.trace(level)
        n_lines = len(paths) // 2
        for line in paths[:n_lines]:
            xs.append(line[:,0])
            ys.append(line[:,1])
            
    return xs, ys


# def distribution_plot_app(doc, scipy_dist=None, custom_pdf=None,
#     custom_pmf=None,
#     custom_cdf=None, params=None, plot_height=200, plot_width=300,
#     x_axis_label=None, pdf_y_axis_label=None, pmf_y_axis_label=None, 
#     cdf_y_axis_label='CDF', dist_name=None):
#     """
#     Function to build interactive Bokeh app.
#     """

#     if scipy_dist is None:
#         if (custom_pdf is None and custom_pmf is None) or custom_cdf is None:
#             raise RuntimeError('For custom distributions, both PDF/PMF and'
#                                 + ' CDF must be specified.')
#         if (custom_pdf is None + custom_pmf is None) == 2:
#             raise RuntimeError('Can only specify custom PMF or PDF.')
#         if custom_pdf is None:
#             discrete = False
#             if pmf_y_axis_label is None:
#                 p_y_axis_label = 'PMF'
#         else:
#             discrete = True
#             if pdf_y_axis_label is None:
#                 p_y_axis_label = 'PDF'
#     elif (   custom_pdf is not None 
#           or custom_pmf is not None
#           or custom_cdf is not None):
#         raise RuntimeError(
#             'Can only specify either custom or scipy distribution.')
#     else:
#         if hasattr(scipy_dist, 'pmf'):
#             discrete = True
#             if pmf_y_axis_label is None:
#                 p_y_axis_label = 'PMF'
#         else:
#             discrete = False
#             if pdf_y_axis_label is None:
#                 p_y_axis_label = 'PDF'


#     if params is None:
#         raise RuntimeError('`params` must be specified.')


#     def _plot_app(doc):
#         p_p = bokeh.plotting.figure(plot_height=plot_height,
#                                     plot_width=plot_width,
#                                       x_axis_label=x_axis_label,
#                                       y_axis_label='PDF'
#         p_c = bokeh.plotting.figure(plot_height=200,
#                                     plot_width=300,
#                                     x_axis_label='λ',
#                                     y_axis_label=r'F(λ|x̄, n)')

#         # Link the axes
#         p_cdf.x_range = p_pdf.x_range

#         # Set up data for plot
#         lam = np.linspace(0, 50, 400)
#         x_bar = 10
#         n = 100

#         source = bokeh.models.ColumnDataSource(
#                 data={'lam': lam,
#                       'post': posterior(lam, x_bar, n),
#                       'post_cdf': posterior_cdf(lam, x_bar, n)})

#         # Plot PDF and CSF
#         p_pdf.line('lam', 'post', source=source, line_width=2)
#         p_cdf.line('lam', 'post_cdf', source=source, line_width=2)
        
#         def callback(attr, old, new):
#             x_bar = x_bar_slider.value
#             n = n_slider.value
#             source.data['post'] = posterior(lam, x_bar, n)
#             source.data['post_cdf'] = posterior_cdf(lam, x_bar, n)

#         x_bar_slider = bokeh.models.Slider(start=1,
#                                            end=30,
#                                            value=10,
#                                            step=1,
#                                            title='x_bar')
#         x_bar_slider.on_change('value', callback)

#         n_slider = bokeh.models.Slider(start=1,
#                                        end=500,
#                                        value=100,
#                                        step=1,
#                                        title='n')

#         x_bar_slider.on_change('value', callback)
#         n_slider.on_change('value', callback)

#         # Add the plot to the app
#         widgets = bokeh.layouts.widgetbox([x_bar_slider, n_slider])
#         grid = bokeh.layouts.gridplot([p_pdf, p_cdf], ncols=2)
#         doc.add_root(bokeh.layouts.column(widgets, grid))

#     handler = bokeh.application.handlers.FunctionHandler(posterior_plot_app)
#     app = bokeh.application.Application(handler)
