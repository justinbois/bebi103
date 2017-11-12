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


def fill_between(x1, y1, x2, y2, x_axis_label=None, y_axis_label=None,
                 title=None, plot_height=300, plot_width=450,
                 fill_color='#1f77b4', line_color='#1f77b4', show_line=True,
                 line_width=1, fill_alpha=1, line_alpha=1, p=None, **kwargs):
    """
    Create a filled region between two curves.

    Parameters
    ----------
    x1 : array_like
        Array of x-values for first curve
    y1 : array_like
        Array of y-values for first curve
    x2 : array_like
        Array of x-values for second curve
    y2 : array_like
        Array of y-values for second curve
    y_axis_label : str, default None
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    fill_color : str, default '#1f77b4'
        Color of fill as a hex string.
    line_color : str, default '#1f77b4'
        Color of the line as a hex string.
    show_line : bool, default True
        If True, show the lines on the edges of the fill.
    line_width : int, default 1
        Line width of lines on the edgs of the fill.
    fill_alpha : float, default 1.0
        Opacity of the fill.
    line_alpha : float, default 1.0
        Opacity of the lines.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with fill-between.

    Notes
    -----
    .. Any remaining kwargs are passed to bokeh.models.patch().
    """
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)


    p.patch(x=np.concatenate((x1, x2[::-1])),
            y=np.concatenate((y1, y2[::-1])),
            alpha=fill_alpha,
            fill_color=fill_color,
            line_width=0,
            **kwargs)

    if show_line:
        p.line(x1,
               y1, 
               line_width=line_width, 
               alpha=line_alpha, 
               color=line_color)
        p.line(x2, 
               y2, 
               line_width=line_width, 
               alpha=line_alpha, 
               color=line_color)

    return p


def ecdf(data, p=None, x_axis_label=None, y_axis_label='ECDF', title=None,
         plot_height=300, plot_width=450, formal=False, x_axis_type='linear',
         y_axis_type='linear', **kwargs):
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
            x_axis_label=x_axis_label, y_axis_label=y_axis_label,
            x_axis_type=x_axis_type, y_axis_type=y_axis_type, title=title)

    if formal:
        # Line of steps
        p.line(x, y, **kwargs)

        # Rays for ends
        p.ray(x[0], 0, None, np.pi, **kwargs)
        p.ray(x[-1], 1, None, 0, **kwargs)      
    else:
        p.circle(x, y, **kwargs)

    return p


def histogram(data, bins=10, p=None, x_axis_label=None, y_axis_label=None,
              title=None, plot_height=300, plot_width=450, density=True,
              kind='step', **kwargs):
    """
    Make a plot of a histogram of a data set.

    Parameters
    ----------
    data : array_like
        1D array of data to make a histogram out of
    bins : int or array_like, default 10
        Setting for `bins` kwarg to be passed to `np.histogram()`.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default None
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    density : bool, default True
        If True, normalized the histogram. Otherwise, base the histogram
        on counts.
    kind : str, default 'step'
        The kind of histogram to display. Allowed values are 'step' and
        'step_filled'.

    Returns
    -------
    output : Bokeh figure
        Figure populted with histogram.
    """
   # Instantiate Bokeh plot if not already passed in
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)

    # Compute histogram
    f, e = np.histogram(data, bins=bins, density=density)
    e0 = np.empty(2*len(e))
    f0 = np.empty(2*len(e))
    e0[::2] = e
    e0[1::2] = e
    f0[0] = 0
    f0[-1] = 0
    f0[1:-1:2] = f
    f0[2:-1:2] = f

    if kind == 'step':
        p.line(e0, f0, **kwargs)

    if kind == 'step_filled':
        x2 = [e0.min(), e0.max()]
        y2 = [0, 0]
        p = fill_between(e0, f0, x2, y2, show_line=True, p=p, **kwargs)

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
    

def _catplot(df, cats, val, kind, p=None, x_axis_label=None,
             y_axis_label=None, title=None, plot_height=300, plot_width=400, 
             palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
             show_legend=False, formal=False, width=0.5, order=None,
             x_axis_type='linear', y_axis_type='linear', **kwargs):
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
    x_axis_type : 'linear' or 'log'
        Type of x-axis.
    y_axis_type : 'linear' or 'log'
        Type of y-axis.
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
            x_axis_label=x_axis_label, y_axis_label=y_axis_label,
            x_axis_type=x_axis_type, y_axis_type=y_axis_type, title=title)

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

    # Set up the iterator over the groupby object
    if order is None:
        order = list(gb.groups.keys())
    gb_iterator = [(order_val, gb.get_group(order_val)) 
                        for order_val in order]

    labels = {}
    for i, g in enumerate(gb_iterator):
        if kind in ['box', 'jitter']:
            x = i + 0.5

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
        show_legend=True, formal=False, order=None, x_axis_type='linear',
        **kwargs):
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
    x_axis_type : 'linear' or 'log'
        Type of x-axis.
    kwargs
        Any kwargs to be passed to p.circle when making the ECDF.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDFs.
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
                    x_axis_type=x_axis_type,
                    **kwargs)


def colored_ecdf(
        df, cats, val, p=None, x_axis_label=None, y_axis_label=None,
        title=None, plot_height=300, plot_width=400, 
        palette=['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'],
        show_legend=True, order=None, x_axis_type='linear', **kwargs):
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
    x_axis_type : 'linear' or 'log'
        Type of x-axis.
    kwargs
        Any kwargs to be passed to p.circle when making the ECDF.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with a colored ECDF.
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
                    x_axis_type=x_axis_type,
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


def _display_clicks(div, attributes=[],
                    style='float:left;clear:left;font_size=0.5pt'):
    """Build a suitable CustomJS to display the current event
    in the div model."""
    return bokeh.models.CustomJS(args=dict(div=div), code="""
        var attrs = %s; var args = [];
        for (var i=0; i<attrs.length; i++ ) {
            args.push(Number(cb_obj[attrs[i]]).toFixed(4));
        }
        var line = "<span style=%r>[" + args.join(", ") + "], </span>\\n";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if ( lines.length > 35 ) { lines.shift(); }
        div.text = lines.join("\\n");
    """ % (attributes, style))


def imshow(im, color_mapper=None, plot_height=400, plot_width=None,
           length_units='pixels', interpixel_distance=1.0,
           x_range=None, y_range=None,
           no_ticks=False, x_axis_label=None, y_axis_label=None, 
           title=None, flip=True, return_im=False, record_clicks=False):
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
    interpixel_distance_x : float, default 1.0
        Interpixel distance along x-axis in units of `length_units`.
    interpixel_distance_y : float, default None
        Interpixel distance along x-axis in units of `length_units`. If
        None, then same as `interpixel_distance_x`.        
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
    return_im : bool, default False
        If True, return the GlyphRenderer instance of the image being
        displayed.
    record_clicks : bool, default False
        If True, enables recording of clicks on the image. The clicks are
        displayed in copy-able text next to the displayed figure. 
        
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
    if x_range is not None and y_range is not None:
        dw = x_range[1] - x_range[0]
        dh = y_range[1] - y_range[0]
    else:
        dw = m * interpixel_distance
        dh = n * interpixel_distance
        x_range = [0, dw]
        y_range = [0, dh]
    
    # Set up figure with appropriate dimensions
    if plot_width is None:
        plot_width = int(m/n * plot_height)
    p = bokeh.plotting.figure(plot_height=plot_height,
                              plot_width=plot_width,
                              x_range=x_range,
                              y_range=y_range,
                              title=title,
                              tools='pan,box_zoom,wheel_zoom,reset')
    if no_ticks:
        p.xaxis.major_label_text_font_size = '0pt'
        p.yaxis.major_label_text_font_size = '0pt'
        p.xaxis.major_tick_line_color = None 
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None 
        p.yaxis.minor_tick_line_color = None
    else:
        if x_axis_label is None:
            p.xaxis.axis_label = length_units
        else:
            p.xaxis.axis_label = x_axis_label
        if y_axis_label is None:
            p.yaxis.axis_label = length_units
        else:
            p.yaxis.axis_label = y_axis_label

    # Display the image
    if im.ndim == 2:
        if flip:
            im = im[::-1,:]
        im_bokeh = p.image(image=[im],
                           x=x_range[0], 
                           y=y_range[0], 
                           dw=dw, 
                           dh=dh, 
                           color_mapper=color_mapper)
    else:
        im_bokeh = p.image_rgba(image=[rgb_to_rgba32(im, flip=flip)], 
                                x=x_range[0],
                                y=y_range[0],
                                dw=dw, 
                                dh=dh)

    if record_clicks:
        div = bokeh.models.Div(width=200)
        layout = bokeh.layouts.row(p, div)
        p.js_on_event(bokeh.events.Tap,
                      _display_clicks(div, attributes=['x', 'y']))
        if return_im:
            return layout, im_bokeh
        else:
            return layout

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


def corner(trace, vars=None, labels=None, datashade=True, plot_width=150, 
           smooth=1, bins=20, cmap='black', contour_color='black', 
           hist_color='black', alpha=1, bins_2d=50, plot_ecdf=False,
           plot_width_correction=50, plot_height_correction=40, levels=None,
           weights=None, show_contours=True, extend_contour_domain=False):
    """
    Make a corner plot of MCMC results. Heavily influenced by the corner
    package by Dan Foreman-Mackey.

    Parameters
    ----------
    trace : PyMC3 Trace or MultiTrace instance or Pandas DataFrame
        Trace of MCMC sampler.
    vars : list
        List of variables as strings included in `trace` to construct
        corner plot.
    labels : list, default None
        List of labels for the respective variables given in `vars`. If
        None, the variable names from `vars` are used.
    datashade : bool, default True
        Whether or not to convert sampled points to a raster image using
        Datashader. For almost all applications, this should be true.
        Otherwise, you will try to render thousands and thousands of
        points.
    plot_width : int, default 150
        Width of each plot in the corner plot in pixels. The height is
        computed from the width to make the plots roughly square.
    smooth : int or None, default 1
        Width of smoothing kernel for making contours.
    bins : int, default 20
        Number of binds to use in constructing histograms. Ignored if
        `plot_ecdf` is True.
    cmap : str, default 'black'
        Valid colormap string for DataShader and for coloring Bokeh
        glyphs.
    contour_color : str, default 'black'
        Color of contour lines
    hist_color : str, default 'black'
        Color of histogram lines
    alpha : float, default 1.0
        Opacity of glyphs. Ignored if `datashade` is True.
    bins_2d : int, default 50
        Number of bins in each direction for binning 2D histograms when
        computing contours
    plot_ecdf : bool, default False
        If True, plot ECDFs of samples on the diagonal of the corner
        plot. If False, histograms are plotted.
    plot_width_correction : int, default 50
        Correction for width of plot taking into account tick and axis
        labels.
    plot_height_correction : int, default 40
        Correction for height of plot taking into account tick and axis
        labels.
    levels : list of floats, default None
        Levels to use when constructing contours. By default, these are
        chosen according to this principle from Dan Foreman-Mackey:
        http://corner.readthedocs.io/en/latest/pages/sigmas.html
    weights : default None
        Value to pass as `weights` kwarg to np.histogram2d().
    show_contours : bool, default True
        If True, show contour plot on top of samples.
    extend_contour_domain : bool, default False
        If True, extend the domain of the contours a little bit beyond
        the extend of the samples. This is done in the corner module,
        but I prefer not to do it.

    Returns
    -------
    output : Bokeh gridplot
        Corner plot as a Bokeh gridplot.
    """

    if vars is None:
        raise RuntimeError('Must specify vars.')

    if type(vars) not in (list, tuple):
        raise RuntimeError('`vars` must be a list or tuple.')

    if type(trace) == pd.core.frame.DataFrame:
        df = trace
    else:
        df = pm.trace_to_dataframe(trace) 

    if len(vars) > 6:
        raise RuntimeError(
                    'For space purposes, can show only six variables.')
        
    for col in vars:
        if col not in df.columns:
            raise RuntimeError(
                        'Column ' + col + ' not in the columns of DataFrame.')
            
    if labels is None:
        labels = vars
    elif len(labels) != len(vars):
        raise RuntimeError('len(vars) must equal len(labels)')

    if len(vars) == 1:
        x = vars[0]
        if plot_ecdf:
            if datashade:
                if plot_width == 150:
                    plot_height = 200
                    plot_width = 300
                else:
                    plot_width = 200
                    plot_height=200
                x_range, _ = _data_range(df, vars[0], vars[0])
                p = bokeh.plotting.figure(
                        x_range=x_range, y_range=[-0.02, 1.02], 
                        plot_width=plot_width, plot_height=plot_height)
                x_ecdf, y_ecdf = _ecdf_vals(df[vars[0]], formal=True)
                df_ecdf = pd.DataFrame(data={vars[0]: x_ecdf, 'ECDF': y_ecdf})
                _ = datashader.bokeh_ext.InteractiveImage(
                        p, _create_line_image, df=df_ecdf, 
                        x=x, y='ECDF', cmap=hist_color)
            else:
                return ecdf(df[vars[0]], formal=True,
                            line_width=2, line_color=hist_color)
        else:
            return histogram(df[vars[0]],
                             bins=bins,
                             density=True, 
                             line_width=2,
                             color=hist_color,
                             x_axis_label=vars[0])
        
    if not datashade:
        if len(df) > 10000:
            raise RuntimeError(
                'Cannot render more than 10,000 samples without DataShader.')
        elif len(df) > 1000:
            warnings.warn(
                'Rendering so many points without DataShader is ill-advised.')

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

            if show_contours:
                xs, ys = _get_contour_lines_from_samples(
                                df[x].values,
                                df[y].values, 
                                bins=bins_2d, 
                                smooth=smooth, 
                                levels=levels,
                                weights=weights, 
                                extend_domain=extend_contour_domain)
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


def contour(X, Y, Z, levels=None, p=None, overlaid=False, plot_width=350, 
            plot_height=300, x_axis_label='x', y_axis_label='y', title=None, 
            line_color=None, line_width=2, color_mapper=None,
            overlay_grid=False, fill=False, fill_palette=None,
            fill_alpha=0.75, **kwargs):
    """
    Make a contour plot, possibly overlaid on an image.

    Parameters
    ----------
    X : 2D Numpy array
        Array of x-values, as would be produced using np.meshgrid()
    Y : 2D Numpy array
        Array of y-values, as would be produced using np.meshgrid()
    Z : 2D Numpy array
        Array of z-values.
    levels : array_like
        Levels to plot, ranging from 0 to 1. The contour around a given
        level contains that fraction of the total probability if the
        contour plot is for a 2D probability density function. By 
        default, the levels are given by the one, two, three, and four
        sigma levels corresponding to a marginalized distribution from
        a 2D Gaussian distribution.
    p : bokeh plotting object, default None
        If not None, the contour are added to `p`. This option is not
        allowed if `overlaid` is True.
    overlaid : bool, default False
        If True, `Z` is displayed as an image and the contours are
        overlaid.
    plot_width : int, default 350
        Width of the plot in pixels. Ignored if `p` is not None.
    plot_height : int, default 300
        Height of the plot in pixels. Ignored if `p` is not None.
    x_axis_label : str, default 'x'
        Label for the x-axis. Ignored if `p` is not None.
    y_axis_label : str, default 'y'
        Label for the y-axis. Ignored if `p` is not None.
    title : str, default None
        Title of the plot. Ignored if `p` is not None.
    line_color : str, defaults to Bokeh default
        Color, either named CSS color or hex, of contour lines.
    line_width : int, default 2
        Width of contour lines.
    color_mapper : bokeh.models.LinearColorMapper, default Viridis
        Mapping of `Z` level to color. Ignored if `overlaid` is False.
    overlay_grid : bool, default False
        If True, faintly overlay the grid on top of image. Ignored if
        overlaid is False.

    Returns
    -------
    output : Bokeh plotting object
        Plot populated with contours, possible with an image.
    """
    if len(X.shape) != 2 or Y.shape != X.shape or Z.shape != X.shape:
        raise RuntimeError('All arrays must be 2D and of same shape.')

    if overlaid and p is not None:
        raise RuntimeError('Cannot specify `p` if showing image.')

    if line_color is None:
        if overlaid:
            line_color = 'white'
        else:
            line_color = 'black'

    if p is None:
        if overlaid:
            p = imshow(Z,
                       color_mapper=color_mapper,
                       plot_height=plot_height,
                       plot_width=plot_width,
                       x_axis_label=x_axis_label,
                       y_axis_label=y_axis_label,
                       title=title,
                       x_range = [X.min(), X.max()],
                       y_range = [Y.min(), Y.max()],
                       no_ticks=False, 
                       flip=False, 
                       return_im=False)
        else:
            p = bokeh.plotting.figure(plot_width=plot_width,
                                      plot_height=plot_height,
                                      x_axis_label=x_axis_label,
                                      y_axis_label=y_axis_label,
                                      title=title)

    # Set default levels
    if levels is None:
        levels = 1.0 - np.exp(-np.arange(0.5, 2.1, 0.5)**2 / 2)

    # Compute contour lines
    if fill or line_width:
        xs, ys = _contour_lines(X, Y, Z, levels)

    # Make fills. This is currently not supported
    if fill:
        raise NotImplementedError('Filled contours are not yet implemented.')
        if fill_palette is None:
            if len(levels) <= 6:
                fill_palette = bokeh.palettes.Greys[len(levels)+3][1:-1]
            elif len(levels) <= 10:
                fill_palette = bokeh.palettes.Viridis[len(levels)+1]
            else:
                raise RuntimeError(
                    'Can only have maximally 10 levels with filled contours' +
                    ' unless user specifies `fill_palette`.')
        elif len(fill_palette) != len(levels) + 1:
            raise RuntimeError('`fill_palette` must have 1 more entry' +
                               ' than `levels`')

        p.patch(xs[-1], ys[-1],
                color=fill_palette[0],
                alpha=fill_alpha,
                line_color=None)
        for i in range(1, len(levels)):
            x_p = np.concatenate((xs[-1-i], xs[-i][::-1]))
            y_p = np.concatenate((ys[-1-i], ys[-i][::-1]))
            print(len(x_p), len(y_p))
            p.patch(x_p, 
                    y_p, 
                    color=fill_palette[i],
                    alpha=fill_alpha,
                    line_color=None)

        p.background_fill_color=fill_palette[-1]

    # Populate the plot with contour lines
    if line_width:
        p.multi_line(xs, ys, line_color=line_color, line_width=line_width,
                     **kwargs)

    if overlay_grid and overlaid:
        p.grid.level = 'overlay'
        p.grid.grid_line_alpha = 0.2

    return p


def ds_line_plot(df, x, y, cmap='#1f77b4', plot_height=300, plot_width=500,
                 x_axis_label=None, y_axis_label=None, title=None,
                 margin=0.02):
    """
    Make a datashaded line plot.

    Params
    ------
    df : pandas DataFrame
        DataFrame containing the data
    x : Valid column name of Pandas DataFrame
        Column containing the x-data.
    y : Valid column name of Pandas DataFrame
        Column containing the y-data.
    cmap : str, default '#1f77b4'
        Valid colormap string for DataShader and for coloring Bokeh
        glyphs.
    plot_height : int, default 300
        Height of plot, in pixels.
    plot_width : int, default 500
        Width of plot, in pixels.
    x_axis_label : str, default None
        Label for the x-axis.
    y_axis_label : str, default None
        Label for the y-axis.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    margin : float, default 0.02
        Margin, in units of `plot_width` or `plot_height`, to leave
        around the plotted line.

    Returns
    -------
    output : datashader.bokeh_ext.InteractiveImage
        Interactive image of plot. Note that you should *not* use
        bokeh.io.show() to view the image. For most use cases, you
        should just call this function without variable assignment.
    """

    if x_axis_label is None:
        if type(x) == str:
            x_axis_label = x
        else:
            x_axis_label = 'x'

    if y_axis_label is None:
        if type(y) == str:
            y_axis_label = y
        else:
            y_axis_label = 'y'

    x_range, y_range = _data_range(df, x, y, margin=margin)
    p = bokeh.plotting.figure(plot_height=plot_height,
                              plot_width=plot_width,
                              x_range=x_range,
                              y_range=y_range,
                              x_axis_label=x_axis_label,
                              y_axis_label=y_axis_label,
                              title=title)
    return datashader.bokeh_ext.InteractiveImage(p,
                                                 _create_line_image,
                                                 df=df, 
                                                 x=x, 
                                                 y=y,
                                                 cmap=cmap)
    return p


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


def _contour_lines(X, Y, Z, levels):
    """
    Generate lines for contour plot.
    """
    # Compute the density levels.
    Zflat = Z.flatten()
    inds = np.argsort(Zflat)[::-1]
    Zflat = Zflat[inds]
    sm = np.cumsum(Zflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Zflat[sm <= v0][-1]
        except:
            V[i] = Zflat[0]
    V.sort()
    m = np.diff(V) == 0
    
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Make contours
    c = matplotlib._cntr.Cntr(X, Y, Z)
    xs = []
    ys = []
    for level in V:
        paths = c.trace(level)
        n_lines = len(paths) // 2
        for line in paths[:n_lines]:
            xs.append(line[:,0])
            ys.append(line[:,1])
            
    return xs, ys


def _get_contour_lines_from_samples(x, y, smooth=1, levels=None, bins=50, 
                                    weights=None, extend_domain=False):
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

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    if extend_domain:
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
        X2, Y2 = np.meshgrid(X2, Y2)
    else:
        X2, Y2 = np.meshgrid(X1, Y1)
        H2 = H

    return _contour_lines(X2, Y2, H2.transpose(), levels)


def distribution_plot_app(x_min, x_max, scipy_dist=None, custom_pdf=None,
    custom_pmf=None, custom_cdf=None, params=None, n=400, plot_height=200, 
    plot_width=300, x_axis_label='x', pdf_y_axis_label=None, 
    pmf_y_axis_label=None, cdf_y_axis_label='CDF', title=None):
    """
    Function to build interactive Bokeh app displaying a univariate
    probability distribution.

    Parameters
    ----------
    x_min : float
        Minimum value that the random variable can take in plots.
    x_max : float
        Maximum value that the random variable can take in plots.
    scipy_dist : scipy.stats distribution
        Distribution to use in plotting.
    custom_pdf : function
        Function with call signature f(x, *params) that computes the
        PDF of a distribution.
    custom_pmf : function    
        Function with call signature f(x, *params) that computes the
        PDF of a distribution.
    custom_cdf : function
        Function with call signature F(x, *params) that computes the
        CDF of a distribution.
    params : list of dicts
        A list of parameter specifications. Each entry in the list gives
        specifications for a parameter of the distribution stored as a
        dictionary. Each dictionary must have the following keys.
            name : str, name of the parameter
            start : float, starting point of slider for parameter (the
                smallest allowed value of the parameter)
            end : float, ending point of slider for parameter (the
                largest allowed value of the parameter)
            value : float, the value of the parameter that the slider
                takes initially. Must be between start and end.
            step : float, the step size for the slider
    n : int, default 400
        Number of points to use in making plots of PDF and CDF for 
        continuous distributions. This should be large enough to give
        smooth plots.
    plot_height : int, default 200
        Height of plots.
    plot_width : int, default 300
        Width of plots.
    x_axis_label : str, default 'x'
        Label for x-axis.
    pdf_y_axis_label : str, default 'PDF'
        Label for the y-axis of the PDF plot.
    pmf_y_axis_label : str, default 'PMF'
        Label for the y-axis of the PMF plot.
    cdf_y_axis_label : str, default 'CDF'
        Label for the y-axis of the CDF plot.
    title : str, default None
        Title to be displayed above the PDF or PMF plot.

    Returns
    -------
    output : Bokeh app
        An app to visualize the PDF/PMF and CDF. It can be displayed
        with bokeh.io.show(). If it is displayed in a notebook, the
        notebook_url kwarg should be specified.
    """

    if scipy_dist is None:
        fun_c = custom_cdf
        if (custom_pdf is None and custom_pmf is None) or custom_cdf is None:
            raise RuntimeError('For custom distributions, both PDF/PMF and'
                                + ' CDF must be specified.')
        if (custom_pdf is None and custom_pmf is None) == 2:
            raise RuntimeError('Can only specify custom PMF or PDF.')
        if custom_pdf is None:
            discrete = True
            fun_p = custom_pmf
            if pmf_y_axis_label is None:
                p_y_axis_label = 'PMF'
        else:
            discrete = False
            fun_p = custom_pdf
            if pdf_y_axis_label is None:
                p_y_axis_label = 'PDF'
    elif (   custom_pdf is not None 
          or custom_pmf is not None
          or custom_cdf is not None):
        raise RuntimeError(
            'Can only specify either custom or scipy distribution.')
    else:
        fun_c = scipy_dist.cdf
        if hasattr(scipy_dist, 'pmf'):
            discrete = True
            fun_p = scipy_dist.pmf
            if pmf_y_axis_label is None:
                p_y_axis_label = 'PMF'
        else:
            discrete = False
            fun_p = scipy_dist.pdf
            if pdf_y_axis_label is None:
                p_y_axis_label = 'PDF'



    if params is None:
        raise RuntimeError('`params` must be specified.')


    def _plot_app(doc):
        p_p = bokeh.plotting.figure(plot_height=plot_height,
                                    plot_width=plot_width,
                                    x_axis_label=x_axis_label,
                                    y_axis_label='PDF',
                                    title=title)
        p_c = bokeh.plotting.figure(plot_height=plot_height,
                                    plot_width=plot_width,
                                    x_axis_label=x_axis_label,
                                    y_axis_label='CDF')

        # Link the axes
        p_c.x_range = p_p.x_range

        # Set up data for plot
        if discrete:
            x = np.arange(x_min, x_max+1)
            x_c = np.empty(2*len(x) - 1)
            x_c[0] = x[0]
            x_c[1::2] = x[1:]
            x_c[2::2] = x[1:]
        else:
            x = np.linspace(x_min, x_max, n)

        # Make array of parameter values
        param_vals = tuple([param['value'] for param in params])

        # Compute PDF and CDF
        y_p = fun_p(x, *param_vals)
        y_c = fun_c(x, *param_vals)

        # Set up data sources
        source = bokeh.models.ColumnDataSource(data={'x': x,
                                                     'y_p': y_p, 
                                                     'y_c': y_c})
        # If discrete, need to take care with CDF

        if discrete:
            y_c_plot = np.empty(2*len(x) - 1)
            y_c_plot[::2] = y_c
            y_c_plot[1::2] = y_c[:-1]
            source_discrete_cdf = bokeh.models.ColumnDataSource(
                data={'x': x_c, 'y_c': y_c_plot})

        # Plot PDF and CDF
        if discrete:
            p_p.circle('x', 'y_p', source=source, size=5)
            p_p.segment(x0='x',
                        x1='x',
                        y0=0, 
                        y1='y_p', 
                        source=source, 
                        line_width=2)
            p_c.line('x', 'y_c', source=source_discrete_cdf, line_width=2)
        else:
            p_p.line('x', 'y_p', source=source, line_width=2)
            p_c.line('x', 'y_c', source=source, line_width=2)
        
        def _callback(attr, old, new):
            param_vals = tuple([slider.value for slider in sliders])
            source.data['y_p'] = fun_p(x, *param_vals)
            if discrete:
                y_c = fun_c(x, *param_vals)
                y_c_plot = np.empty(2*len(x) - 1)
                y_c_plot[::2] = y_c
                y_c_plot[1::2] = y_c[:-1]
                source_discrete_cdf.data['y_c'] = y_c_plot
            else:
                source.data['y_c'] = fun_c(x, *param_vals)

        sliders = [bokeh.models.Slider(start=param['start'],
                                       end=param['end'],
                                       value=param['value'],
                                       step=param['step'],
                                       title=param['name'])
                            for param in params]
        for slider in sliders:
            slider.on_change('value', _callback)

        # Add the plot to the app
        widgets = bokeh.layouts.widgetbox(sliders)
        grid = bokeh.layouts.gridplot([p_p, p_c], ncols=2)
        doc.add_root(bokeh.layouts.column(widgets, grid))

    handler = bokeh.application.handlers.FunctionHandler(_plot_app)
    return bokeh.application.Application(handler)


def im_click(im, color_mapper=None, plot_height=400, plot_width=None,
             length_units='pixels', interpixel_distance=1.0,
             x_range=None, y_range=None,
             no_ticks=False, x_axis_label=None, y_axis_label=None, 
             title=None, flip=True):
    """
    """

    def display_event(div, attributes=[],
                      style='float:left;clear:left;font_size=0.5pt'):
        """Build a suitable CustomJS to display the current event
        in the div model."""
        return bokeh.models.CustomJS(args=dict(div=div), code="""
            var attrs = %s; var args = [];
            for (var i=0; i<attrs.length; i++ ) {
                args.push(Number(cb_obj[attrs[i]]).toFixed(4));
            }
            var line = "<span style=%r>[" + args.join(", ") + "],</span>\\n";
            var text = div.text.concat(line);
            var lines = text.split("\\n")
            if ( lines.length > 35 ) { lines.shift(); }
            div.text = lines.join("\\n");
        """ % (attributes, style))

    p = imshow(im,
               color_mapper=color_mapper,
               plot_height=plot_height, 
               plot_width=plot_width,
               length_units=length_units, 
               interpixel_distance=interpixel_distance,
               x_range=x_range, 
               y_range=y_range,
               no_ticks=no_ticks, 
               x_axis_label=x_axis_label, 
               y_axis_label=y_axis_label, 
               title=title, 
               flip=flip)

    div = bokeh.models.Div(width=200)
    layout = bokeh.layout.row(p, div)

    p.js_on_event(bokeh.events.Tap, display_event(div, attributes=['x', 'y']))

    return layout



