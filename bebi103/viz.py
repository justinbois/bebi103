import copy
import re
import warnings

import numpy as np
import pandas as pd
import xarray
import scipy.stats as st
import numba

try:
    import arviz as az
    import arviz.plots.plot_utils
except:
    warnings.warn(
        "Could not import ArviZ. Perhaps it is not installed."
        " Some functionality in the viz submodule will not be available."
    )

import scipy.ndimage

import matplotlib._contour
from matplotlib.pyplot import get_cmap as mpl_get_cmap

import bokeh.models
import bokeh.palettes
import bokeh.plotting

import colorcet


try:
    import holoviews as hv
    import holoviews.operation.datashader

    hv.extension("bokeh")
except ImportError as e:
    warnings.warn(
        f"""DataShader import failed with error "{e}".
Features requiring DataShader will not work and you will get exceptions."""
    )

from . import utils
from . import image

try:
    from . import stan
except:
    warnings.warn(
        "Could not import `stan` submodule. Perhaps pystan or cmdstanpy is not properly installed."
    )


def confints(
    summaries, p=None, marker_kwargs={}, line_kwargs={}, palette=None, **kwargs
):
    """Make a horizontal plot of centers/conf ints with error bars.

    Parameters
    ----------
    summaries : list of dicts
        Each entry in `summaries` is a dictionary containing minimally
        keys 'estimate', 'conf_int', and 'label'.  The 'estimate' value
        is the point estimate, a single scalar. The 'conf_int' value is
        a two-tuple, two-list, or two-numpy array containing the low and
        high end of the confidence interval for the estimate. The
        'label' value is the name of the variable. This gives the label
        of the y-ticks.
    p : bokeh.plotting.Figure instance or None, default None
        If not None, a figure to be populated with confidence interval
        plot. If specified, it is important that `p.y_range` be set to
        contain all of the values in the labels provided in the
        `summaries` input. If `p` is None, then a new figure is created.
    marker_kwargs : dict, default {}
        Kwargs to be passed to p.circle() for plotting estimates.
    line_kwargs : dict, default {}
        Kwargs passsed to p.line() to plot the confidence interval.
    palette : list, str, or None
        If None, default colors (or those given in `marker_kwargs` and
        `line_kwargs` are used). If a str, all glyphs are colored
        accordingly, e.g., 'black'. Otherwise a list of colors is used.
    kwargs : dict
        Any additional kwargs are passed to bokeh.plotting.figure().

    Returns
    -------
    output : Bokeh figure
        Plot of error bars.
    """
    n = len(summaries)

    labels = [summary["label"] for summary in summaries]
    estimates = [summary["estimate"] for summary in summaries]
    conf_intervals = [summary["conf_int"] for summary in summaries]

    if palette is None:
        use_palette = False
    else:
        if (
            "color" in marker_kwargs
            or "line_color" in marker_kwargs
            or "fill_color" in marker_kwargs
            or "color" in line_kwargs
            or "line_color" in line_kwargs
        ):
            raise RuntimeError(
                "`palette` must be None if color is specified in "
                "`marker_kwargs` or `line_kwargs`"
            )

    if type(palette) == str:
        marker_kwargs["color"] = palette
        line_kwargs["color"] = palette
        use_palette = False
    elif type(palette) == list or type(palette) == tuple:
        palette[:n][::-1]
        use_palette = True

    line_width = kwargs.pop("line_width", 3)

    size = marker_kwargs.pop("size", 5)

    if p is None:
        if "plot_height" not in kwargs and "frame_height" not in kwargs:
            kwargs["frame_height"] = 50 * n
        if "plot_width" not in kwargs and "frame_width" not in kwargs:
            kwargs["frame_width"] = 450
        toolbar_location = kwargs.pop("toolbar_location", "above")

        p = bokeh.plotting.figure(
            y_range=labels[::-1], toolbar_location=toolbar_location, **kwargs
        )

    for i, (estimate, conf, label) in enumerate(zip(estimates, conf_intervals, labels)):
        if use_palette:
            marker_kwargs["color"] = palette[i % len(palette)]
            line_kwargs["color"] = palette[i % len(palette)]
        p.circle(x=[estimate], y=[label], size=size, **marker_kwargs)
        p.line(x=conf, y=[label, label], line_width=line_width, **line_kwargs)

    return p


def fill_between(
    x1=None,
    y1=None,
    x2=None,
    y2=None,
    show_line=True,
    patch_kwargs={},
    line_kwargs={},
    p=None,
    **kwargs,
):
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
    show_line : bool, default True
        If True, show the lines on the edges of the fill.
    patch_kwargs : dict
        Any kwargs passed into p.patch(), which generates the fill.
    line_kwargs : dict
        Any kwargs passed into p.line() in generating the line around
        the fill.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    kwargs
        All other kwargs are passed to bokeh.plotting.figure() in
        creating the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with fill-between.
    """

    if "plot_height" not in kwargs and "frame_height" not in kwargs:
        kwargs["frame_height"] = 275
    if "plot_width" not in kwargs and "frame_width" not in kwargs:
        kwargs["frame_width"] = 350

    if p is None:
        p = bokeh.plotting.figure(**kwargs)

    line_width = patch_kwargs.pop("line_width", 0)
    line_alpha = patch_kwargs.pop("line_alpha", 0)
    p.patch(
        x=np.concatenate((x1, x2[::-1])),
        y=np.concatenate((y1, y2[::-1])),
        line_width=line_width,
        line_alpha=line_alpha,
        **patch_kwargs,
    )

    if show_line:
        line_width = line_kwargs.pop("line_width", 2)
        p.line(x1, y1, line_width=line_width, **line_kwargs)
        p.line(x2, y2, line_width=line_width, **line_kwargs)

    return p


def qqplot(
    samples,
    data,
    percentile=95,
    patch_kwargs={},
    line_kwargs={},
    diag_kwargs={},
    p=None,
    **kwargs,
):
    """
    Generate a Q-Q plot.

    Parameters
    ----------
    samples : Numpy array or xarray, shape (n_samples, n) or xarray DataArray
        A Numpy array containing predictive samples.
    data : Numpy array, shape (n,) or xarray DataArray
        One-dimensional data set to use in Q-Q plot.
    percentile : int or float, default 95
        Which percentile to use in displaying the Q-Q plot.
    patch_kwargs : dict
        Any kwargs passed into p.patch(), which generates the filled
        region of the Q-Q plot..
    line_kwargs : dict
        Any kwargs passed into p.line() in generating the line around
        the fill.
    diag_kwargs : dict
        Any kwargs to be passed into p.line() in generating diagonal
        reference line of Q-Q plot.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    kwargs
        All other kwargs are passed to bokeh.plotting.figure() in
        creating the figure.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with Q-Q plot.
    """
    if type(samples) != np.ndarray:
        if type(samples) == xarray.core.dataarray.DataArray:
            samples = samples.squeeze().values
        else:
            raise RuntimeError("`samples` can only be a Numpy array or xarray.")

    if samples.ndim != 2:
        raise RuntimeError(
            "`samples` must be a 2D array, with each row being a sample."
        )

    if len(samples) < 100:
        warnings.warn(
            "`samples` has very few samples. Predictive percentiles may be poor."
        )

    if data is not None and len(data) != samples.shape[1]:
        raise RuntimeError("Mismatch in shape of `data` and `samples`.")

    if "plot_height" not in kwargs and "frame_height" not in kwargs:
        kwargs["frame_height"] = 275
    if "plot_width" not in kwargs and "frame_width" not in kwargs:
        kwargs["frame_width"] = 275
    if "fill_alpha" not in patch_kwargs:
        patch_kwargs["fill_alpha"] = 0.5

    x = np.sort(data)

    samples = np.sort(samples)

    # Upper and lower bounds
    low_theor, up_theor = np.percentile(
        samples, (50 - percentile / 2, 50 + percentile / 2), axis=0
    )

    x_range = [data.min(), data.max()]
    if "x_range" not in kwargs:
        kwargs["x_range"] = x_range

    if p is None:
        p = bokeh.plotting.figure(**kwargs)

    p = fill_between(
        x,
        up_theor,
        x,
        low_theor,
        patch_kwargs=patch_kwargs,
        line_kwargs=line_kwargs,
        show_line=True,
        p=p,
    )

    # Plot 45 degree line
    color = diag_kwargs.pop("color", "black")
    alpha = diag_kwargs.pop("alpha", 0.5)
    line_width = diag_kwargs.pop("line_width", 4)
    p.line(x_range, x_range, line_width=line_width, color=color, alpha=alpha)

    return p


def _ecdf(
    data=None,
    p=None,
    x_axis_label=None,
    y_axis_label="ECDF",
    title=None,
    plot_height=300,
    plot_width=450,
    staircase=False,
    complementary=False,
    x_axis_type="linear",
    y_axis_type="linear",
    **kwargs,
):
    """
    Create a plot of an ECDF.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    conf_int : bool, default False
        If True, display a confidence interval on the ECDF.
    ptiles : list, default [2.5, 97.5]
        The percentiles to use for the confidence interval. Ignored it
        `conf_int` is False.
    n_bs_reps : int, default 1000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    fill_color : str, default 'lightgray'
        Color of the confidence interbal. Ignored if `conf_int` is
        False.
    fill_alpha : float, default 1
        Opacity of confidence interval. Ignored if `conf_int` is False.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored if `p` is not None.
    y_axis_label : str, default 'ECDF' or 'ECCDF'
        Label for the y-axis. Ignored if `p` is not None.
    title : str, default None
        Title of the plot. Ignored if `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored if `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored if `p` is not None.
    staircase : bool, default False
        If True, make a plot of a staircase ECDF (staircase). If False,
        plot the ECDF as dots.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution functon.
    x_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    y_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    kwargs
        Any kwargs to be passed to either p.circle or p.line, for
        `staircase` being False or True, respectively.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    # Check data to make sure legit
    data = utils._convert_data(data)

    # Data points on ECDF
    x, y = _ecdf_vals(data, staircase, complementary)

    # Instantiate Bokeh plot if not already passed in
    if p is None:
        y_axis_label = kwargs.pop("y_axis_label", "ECCDF" if complementary else "ECDF")
        p = bokeh.plotting.figure(
            plot_height=plot_height,
            plot_width=plot_width,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            title=title,
        )

    if staircase:
        # Line of steps
        p.line(x, y, **kwargs)

        # Rays for ends
        if complementary:
            p.ray(x[0], 1, None, np.pi, **kwargs)
            p.ray(x[-1], 0, None, 0, **kwargs)
        else:
            p.ray(x[0], 0, None, np.pi, **kwargs)
            p.ray(x[-1], 1, None, 0, **kwargs)
    else:
        p.circle(x, y, **kwargs)

    return p


def _histogram(
    data=None,
    bins="freedman-diaconis",
    p=None,
    density=False,
    kind="step",
    line_kwargs={},
    patch_kwargs={},
    **kwargs,
):
    """
    Make a plot of a histogram of a data set.

    Parameters
    ----------
    data : array_like
        1D array of data to make a histogram out of
    bins : int, array_like, or str, default 'freedman-diaconis'
        If int or array_like, setting for `bins` kwarg to be passed to
        `np.histogram()`. If 'exact', then each unique value in the
        data gets its own bin. If 'integer', then integer data is
        assumed and each integer gets its own bin. If 'sqrt', uses the
        square root rule to determine number of bins. If
        `freedman-diaconis`, uses the Freedman-Diaconis rule for number
        of bins.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    density : bool, default False
        If True, normalized the histogram. Otherwise, base the histogram
        on counts.
    kind : str, default 'step'
        The kind of histogram to display. Allowed values are 'step' and
        'step_filled'.
    line_kwargs : dict
        Any kwargs to be passed to p.line() in making the line of the
        histogram.
    patch_kwargs : dict
        Any kwargs to be passed to p.patch() in making the fill of the
        histogram.
    kwargs : dict
        All other kwargs are passed to bokeh.plotting.figure()

    Returns
    -------
    output : Bokeh figure
        Figure populated with histogram.
    """
    if data is None:
        raise RuntimeError("Input `data` must be specified.")

    # Instantiate Bokeh plot if not already passed in
    if p is None:
        y_axis_label = kwargs.pop("y_axis_label", "density" if density else "count")

        if "plot_height" not in kwargs and "frame_height" not in kwargs:
            kwargs["frame_height"] = 275
        if "plot_width" not in kwargs and "frame_width" not in kwargs:
            kwargs["frame_width"] = 400
        y_range = kwargs.pop("y_range", bokeh.models.DataRange1d(start=0))

        p = bokeh.plotting.figure(y_axis_label=y_axis_label, y_range=y_range, **kwargs)

    # Compute histogram
    bins = _bins_to_np(df[x].values, bins)
    e0, f0 = _compute_histogram(df[x].values, bins, density)

    if kind == "step":
        p.line(e0, f0, **line_kwargs)

    if kind == "step_filled":
        x2 = [e0.min(), e0.max()]
        y2 = [0, 0]
        p = fill_between(e0, f0, x2, y2, show_line=True, p=p, patch_kwargs=patch_kwargs)

    return p


def _bins_to_np(data, bins):
    """Compute a Numpy array to pass to np.histogram() as bins."""
    if type(bins) == str and bins not in [
        "integer",
        "exact",
        "sqrt",
        "freedman-diaconis",
    ]:
        raise RuntimeError("Invalid bins specification.")

    if type(bins) == str and bins == "exact":
        a = np.unique(data)
        if len(a) == 1:
            bins = np.array([a[0] - 0.5, a[0] + 0.5])
        else:
            bins = np.concatenate(
                (
                    (a[0] - (a[1] - a[0]) / 2,),
                    (a[1:] + a[:-1]) / 2,
                    (a[-1] + (a[-1] - a[-2]) / 2,),
                )
            )
    elif type(bins) == str and bins == "integer":
        if np.any(data != np.round(data)):
            raise RuntimeError("'integer' bins chosen, but data are not integer.")
        bins = np.arange(data.min() - 1, data.max() + 1) + 0.5
    elif type(bins) == str and bins == "sqrt":
        bins = int(np.ceil(np.sqrt(len(data))))
    elif type(bins) == str and bins == "freedman-diaconis":
        h = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.cbrt(len(data))
        if h == 0.0:
            bins = 3
        else:
            bins = int(np.ceil((data.max() - data.min()) / h))

    return bins


def _compute_histogram(data, bins, density):
    """Compute values of histogram for plotting."""
    f, e = np.histogram(data, bins=bins, density=density)
    e0 = np.empty(2 * len(e))
    f0 = np.empty(2 * len(e))
    e0[::2] = e
    e0[1::2] = e
    f0[0] = 0
    f0[-1] = 0
    f0[1:-1:2] = f
    f0[2:-1:2] = f

    return e0, f0


def predictive_ecdf(
    samples,
    data=None,
    diff=None,
    percentiles=(95, 68),
    color="blue",
    data_color="orange",
    data_staircase=True,
    data_size=2,
    x=None,
    discrete=False,
    p=None,
    **kwargs,
):
    """Plot a predictive ECDF from samples.

    Parameters
    ----------
    samples : Numpy array or xarray, shape (n_samples, n) or xarray DataArray
        A Numpy array containing predictive samples.
    data : Numpy array, shape (n,) or xarray DataArray, or None
        If not None, ECDF of measured data is overlaid with predictive
        ECDF.
    diff : 'ecdf', 'iecdf', 'eppf', or None, default None
        Referring to the variable as x, if `diff` is 'iecdf' or 'eppf',
        for each value of the ECDF, plot the value of x minus the median
        x. If 'ecdf', plot the value of the ECDF minus the median ECDF
        value. If None, just plot the ECDFs.
    percentiles : list or tuple, default (95, 68)
        Percentiles for making colored envelopes for confidence
        intervals for the predictive ECDFs. Maximally four can be
        specified.
    color : str, default 'blue'
        One of ['green', 'blue', 'red', 'gray', 'purple', 'orange'].
        There are used to make the color scheme of shading of
        percentiles.
    data_color : str, default 'orange'
        String representing the color of the data to be plotted over the
        confidence interval envelopes.
    data_staircase : bool, default True
        If True, plot the ECDF of the data as a staircase.
        Otherwise plot it as dots.
    data_size : int, default 2
        Size of marker (if `data_line` if False) or thickness of line
        (if `data_staircase` is True) of plot of data.
    x : Numpy array, default None
        Points at which to evaluate the ECDF. If None, points are
        automatically generated based on the data range.
    discrete : bool, default False
        If True, the samples take on discrete values.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    kwargs
        All other kwargs are passed to bokeh.plotting.figure().

    Returns
    -------
    output : Bokeh figure
        Figure populated with glyphs describing range of values for the
        ECDF of the samples. The shading goes according to percentiles
        of samples of the ECDF, with the median ECDF plotted as line in
        the middle.
    """
    if diff == True:
        diff = "ecdf"
        warnings.warn(
            "`diff` as a Boolean is deprecated. Use 'ecdf', 'iecdf', or None."
            " Using `diff = 'ecdf'`.",
            DeprecationWarning,
        )
    elif diff == False:
        diff = None
        warnings.warn(
            "`diff` as a Boolean is deprecated. Use 'ecdf', 'iecdf', or None."
            " Using `diff = None`.",
            DeprecationWarning,
        )

    if diff is not None:
        diff = diff.lower()

    if diff == 'eppf':
        diff = 'iecdf'

    if type(samples) != np.ndarray:
        if type(samples) == xarray.core.dataarray.DataArray:
            samples = samples.squeeze().values
        else:
            raise RuntimeError("`samples` can only be a Numpy array or xarray.")

    if samples.ndim != 2:
        raise RuntimeError(
            "`samples` must be a 2D array, with each row being a sample."
        )

    if len(samples) < 100:
        warnings.warn(
            "`samples` has very few samples. Predictive percentiles may be poor."
        )

    if data is not None and len(data) != samples.shape[1]:
        raise RuntimeError("Mismatch in shape of `data` and `samples`.")

    if len(percentiles) > 4:
        raise RuntimeError("Can specify maximally four percentiles.")

    # Build ptiles
    percentiles = np.sort(percentiles)[::-1]
    ptiles = [pt for pt in percentiles if pt > 0]
    ptiles = (
        [50 - pt / 2 for pt in percentiles]
        + [50]
        + [50 + pt / 2 for pt in percentiles[::-1]]
    )
    ptiles_str = [str(pt) for pt in ptiles]

    if color not in ["green", "blue", "red", "gray", "purple", "orange", "betancourt"]:
        raise RuntimeError(
            "Only allowed colors are 'green', 'blue', 'red', 'gray', 'purple', 'orange'"
        )

    colors = {
        "blue": ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
        "green": ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
        "red": ["#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
        "orange": ["#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04"],
        "purple": ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
        "gray": ["#bdbdbd", "#969696", "#737373", "#525252", "#252525"],
        "betancourt": [
            "#DCBCBC",
            "#C79999",
            "#B97C7C",
            "#A25050",
            "#8F2727",
            "#7C0000",
        ],
    }

    samples = np.sort(samples)
    n = samples.shape[1]

    if data is not None:
        data_plot = np.sort(np.array(data))

    # y-values for ECDFs
    y = np.arange(1, n + 1) / n

    df_ecdf = pd.DataFrame(dict(y=y))
    for ptile in ptiles:
        df_ecdf[str(ptile)] = np.percentile(samples, ptile, axis=0)

    # Set up plot
    if p is None:
        x_axis_label = kwargs.pop(
            "x_axis_label", "x difference" if diff == "iecdf" else "x"
        )
        y_axis_label = kwargs.pop(
            "y_axis_label", "ECDF difference" if diff == "ecdf" else "ECDF"
        )

        if "plot_height" not in kwargs and "frame_height" not in kwargs:
            kwargs["frame_height"] = 325
        if "plot_width" not in kwargs and "frame_width" not in kwargs:
            kwargs["frame_width"] = 400
        p = bokeh.plotting.figure(
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, **kwargs
        )

    # Plot the predictive intervals
    for i, ptile in enumerate(ptiles_str[: len(ptiles_str) // 2]):
        if diff == "ecdf":
            med = df_ecdf["50"].values
            x1_val = df_ecdf[ptile].values
            x1_post = med[med > x1_val.max()]
            x2_val = df_ecdf[ptiles_str[-i - 1]].values
            x2_pre = med[med < x2_val.min()]

            x1 = np.concatenate((x1_val, x1_post))
            y1 = np.concatenate((y, np.ones_like(x1_post)))
            y1 -= _ecdf_arbitrary_points(df_ecdf["50"].values, x1)

            x2 = np.concatenate((x2_pre, x2_val))
            y2 = np.concatenate((np.zeros_like(x2_pre), y))
            y2 -= _ecdf_arbitrary_points(df_ecdf["50"].values, x2)

            x1, y1 = cdf_to_staircase(x1, y1)
            x2, y2 = cdf_to_staircase(x2, y2)
        else:
            if diff == "iecdf":
                df_ecdf[ptile] -= df_ecdf["50"]
                df_ecdf[ptiles_str[-i - 1]] -= df_ecdf["50"]

            x1, y1 = cdf_to_staircase(df_ecdf[ptile].values, y)
            x2, y2 = cdf_to_staircase(df_ecdf[ptiles_str[-i - 1]].values, y)

        fill_between(
            x1,
            y1,
            x2,
            y2,
            p=p,
            show_line=False,
            patch_kwargs=dict(color=colors[color][i], alpha=0.5),
        )

    # The median as a solid line
    if diff == "ecdf":
        p.ray(
            df_ecdf["50"].min(), 0.0, None, np.pi, line_width=2, color=colors[color][-1]
        )
        p.ray(df_ecdf["50"].min(), 0.0, None, 0, line_width=2, color=colors[color][-1])
    elif diff == "iecdf":
        p.line([0.0, 0.0], [0.0, 1.0], line_width=2, color=colors[color][-1])
    else:
        x, y_median = cdf_to_staircase(df_ecdf["50"], y)
        p.line(x, y_median, line_width=2, color=colors[color][-1])
        p.ray(x.min(), 0.0, None, np.pi, line_width=2, color=colors[color][-1])
        p.ray(x.max(), diff is None, None, 0, line_width=2, color=colors[color][-1])

    # Overlay data set
    if data is not None:
        if data_staircase:
            if diff == "iecdf":
                data_plot -= df_ecdf["50"]
                x_data, y_data = cdf_to_staircase(data_plot, y)
            elif diff == "ecdf":
                med = df_ecdf["50"].values
                x_data = np.sort(np.unique(np.concatenate((data_plot, med))))
                data_ecdf = _ecdf_arbitrary_points(data_plot, x_data)
                med_ecdf = _ecdf_arbitrary_points(med, x_data)
                x_data, y_data = cdf_to_staircase(x_data, data_ecdf - med_ecdf)
            else:
                x_data, y_data = cdf_to_staircase(data_plot, y)

            p.line(x_data, y_data, color=data_color, line_width=data_size)

            # Extend to infinity
            if diff != "iecdf":
                p.ray(
                    x_data.min(),
                    0.0,
                    None,
                    np.pi,
                    line_width=data_size,
                    color=data_color,
                )
                p.ray(
                    x_data.max(),
                    diff is None,
                    None,
                    0,
                    line_width=data_size,
                    color=data_color,
                )
        else:
            if diff == "iecdf":
                p.circle(data_plot - df_ecdf["50"], y, color=data_color, size=data_size)
            elif diff == "ecdf":
                p.circle(
                    data_plot,
                    y - _ecdf_arbitrary_points(df_ecdf["50"].values, data_plot),
                    color=data_color,
                    size=data_size,
                )
            else:
                p.circle(data_plot, y, color=data_color, size=data_size)

    return p


def predictive_regression(
    samples,
    samples_x,
    data=None,
    diff=False,
    percentiles=[95, 68],
    color="blue",
    data_kwargs={},
    p=None,
    **kwargs,
):
    """Plot a predictive regression plot from samples.

    Parameters
    ----------
    samples : Numpy array, shape (n_samples, n_x) or xarray DataArray
        Numpy array containing predictive samples of y-values.
    sample_x : Numpy array, shape (n_x,)
    data : Numpy array, shape (n, 2) or xarray DataArray
        If not None, the measured data. The first column is the x-data,
        and the second the y-data. These are plotted as points over the
        predictive plot.
    diff : bool, default True
        If True, the predictive y-values minus the median of the
        predictive y-values are plotted.
    percentiles : list, default [95, 68]
        Percentiles for making colored envelopes for confidence
        intervals for the predictive ECDFs. Maximally four can be
        specified.
    color : str, default 'blue'
        One of ['green', 'blue', 'red', 'gray', 'purple', 'orange'].
        There are used to make the color scheme of shading of
        percentiles.
    data_kwargs : dict
        Any kwargs to be passed to p.circle() when plotting the data
        points.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    kwargs
        All other kwargs are passed to bokeh.plotting.figure().

    Returns
    -------
    output : Bokeh figure
        Figure populated with glyphs describing range of values for the
        the samples. The shading goes according to percentiles of
        samples, with the median plotted as line in the middle.
    """
    if type(samples) != np.ndarray:
        if type(samples) == xarray.core.dataarray.DataArray:
            samples = samples.squeeze().values
        else:
            raise RuntimeError("Samples can only be Numpy arrays and xarrays.")

    if type(samples_x) != np.ndarray:
        if type(samples_x) == xarray.core.dataarray.DataArray:
            samples_x = samples_x.squeeze().values
        else:
            raise RuntimeError("`samples_x` can only be Numpy array or xarray.")

    if len(percentiles) > 4:
        raise RuntimeError("Can specify maximally four percentiles.")

    # Build ptiles
    percentiles = np.sort(percentiles)[::-1]
    ptiles = [pt for pt in percentiles if pt > 0]
    ptiles = (
        [50 - pt / 2 for pt in percentiles]
        + [50]
        + [50 + pt / 2 for pt in percentiles[::-1]]
    )
    ptiles_str = [str(pt) for pt in ptiles]

    if color not in ["green", "blue", "red", "gray", "purple", "orange", "betancourt"]:
        raise RuntimeError(
            "Only allowed colors are 'green', 'blue', 'red', 'gray', 'purple', 'orange'"
        )

    colors = {
        "blue": ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
        "green": ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
        "red": ["#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
        "orange": ["#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04"],
        "purple": ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
        "gray": ["#bdbdbd", "#969696", "#737373", "#525252", "#252525"],
        "betancourt": [
            "#DCBCBC",
            "#C79999",
            "#B97C7C",
            "#A25050",
            "#8F2727",
            "#7C0000",
        ],
    }

    if samples.shape[1] != len(samples_x):
        raise ValueError(
            "`samples_x must have the same number of entries as `samples` does columns."
        )

    # It's useful to have data as a data frame
    if data is not None:
        if type(data) == tuple and len(data) == 2 and len(data[0]) == len(data[1]):
            data = np.vstack(data).transpose()
        df_data = pd.DataFrame(data=data, columns=["__data_x", "__data_y"])
        df_data = df_data.sort_values(by="__data_x")

    # Make sure all entries in x-data in samples_x
    if diff:
        if len(samples_x) != len(df_data) or not np.allclose(
            np.sort(samples_x), df_data["__data_x"].values
        ):
            raise ValueError(
                "If `diff == True`, then samples_x must match the x-values of `data`."
            )

    df_pred = pd.DataFrame(
        data=np.percentile(samples, ptiles, axis=0).transpose(),
        columns=[str(ptile) for ptile in ptiles],
    )
    df_pred["__x"] = samples_x
    df_pred = df_pred.sort_values(by="__x")

    if p is None:
        x_axis_label = kwargs.pop("x_axis_label", "x")
        y_axis_label = kwargs.pop("y_axis_label", "y difference" if diff else "y")

        if "plot_height" not in kwargs and "frame_height" not in kwargs:
            kwargs["frame_height"] = 325
        if "plot_width" not in kwargs and "frame_width" not in kwargs:
            kwargs["frame_width"] = 400
        p = bokeh.plotting.figure(
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, **kwargs
        )

    for i, ptile in enumerate(ptiles_str[: len(ptiles_str) // 2]):
        if diff:
            y1 = df_pred[ptile] - df_pred["50"]
            y2 = df_pred[ptiles_str[-i - 1]] - df_pred["50"]
        else:
            y1 = df_pred[ptile]
            y2 = df_pred[ptiles_str[-i - 1]]

        fill_between(
            x1=df_pred["__x"],
            x2=df_pred["__x"],
            y1=y1,
            y2=y2,
            p=p,
            show_line=False,
            patch_kwargs=dict(fill_color=colors[color][i]),
        )

    # The median as a solid line
    if diff:
        p.line(
            df_pred["__x"],
            np.zeros_like(samples_x),
            line_width=2,
            color=colors[color][-1],
        )
    else:
        p.line(df_pred["__x"], df_pred["50"], line_width=2, color=colors[color][-1])

    # Overlay data set
    if data is not None:
        data_color = data_kwargs.pop("color", "orange")
        data_alpha = data_kwargs.pop("alpha", 1.0)
        data_size = data_kwargs.pop("size", 2)
        if diff:
            p.circle(
                df_data["__data_x"],
                df_data["__data_y"] - df_pred["50"],
                color=data_color,
                size=data_size,
                alpha=data_alpha,
                **data_kwargs,
            )
        else:
            p.circle(
                df_data["__data_x"],
                df_data["__data_y"],
                color=data_color,
                size=data_size,
                alpha=data_alpha,
                **data_kwargs,
            )

    return p


def sbc_rank_ecdf(
    sbc_output=None,
    parameters=None,
    diff=True,
    ptile=99.0,
    bootstrap_envelope=False,
    n_bs_reps=None,
    show_envelope=True,
    show_envelope_line=True,
    color_by_warning_code=False,
    staircase=False,
    p=None,
    marker_kwargs={},
    envelope_patch_kwargs={},
    envelope_line_kwargs={},
    palette=None,
    show_legend=True,
    **kwargs,
):
    """Make a rank ECDF plot from simulation-based calibration.

    Parameters
    ----------
    sbc_output : DataFrame
        Output of bebi103.stan.sbc() containing results from an SBC
        calculation.
    parameters : list of str, or None (default)
        List of parameters to include in the SBC rank ECDF plot. If
        None, use all parameters. For multidimensional parameters, each
        entry must be given separately, e.g.,
        `['alpha[0]', 'alpha[1]', 'beta[0,1]']`.
    diff : bool, default True
        If True, plot the ECDF minus the ECDF of a Uniform distribution.
        Otherwise, plot the ECDF of the rank statistic from SBC.
    ptile : float, default 99
        Which precentile to use as the envelope in the plot.
    bootstrap_envelope : bool, default False
        If True, use bootstrapping on the appropriate Uniform
        distribution to compute the envelope. Otherwise, use the
        Gaussian approximation for the envelope.
    n_bs_reps : bool, default None
        Number of bootstrap replicates to use when computing the
        envelope. If None, n_bs_reps is determined from the formula
        int(max(n, max(L+1, 100/(100-ptile))) * 100), where n is the
        number of simulations used in the SBC calculation.
    show_envelope : bool, default True
        If True, display the envelope encompassing the ptile percent
        confidence interval for the SBC ECDF.
    show_envelope_line : bool, default True
        If True, and `show_envelope` is also True, plot a line around
        the envelope.
    color_by_warning_code : bool, default False
        If True, color glyphs by diagnostics warning code instead of
        coloring the glyphs by parameter
    staircase : bool, default False
        If True, plot the ECDF as a staircase. Otherwise, plot with
        dots.
    p : bokeh.plotting.Figure instance, default None
        Plot to which to add the SBC rank ECDF plot. If None, create a
        new figure.
    marker_kwargs : dict, default {}
        Dictionary of kwargs to pass to `p.circle()` or `p.line()` when
        plotting the SBC ECDF.
    envelope_patch_kwargs : dict
        Any kwargs passed into p.patch(), which generates the fill of
        the envelope.
    envelope_line_kwargs : dict
        Any kwargs passed into p.line() in generating the line around
        the fill of the envelope.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        colorcet.b_glasbey_category10 from the colorcet package.
    show_legend : bool, default True
        If True, show legend.
    kwargs : dict
        Any kwargs passed to `bokeh.plotting.figure()` when creating the
        plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        A plot containing the SBC plot.

    Notes
    -----
    You can see example SBC ECDF plots in Fig. 14 b and c in this
    paper: https://arxiv.org/abs/1804.06788
    """
    if sbc_output is None:
        raise RuntimeError("Argument `sbc_output` must be specified.")

    # Defaults
    if palette is None:
        palette = colorcet.b_glasbey_category10
    elif palette not in [list, tuple]:
        palette = [palette]

    if "x_axis_label" not in kwargs:
        kwargs["x_axis_label"] = "rank statistic"
    if "y_axis_label" not in kwargs:
        kwargs["y_axis_label"] = "ECDF difference" if diff else "ECDF"

    if "plot_height" not in kwargs and "frame_height" not in kwargs:
        kwargs["frame_height"] = 275
    if "plot_width" not in kwargs and "frame_width" not in kwargs:
        kwargs["frame_width"] = 450
    toolbar_location = kwargs.pop("toolbar_location", "above")

    if "fill_color" not in envelope_patch_kwargs:
        envelope_patch_kwargs["fill_color"] = "gray"
    if "fill_alpha" not in envelope_patch_kwargs:
        envelope_patch_kwargs["fill_alpha"] = 0.5
    if "line_color" not in envelope_line_kwargs:
        envelope_line_kwargs["line_color"] = "gray"

    if "color" in "marker_kwargs" and color_by_warning_code:
        raise RuntimeError(
            "Cannot specify marker color when `color_by_warning_code` is True."
        )
    if staircase and color_by_warning_code:
        raise RuntimeError("Cannot color by warning code for staircase ECDFs.")

    if parameters is None:
        parameters = list(sbc_output["parameter"].unique())
    elif type(parameters) not in [list, tuple]:
        parameters = [parameters]

    L = sbc_output["L"].iloc[0]
    df = sbc_output.loc[
        sbc_output["parameter"].isin(parameters),
        ["parameter", "rank_statistic", "warning_code"],
    ]
    n = (df["parameter"] == df["parameter"].unique()[0]).sum()

    if show_envelope:
        x, y_low, y_high = _sbc_rank_envelope(
            L,
            n,
            ptile=ptile,
            diff=diff,
            bootstrap=bootstrap_envelope,
            n_bs_reps=n_bs_reps,
        )
        p = fill_between(
            x1=x,
            x2=x,
            y1=y_high,
            y2=y_low,
            patch_kwargs=envelope_patch_kwargs,
            line_kwargs=envelope_line_kwargs,
            show_line=show_envelope_line,
            p=p,
            toolbar_location=toolbar_location,
            **kwargs,
        )
    else:
        p = bokeh.plotting.figure(toolbar_location=toolbar_location, **kwargs)

    if staircase:
        dfs = []
        for param in parameters:
            if diff:
                x_data, y_data = _ecdf_diff(
                    df.loc[df["parameter"] == param, "rank_statistic"],
                    L,
                    staircase=True,
                )
            else:
                x_data, y_data = _ecdf_vals(
                    df.loc[df["parameter"] == param, "rank_statistic"], staircase=True
                )
            dfs.append(
                pd.DataFrame(
                    data=dict(rank_statistic=x_data, __ECDF=y_data, parameter=param)
                )
            )
        df = pd.concat(dfs, ignore_index=True)
    else:
        df["__ECDF"] = df.groupby("parameter")["rank_statistic"].transform(_ecdf_y)
        if diff:
            df["__ECDF"] -= (df["rank_statistic"] + 1) / L

    if staircase:
        color = marker_kwargs.pop("color", palette)
        if type(color) == str:
            color = [color] * len(parameters)
    elif "color" not in marker_kwargs:
        color = palette
    else:
        color = [marker_kwargs.pop("color")] * len(parameters)

    if color_by_warning_code:
        if len(color) < len(df["warning_code"].unique()):
            raise RuntimeError(
                "Not enough colors in palette to cover all warning codes."
            )
    elif len(color) < len(parameters):
        raise RuntimeError("Not enough colors in palette to cover all parameters.")

    if staircase:
        plot_cmd = p.line
    else:
        plot_cmd = p.circle

    if show_legend:
        legend_items = []

    if color_by_warning_code:
        for i, (warning_code, g) in enumerate(df.groupby("warning_code")):
            if show_legend:
                legend_items.append(
                    (
                        str(warning_code),
                        [
                            plot_cmd(
                                source=g,
                                x="rank_statistic",
                                y="__ECDF",
                                color=color[i],
                                **marker_kwargs,
                            )
                        ],
                    )
                )
            else:
                plot_cmd(
                    source=g,
                    x="rank_statistic",
                    y="__ECDF",
                    color=color[i],
                    **marker_kwargs,
                )
    else:
        for i, (param, g) in enumerate(df.groupby("parameter")):
            if show_legend:
                legend_items.append(
                    (
                        param,
                        [
                            plot_cmd(
                                source=g,
                                x="rank_statistic",
                                y="__ECDF",
                                color=color[i],
                                **marker_kwargs,
                            )
                        ],
                    )
                )
            else:
                plot_cmd(
                    source=g,
                    x="rank_statistic",
                    y="__ECDF",
                    color=color[i],
                    **marker_kwargs,
                )

    if show_legend:
        legend = bokeh.models.Legend(items=legend_items)
        p.add_layout(legend, "right")
        p.legend.click_policy = "hide"

    return p


def parcoord(
    samples=None,
    parameters=None,
    palette=None,
    omit=None,
    include_ppc=False,
    include_log_lik=False,
    transformation=None,
    color_by_chain=False,
    line_kwargs={},
    divergence_kwargs={},
    xtick_label_orientation=0.7853981633974483,
    **kwargs,
):
    """
    Make a parallel coordinate plot of MCMC samples. The x-axis is the
    parameter name and the y-axis is the value of the parameter,
    possibly transformed to so the scale of all parameters are similar.

    Parameters
    ----------
    samples : ArviZ InferenceData instance or xarray Dataset instance
        Result of MCMC sampling.
    parameters : list of str, or None (default)
        Names of parameters to include in the plot. If None, use all
        parameters. For multidimensional parameters, each entry must be
        given separately, e.g., `['alpha[0]', 'alpha[1]', 'beta[0,1]']`.
        If a given entry is a 2-tuple, the first entry is the variable
        name, and the second entry is the label for the parameter in
        plots.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        colorcet.b_glasbey_category10 from the colorcet package.
    omit : str, re.Pattern, or list or tuple of str and re.Pattern
        If `parameters` is not provided, all parameters are used in the
        parallel coordinate plot. We often want to ignore samples of
        some variables. For each string entry in `omit`, the variable
        given by the string is omitted. For each entry that is a
        compiled regular expression patters (`re.Pattern`), any variable
        name matching the pattern is omitted.
    include_ppc : bool, default False
        If True, include variables ending in _ppc, which denotes
        posterior predictive checks, in the plot.
    include_log_lik: bool, default False
        If True, include variables starting with log_lik or loglik.
        These denote log-likelihood contributions.
    transformation : function, str, or dict, default None
        A transformation to apply to each set of samples. The function
        must take a single array as input and return an array as the
        same size. If None, nor transformation is done. If a dictionary,
        each key is the variable name and the corresponding value is a
        function for the transformation of that variable. Alternatively,
        if `transformation` is `'minmax'`, the data are scaled to range
        from zero to one, or if `transformation` is `'rank'`, the rank
        of the each data is used.
    color_by_chain : bool, default False
        If True, color the lines by chain.
    line_kwargs: dict
        Dictionary of kwargs to be passed to `p.multi_line()` in making
        the plot of non-divergent samples.
    divergence_kwargs: dict
        Dictionary of kwargs to be passed to `p.multi_line()` in making
        the plot of divergent samples.
    xtick_label_orientation : str or float, default π/4.
        Orientation of x tick labels. In some plots, horizontally
        labeled ticks will have label clashes, and this can fix that.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()` when
        instantiating the figure.

    Returns
    -------
    output : Bokeh plot
        Parallel coordinates plot.

    """
    if parameters is not None and omit is not None:
        raise RuntimeError("At least one of `parameters` and `omit` must be None.")

    omit = _parse_omit(omit, include_ppc, include_log_lik)

    # Default properties
    if palette is None:
        palette = colorcet.b_glasbey_category10
    line_width = line_kwargs.pop("line_width", 0.5)
    alpha = line_kwargs.pop("alpha", 0.02)
    line_join = line_kwargs.pop("line_join", "bevel")
    if "color" in line_kwargs and color_by_chain:
        raise RuntimeError(
            "Cannot specify line color and also color by chain. If coloring by chain, use `palette` kwarg to specify color scheme."
        )
    color = line_kwargs.pop("color", "black")

    divergence_line_join = divergence_kwargs.pop("line_join", "bevel")
    divergence_line_width = divergence_kwargs.pop("line_width", 1)
    divergence_color = divergence_kwargs.pop("color", "orange")
    divergence_alpha = divergence_kwargs.pop("alpha", 1)

    if "plot_height" not in kwargs and "frame_height" not in kwargs:
        kwargs["frame_height"] = 175
    if "plot_width" not in kwargs and "frame_width" not in kwargs:
        kwargs["frame_width"] = 600
    toolbar_location = kwargs.pop("toolbar_location", "above")
    if "x_range" in kwargs:
        raise RuntimeError("Cannot specify x_range; this is inferred.")

    if not color_by_chain:
        palette = [color] * len(palette)

    if type(samples) != az.data.inference_data.InferenceData:
        raise RuntimeError("Input must be an ArviZ InferenceData instance.")

    if not hasattr(samples, "posterior"):
        raise RuntimeError("Input samples do not have 'posterior' group.")

    if not (
        hasattr(samples, "sample_stats") and hasattr(samples.sample_stats, "diverging")
    ):
        warnings.warn("No divergence information available.")

    parameters, df = stan._samples_parameters_to_df(samples, parameters, omit)
    parameters, labels = _parse_parameters(parameters)
    df.rename(columns={param: label for param, label in zip(parameters, labels)})

    if transformation == "minmax":
        transformation = {
            param: lambda x: (x - x.min()) / (x.max() - x.min())
            if x.min() < x.max()
            else 0.0
            for param in labels
        }
    elif transformation == "rank":
        transformation = {param: lambda x: st.rankdata(x) for param in labels}

    if transformation is None:
        transformation = {param: lambda x: x for param in labels}

    if callable(transformation) or transformation is None:
        transformation = {param: transformation for param in labels}

    for col, trans in transformation.items():
        df[col] = trans(df[col])
    df = df.melt(id_vars=["diverging__", "chain__", "draw__"])

    p = bokeh.plotting.figure(
        x_range=bokeh.models.FactorRange(*labels),
        toolbar_location=toolbar_location,
        **kwargs,
    )

    # Plots for samples that were not divergent
    ys = np.array(
        [
            group["value"].values
            for _, group in df.loc[~df["diverging__"]].groupby(["chain__", "draw__"])
        ]
    )
    if len(ys) > 0:
        ys = [y for y in ys]
        xs = [list(df["variable"].unique())] * len(ys)

        p.multi_line(
            xs,
            ys,
            line_width=line_width,
            alpha=alpha,
            line_join=line_join,
            color=[palette[i % len(palette)] for i in range(len(ys))],
            **line_kwargs,
        )

    # Plots for samples that were divergent
    ys = np.array(
        [
            group["value"].values
            for _, group in df.loc[df["diverging__"]].groupby(["chain__", "draw__"])
        ]
    )
    if len(ys) > 0:
        ys = [y for y in ys]
        xs = [list(df["variable"].unique())] * len(ys)

        p.multi_line(
            xs,
            ys,
            alpha=divergence_alpha,
            line_join=line_join,
            color=divergence_color,
            line_width=divergence_line_width,
            **divergence_kwargs,
        )

    p.xaxis.major_label_orientation = xtick_label_orientation

    return p


def trace(
    samples=None,
    parameters=None,
    palette=None,
    omit=None,
    include_ppc=False,
    include_log_lik=False,
    line_kwargs={},
    **kwargs,
):
    """
    Make a trace plot of MCMC samples.

    Parameters
    ----------
    samples : ArviZ InferenceData instance or xarray Dataset instance
        Result of MCMC sampling.
    parameters : list of str, or None (default)
        Names of parameters to include in the plot. If None, use all
        parameters. For multidimensional parameters, each entry must be
        given separately, e.g., `['alpha[0]', 'alpha[1]', 'beta[0,1]']`.
        If a given entry is a 2-tuple, the first entry is the variable
        name, and the second entry is the label for the parameter in
        plots.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        colorcet.b_glasbey_category10 from the colorcet package.
    omit : str, re.Pattern, or list or tuple of str and re.Pattern
        If `parameters` is not provided, all parameters are used in the
        parallel coordinate plot. We often want to ignore samples of
        some variables. For each string entry in `omit`, the variable
        given by the string is omitted. For each entry that is a
        compiled regular expression patters (`re.Pattern`), any variable
        name matching the pattern is omitted.
    include_ppc : bool, default False
        If True, include variables ending in _ppc, which denotes
        posterior predictive checks, in the plot.
    include_log_lik: bool, default False
        If True, include variables starting with log_lik or loglik.
        These denote log-likelihood contributions.
    line_kwargs: dict
        Dictionary of kwargs to be passed to `p.multi_line()` in making
        the plot of non-divergent samples.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()`.

    Returns
    -------
    output : Bokeh gridplot
        Set of chain traces as a Bokeh gridplot.
    """
    if parameters is not None and omit is not None:
        raise RuntimeError("At least one of `parameters` and `omit` must be None.")

    omit = _parse_omit(omit, include_ppc, include_log_lik)

    if type(samples) != az.data.inference_data.InferenceData:
        raise RuntimeError("Input must be an ArviZ InferenceData instance.")

    if not hasattr(samples, "posterior"):
        raise RuntimeError("Input samples do not have 'posterior' group.")

    parameters, df = stan._samples_parameters_to_df(samples, parameters, omit)
    parameters, labels = _parse_parameters(parameters)

    # Default properties
    if palette is None:
        palette = colorcet.b_glasbey_category10
    line_width = line_kwargs.pop("line_width", 0.5)
    alpha = line_kwargs.pop("alpha", 0.5)
    line_join = line_kwargs.pop("line_join", "bevel")
    if "color" in line_kwargs:
        raise RuntimeError(
            "Cannot specify line color. Specify color scheme with `palette` kwarg."
        )

    if "plot_height" not in kwargs and "frame_height" not in kwargs:
        kwargs["frame_height"] = 150
    if "plot_width" not in kwargs and "frame_width" not in kwargs:
        kwargs["frame_width"] = 600
    x_axis_label = kwargs.pop("x_axis_label", "step")
    if "y_axis_label" in kwargs:
        raise RuntimeError(
            "`y_axis_label` cannot be specified; it is inferred from samples."
        )
    if "x_range" not in kwargs:
        kwargs["x_range"] = [df["draw__"].min(), df["draw__"].max()]

    plots = []
    grouped = df.groupby("chain__")
    for i, (var, label) in enumerate(zip(parameters, labels)):
        p = bokeh.plotting.figure(
            x_axis_label=x_axis_label, y_axis_label=label, **kwargs
        )
        for i, (chain, group) in enumerate(grouped):
            p.line(
                group["draw__"],
                group[var],
                line_width=line_width,
                line_join=line_join,
                color=palette[i],
                *line_kwargs,
            )

        plots.append(p)

    if len(plots) == 1:
        return plots[0]

    # Link ranges
    for i, p in enumerate(plots[:-1]):
        plots[i].x_range = plots[-1].x_range

    return bokeh.layouts.gridplot(plots, ncols=1)


def corner(
    samples=None,
    parameters=None,
    palette=None,
    omit=None,
    include_ppc=False,
    include_log_lik=False,
    max_plotted=10000,
    datashade=False,
    frame_width=None,
    frame_height=None,
    plot_ecdf=False,
    ecdf_staircase=False,
    cmap="black",
    color_by_chain=False,
    divergence_color="orange",
    alpha=None,
    single_var_color="black",
    bins="freedman-diaconis",
    show_contours=False,
    contour_color="black",
    bins_2d=50,
    levels=None,
    weights=None,
    smooth=0.02,
    extend_contour_domain=False,
    xtick_label_orientation="horizontal",
):
    """
    Make a corner plot of sampling results. Heavily influenced by the
    corner package by Dan Foreman-Mackey.

    Parameters
    ----------
    samples : Numpy array, Pandas DataFrame, or ArviZ InferenceData
        Results of sampling. If a Numpy array or Pandas DataFrame, each
        row is a sample and each column corresponds to a variable.
    parameters : list
        List of variables as strings included in `samples` to construct
        corner plot. If None, use all parameters. The entries correspond
        to column headings if `samples` is in a Pandas DataFrame. If the
        input is a Numpy array, `parameters` is a list of indices of
        columns to use in the plot. If `samples` is an ArviZ
        InferenceData instance, `parameters` contains the names of
        parameters to include in the  plot. For multidimensional
        parameters, each entry must be given separately, e.g.,
        `['alpha[0]', 'alpha[1]', 'beta[0,1]']`. If a given entry is a
        2-tuple, the first entry is the variable name, and the second
        entry is the label for the parameter in plots.
    palette : list of strings of hex colors, or single hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by Altair. Ignored is
        `color_by_chain` is False.
    omit : str, re.Pattern, or list or tuple of str and re.Pattern
        If `parameters` is not provided, all parameters are used in the
        parallel coordinate plot. We often want to ignore samples of
        some variables. For each string entry in `omit`, the variable
        given by the string is omitted. For each entry that is a
        compiled regular expression patters (`re.Pattern`), any variable
        name matching the pattern is omitted.
    include_ppc : bool, default False
        If True, include variables ending in _ppc, which denotes
        posterior predictive checks, in the plot.
    include_log_lik: bool, default False
        If True, include variables starting with log_lik or loglik.
        These denote log-likelihood contributions.
    max_plotted : int, default 10000
        Maximum number of points to be plotted.
    datashade : bool, default False
        Whether or not to convert sampled points to a raster image using
        Datashader.
    frame_width : int or None, default None
        Width of each plot in the corner plot in pixels. Default is set
        based on number of parameters plotted. If None and
        `frame_height` is specificed, `frame_width` is set to
        `frame_height`.
    frame_height : int or None, default None
        Height of each plot in the corner plot in pixels. Default is set
        based on number of parameters plotted. If None and `frame_width`
        is specificed, `frame_height` is set to `frame_width`.
    plot_ecdf : bool, default False
        If True, plot ECDFs of samples on the diagonal of the corner
        plot. If False, histograms are plotted.
    ecdf_staircase : bool, default False
        If True, plot the ECDF in "staircase" style. Otherwise, plot as
        dots. Ignored if `plot_ecdf` is False.
    cmap : str, default 'black'
        Valid colormap string for DataShader or for coloring Bokeh
        glyphs.
    color_by_chain : bool, default False
        If True, color the glyphs by chain index.
    divergence_color : str, default 'orange'
        Color to use for showing points where the sampler experienced a
        divergence.
    alpha : float or None, default None
        Opacity of glyphs. If None, inferred.
    single_var_color : str, default 'black'
        Color of histogram or ECDF lines.
    bins : int, array_like, or str, default 'freedman-diaconis'
        If int or array_like, setting for `bins` kwarg to be passed to
        `np.histogram()`. If 'exact', then each unique value in the
        data gets its own bin. If 'integer', then integer data is
        assumed and each integer gets its own bin. If 'sqrt', uses the
        square root rule to determine number of bins. If
        `freedman-diaconis`, uses the Freedman-Diaconis rule for number
        of bins. Ignored if `plot_ecdf` is True.
    show_contours : bool, default False
        If True, show contour plot on top of samples.
    contour_color : str, default 'black'
        Color of contour lines
    bins_2d : int, default 50
        Number of bins in each direction for binning 2D histograms when
        computing contours.
    levels : list of floats, default None
        Levels to use when constructing contours. By default, these are
        chosen according to this principle from Dan Foreman-Mackey:
        http://corner.readthedocs.io/en/latest/pages/sigmas.html
    weights : default None
        Value to pass as `weights` kwarg to np.histogram2d(), used in
        constructing contours.
    smooth : int or None, default 1
        Width of smoothing kernel for making contours.
    extend_contour_domain : bool, default False
        If True, extend the domain of the contours a little bit beyond
        the extend of the samples. This is done in the corner package,
        but I prefer not to do it.
    xtick_label_orientation : str or float, default 'horizontal'.
        Orientation of x tick labels. In some plots, horizontally
        labeled ticks will have label clashes, and this can fix that.
        A preferred alternative to 'horizontal' is `np.pi/4`. Be aware,
        though, that non-horizontal tick labels may disrupt alignment
        of some of the plots in the corner plot.

    Returns
    -------
    output : Bokeh gridplot
        Corner plot as a Bokeh gridplot.
    """
    if parameters is not None and omit is not None:
        raise RuntimeError("At least one of `parameters` and `omit` must be None.")

    omit = _parse_omit(omit, include_ppc, include_log_lik)

    # Datashading not implemented
    if datashade:
        raise NotImplementedError("Datashading is not yet implemented.")

    # Tools, also allowing linked brushing
    tools = "pan,box_zoom,wheel_zoom,box_select,lasso_select,save,reset"

    # Default properties
    if palette is None:
        palette = colorcet.b_glasbey_category10

    if color_by_chain:
        if cmap not in ["black", None]:
            warnings.warn("Ignoring cmap values to color by chain.")

    if divergence_color is None:
        divergence_color = cmap

    if type(samples) == pd.core.frame.DataFrame:
        df = samples
        if parameters is None:
            parameters = [
                col
                for col in df.columns
                if stan._screen_param(col, omit) and (len(col) < 2 or col[-2:] != "__")
            ]
    elif type(samples) == np.ndarray:
        if parameters is None:
            parameters = list(range(samples.shape[1]))
        for param in parameters:
            if (type(param) in [tuple, list] and type(param[0]) != int) or type(
                param
            ) != int:
                raise RuntimeError(
                    "When `samples` is inputted as a Numpy array, `parameters` must be"
                    " a list of indices."
                )
        new_parameters = [
            str(param) if type(param) == int else param[1] for param in parameters
        ]
        df = pd.DataFrame(samples[:, parameters], columns=new_parameters)
        parameters = new_parameters
    else:
        parameters, df = stan._samples_parameters_to_df(samples, parameters, omit)

    parameters, labels = _parse_parameters(parameters)

    # Set default frame width and height.
    if frame_height is None:
        if frame_width is None:
            default_dim = 25 * (9 - len(parameters)) if len(parameters) < 5 else 100
            frame_height = default_dim
            frame_width = default_dim
        else:
            frame_height = frame_width
    elif frame_width is None:
        frame_width = frame_height

    if len(parameters) > 6:
        raise RuntimeError("For space purposes, can show only six parameters.")

    if color_by_chain:
        # Have to convert datatype to string to play nice with Bokeh
        df["chain__"] = df["chain__"].astype(str)

        factors = tuple(df["chain__"].unique())
        cmap = bokeh.transform.factor_cmap("chain__", palette=palette, factors=factors)

    # Add dummy divergent column if no divergence information is given
    if "diverging__" not in df.columns:
        df = df.copy()
        df["diverging__"] = 0

    # Add dummy chain column if no divergence information is given
    if "chain__" not in df.columns:
        df = df.copy()
        df["chain__"] = 0

    for col in parameters:
        if col not in df.columns:
            raise RuntimeError("Column " + col + " not in the columns of DataFrame.")

    if labels is None:
        labels = parameters
    elif len(labels) != len(parameters):
        raise RuntimeError("len(parameters) must equal len(labels)")

    if plot_ecdf and not ecdf_staircase:
        for param in parameters:
            df[f"__ECDF_{param}"] = df[param].rank(method="first") / len(df)

    n_nondivergent = np.sum(df["diverging__"] == 0)
    if n_nondivergent > max_plotted:
        inds = np.random.choice(
            np.arange(n_nondivergent), replace=False, size=max_plotted
        )
    else:
        inds = np.arange(np.sum(df["diverging__"] == 0))

    if alpha is None:
        if len(inds) < 100:
            alpha = 1
        elif len(inds) > 10000:
            alpha = 0.02
        else:
            alpha = np.exp(-(np.log10(len(inds)) - 2) / (-2 / np.log(0.02)))

    # Set up column data sources to allow linked brushing
    cds = bokeh.models.ColumnDataSource(df.loc[df["diverging__"] == 0, :].iloc[inds, :])
    cds_div = bokeh.models.ColumnDataSource(df.loc[df["diverging__"] == 1, :])

    # Just make a single plot if only one parameter
    if len(parameters) == 1:
        x = parameters[0]
        if plot_ecdf:
            p = bokeh.plotting.figure(
                frame_width=frame_width,
                frame_height=frame_height,
                x_axis_label=labels[0],
                y_axis_label="ECDF",
            )
            if ecdf_staircase:
                p = _ecdf(
                    df[x].iloc[inds],
                    staircase=True,
                    line_width=2,
                    line_color=single_var_color,
                    p=p,
                )
            else:
                p.circle(source=cds, x=x, y=f"__ECDF_{x}", color=single_var_color)
                p.circle(source=cds_div, x=x, y=f"__ECDF_{x}", color=divergence_color)
        else:
            p = bokeh.plotting.figure(
                frame_width=frame_width,
                frame_height=frame_height,
                x_axis_label=labels[0],
                y_axis_label="density",
            )
            p = _histogram(
                df[x],
                bins=bins,
                density=True,
                line_kwargs=dict(line_width=2, line_color=single_var_color),
                p=p,
            )

        if frame_height < 200:
            p.toolbar_location = "above"
        p.xaxis.major_label_orientation = xtick_label_orientation

        return p

    plots = [[None for _ in range(len(parameters))] for _ in range(len(parameters))]

    for i, j in zip(*np.tril_indices(len(parameters))):
        x = parameters[j]
        if i != j:
            y = parameters[i]
            x_range, y_range = _data_range(df, x, y)
            plots[i][j] = bokeh.plotting.figure(
                x_range=x_range,
                y_range=y_range,
                frame_width=frame_width,
                frame_height=frame_height,
                align="end",
                tools=tools,
            )

            plots[i][j].circle(
                source=cds,
                x=x,
                y=y,
                size=2,
                alpha=alpha,
                color=cmap,
                nonselection_fill_alpha=0,
                nonselection_line_alpha=0,
            )

            if divergence_color is not None:
                plots[i][j].circle(
                    source=cds_div,
                    x=x,
                    y=y,
                    size=2,
                    color=divergence_color,
                    nonselection_fill_alpha=0,
                    nonselection_line_alpha=0,
                )

            if show_contours:
                xs, ys = contour_lines_from_samples(
                    df[x].values,
                    df[y].values,
                    bins=bins_2d,
                    smooth=smooth,
                    levels=levels,
                    weights=weights,
                    extend_domain=extend_contour_domain,
                )
                plots[i][j].multi_line(xs, ys, line_color=contour_color, line_width=2)
        else:
            if plot_ecdf:
                x_range, _ = _data_range(df, x, x)
                plots[i][i] = bokeh.plotting.figure(
                    x_range=x_range,
                    y_range=[-0.02, 1.02],
                    frame_width=frame_width,
                    frame_height=frame_height,
                    align="end",
                    tools=tools,
                )
                if ecdf_staircase:
                    plots[i][i] = _ecdf(
                        df[x],
                        p=plots[i][i],
                        staircase=True,
                        line_width=2,
                        line_color=single_var_color,
                    )
                else:
                    plots[i][i].circle(
                        source=cds,
                        x=x,
                        y=f"__ECDF_{x}",
                        size=2,
                        color=cmap,
                        nonselection_fill_alpha=0,
                        nonselection_line_alpha=0,
                    )
                    plots[i][i].circle(
                        source=cds_div,
                        x=x,
                        y=f"__ECDF_{x}",
                        size=2,
                        color=divergence_color,
                        nonselection_fill_alpha=0,
                        nonselection_line_alpha=0,
                    )
            else:
                x_range, _ = _data_range(df, x, x)
                plots[i][i] = bokeh.plotting.figure(
                    x_range=x_range,
                    y_range=bokeh.models.DataRange1d(start=0.0),
                    frame_width=frame_width,
                    frame_height=frame_height,
                    align="end",
                    tools=tools,
                )
                bins_plot = _bins_to_np(df[x].values, bins)
                e0, f0 = _compute_histogram(df[x].values, bins=bins_plot, density=True)

                plots[i][i].line(e0, f0, line_width=2, color=single_var_color)
        plots[i][j].xaxis.major_label_orientation = xtick_label_orientation

    # Link axis ranges
    for i in range(1, len(parameters)):
        for j in range(i):
            plots[i][j].x_range = plots[j][j].x_range
            plots[i][j].y_range = plots[i][i].x_range

    # Label axes
    for i, label in enumerate(labels):
        plots[-1][i].xaxis.axis_label = label

    for i, label in enumerate(labels[1:]):
        plots[i + 1][0].yaxis.axis_label = label

    if plot_ecdf:
        plots[0][0].yaxis.axis_label = "ECDF"

    # Take off tick labels
    for i in range(len(parameters) - 1):
        for j in range(i + 1):
            plots[i][j].xaxis.major_label_text_font_size = "0pt"

    if not plot_ecdf:
        plots[0][0].yaxis.major_label_text_font_size = "0pt"

    for i in range(1, len(parameters)):
        for j in range(1, i + 1):
            plots[i][j].yaxis.major_label_text_font_size = "0pt"

    grid = bokeh.layouts.gridplot(plots, toolbar_location="left")

    return grid


def contour(
    X,
    Y,
    Z,
    levels=None,
    p=None,
    overlaid=False,
    cmap=None,
    overlay_grid=False,
    fill=False,
    fill_palette=None,
    fill_alpha=0.75,
    line_kwargs={},
    **kwargs,
):
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
    cmap : str or list of hex colors, default None
        If `im` is an intensity image, `cmap` is a mapping of
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `cmap` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    overlay_grid : bool, default False
        If True, faintly overlay the grid on top of image. Ignored if
        overlaid is False.
    line_kwargs : dict, default {}
        Keyword arguments passed to `p.multiline()` for rendering the
        contour.
    kwargs
        Any kwargs to be passed to `bokeh.plotting.figure()`.

    Returns
    -------
    output : Bokeh plotting object
        Plot populated with contours, possible with an image.
    """
    if len(X.shape) != 2 or Y.shape != X.shape or Z.shape != X.shape:
        raise RuntimeError("All arrays must be 2D and of same shape.")

    if overlaid and p is not None:
        raise RuntimeError("Cannot specify `p` if showing image.")

    # Set defaults
    x_axis_label = kwargs.pop("x_axis_label", "x")
    y_axis_label = kwargs.pop("y_axis_label", "y")

    if "line_color" not in line_kwargs:
        if overlaid:
            line_kwargs["line_color"] = "white"
        else:
            line_kwargs["line_color"] = "black"

    line_width = line_kwargs.pop("line_width", 2)

    if p is None:
        if overlaid:
            frame_height = kwargs.pop("frame_height", 300)
            frame_width = kwargs.pop("frame_width", 300)
            title = kwargs.pop("title", None)
            p = image.imshow(
                Z,
                cmap=cmap,
                frame_height=frame_height,
                frame_width=frame_width,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                x_range=[X.min(), X.max()],
                y_range=[Y.min(), Y.max()],
                no_ticks=False,
                flip=False,
                return_im=False,
            )
        else:
            if "plot_height" not in kwargs and "frame_height" not in kwargs:
                kwargs["frame_height"] = 300
            if "plot_width" not in kwargs and "frame_width" not in kwargs:
                kwargs["frame_width"] = 300
            p = bokeh.plotting.figure(
                x_axis_label=x_axis_label, y_axis_label=y_axis_label, **kwargs
            )

    # Set default levels
    if levels is None:
        levels = 1.0 - np.exp(-np.arange(0.5, 2.1, 0.5) ** 2 / 2)

    # Compute contour lines
    if fill or line_width:
        xs, ys = _contour_lines(X, Y, Z, levels)

    # Make fills. This is currently not supported
    if fill:
        raise NotImplementedError("Filled contours are not yet implemented.")
        if fill_palette is None:
            if len(levels) <= 6:
                fill_palette = bokeh.palettes.Greys[len(levels) + 3][1:-1]
            elif len(levels) <= 10:
                fill_palette = bokeh.palettes.Viridis[len(levels) + 1]
            else:
                raise RuntimeError(
                    "Can only have maximally 10 levels with filled contours"
                    + " unless user specifies `fill_palette`."
                )
        elif len(fill_palette) != len(levels) + 1:
            raise RuntimeError(
                "`fill_palette` must have 1 more entry" + " than `levels`"
            )

        p.patch(
            xs[-1], ys[-1], color=fill_palette[0], alpha=fill_alpha, line_color=None
        )
        for i in range(1, len(levels)):
            x_p = np.concatenate((xs[-1 - i], xs[-i][::-1]))
            y_p = np.concatenate((ys[-1 - i], ys[-i][::-1]))
            p.patch(x_p, y_p, color=fill_palette[i], alpha=fill_alpha, line_color=None)

        p.background_fill_color = fill_palette[-1]

    # Populate the plot with contour lines
    p.multi_line(xs, ys, line_width=line_width, **line_kwargs)

    if overlay_grid and overlaid:
        p.grid.level = "overlay"
        p.grid.grid_line_alpha = 0.2

    return p


def mpl_cmap_to_color_mapper(cmap):
    """
    Convert a Matplotlib colormap to a bokeh.models.LinearColorMapper
    instance.

    Parameters
    ----------
    cmap : str
        A string giving the name of the color map.

    Returns
    -------
    output : bokeh.models.LinearColorMapper instance
        A linear color_mapper with 25 gradations.

    Notes
    -----
    See https://matplotlib.org/examples/color/colormaps_reference.html
    for available Matplotlib colormaps.
    """
    cm = mpl_get_cmap(cmap)
    palette = [rgb_frac_to_hex(cm(i)[:3]) for i in range(256)]
    return bokeh.models.LinearColorMapper(palette=palette)


def _ecdf_vals(data, staircase=False, complementary=False):
    """Get x, y, values of an ECDF for plotting.
    Parameters
    ----------
    data : ndarray
        One dimensional Numpy array with data.
    staircase : bool, default False
        If True, generate x and y values for staircase ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    complementary : bool
        If True, return values for ECCDF.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    if staircase:
        x, y = cdf_to_staircase(x, y)
        if complementary:
            y = 1 - y
    elif complementary:
        y = 1 - y + 1 / len(y)

    return x, y


@numba.jit(nopython=True)
def _ecdf_arbitrary_points(data, x):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]


def _ecdf_from_samples(df, name, ptiles, x):
    """Compute ECDFs and percentiles from samples."""
    df_ecdf = pd.DataFrame()
    df_ecdf_vals = pd.DataFrame()
    grouped = df.groupby(["chain", "chain_idx"])
    for i, g in grouped:
        df_ecdf_vals[i] = _ecdf_arbitrary_points(g[name].values, x)

    for ptile in ptiles:
        df_ecdf[str(ptile)] = df_ecdf_vals.quantile(
            ptile / 100, axis=1, interpolation="higher"
        )
    df_ecdf["x"] = x

    return df_ecdf


def cdf_to_staircase(x, y):
    """Convert discrete values of CDF to staircase for plotting.

    Parameters
    ----------
    x : array_like, shape (n,)
        x-values for concave corners of CDF
    y : array_like, shape (n,)
        y-values of the concave corners of the CDF

    Returns
    -------
    x_staircase : array_like, shape (2*n, )
        x-values for staircase CDF.
    y_staircase : array_like, shape (2*n, )
        y-values for staircase CDF.
    """
    # Set up output arrays
    x_staircase = np.empty(2 * len(x))
    y_staircase = np.empty(2 * len(x))

    # y-values for steps
    y_staircase[0] = 0
    y_staircase[1::2] = y
    y_staircase[2::2] = y[:-1]

    # x- values for steps
    x_staircase[::2] = x
    x_staircase[1::2] = x

    return x_staircase, y_staircase


def _parse_omit(omit, include_ppc, include_log_lik):
    """Determine which variables to omit in plot."""
    if omit is None:
        omit = []

    if type(omit) == tuple:
        omit = list(omit)

    if type(omit) != list:
        omit = [omit]
    else:
        omit = copy.copy(omit)

    if not include_ppc:
        if not include_log_lik:
            omit += [re.compile("(.*_ppc\[?\d*\]?$)|(log_?lik.*\[?\d*\]?$)")]
        else:
            omit += [re.compile("(.*_ppc\[?\d*\]?$)")]
    elif not include_log_like:
        omit += [re.compile("(log_?lik.*\[?\d*\]?$)")]

    return tuple(omit)


def _parse_parameters(params):
    parameters = []
    labels = []
    for param in params:
        if type(param) in (tuple, list):
            if len(param) != 2:
                raise RuntimeError("f{param} is not a proper parameter specification.")
            parameters.append(param[0])
            labels.append(param[1])
        elif type(param) == str:
            parameters.append(param)
            labels.append(param)
        else:
            raise RuntimeError("f{param} is not a proper parameter specification.")

    return parameters, labels


@numba.jit(nopython=True)
def _y_ecdf(data, x):
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]


@numba.jit(nopython=True)
def _draw_ecdf_bootstrap(L, n, n_bs_reps=100000):
    x = np.arange(L + 1)
    ys = np.empty((n_bs_reps, len(x)))
    for i in range(n_bs_reps):
        draws = np.random.randint(0, L + 1, size=n)
        ys[i, :] = _y_ecdf(draws, x)
    return ys


def _sbc_rank_envelope(L, n, ptile=95, diff=True, bootstrap=False, n_bs_reps=None):
    x = np.arange(L + 1)
    y = st.randint.cdf(x, 0, L + 1)
    std = np.sqrt(y * (1 - y) / n)

    if bootstrap:
        if n_bs_reps is None:
            n_bs_reps = int(max(n, max(L + 1, 100 / (100 - ptile))) * 100)
        ys = _draw_ecdf_bootstrap(L, n, n_bs_reps=n_bs_reps)
        y_low, y_high = np.percentile(ys, [50 - ptile / 2, 50 + ptile / 2], axis=0)
    else:
        y_low = np.concatenate(
            (st.norm.ppf((50 - ptile / 2) / 100, y[:-1], std[:-1]), (1.0,))
        )
        y_high = np.concatenate(
            (st.norm.ppf((50 + ptile / 2) / 100, y[:-1], std[:-1]), (1.0,))
        )

    # Ensure that ends are appropriate
    y_low = np.maximum(0, y_low)
    y_high = np.minimum(1, y_high)

    # Make "staircase" stepped ECDFs
    _, y_low = cdf_to_staircase(x, y_low)
    x_staircase, y_high = cdf_to_staircase(x, y_high)

    if diff:
        _, y = cdf_to_staircase(x, y)
        y_low -= y
        y_high -= y

    return x_staircase, y_low, y_high


def _ecdf_diff(data, L, staircase=False):
    x, y = _ecdf_vals(data)
    y_uniform = (x + 1) / L
    if staircase:
        x, y = cdf_to_staircase(x, y)
        _, y_uniform = cdf_to_staircase(np.arange(len(data)), y_uniform)
    y -= y_uniform

    return x, y


def _ecdf_y(data, complementary=False):
    """Give y-values of an ECDF for an unsorted column in a data frame.

    Parameters
    ----------
    data : Pandas Series
        Series (or column of a DataFrame) from which to generate ECDF
        values
    complementary : bool, default False
        If True, give the ECCDF values.

    Returns
    -------
    output : Pandas Series
        Corresponding y-values for an ECDF when plotted with dots.

    Notes
    -----
    This only works for plotting an ECDF with points, not for staircase
    ECDFs
    """
    if complementary:
        return 1 - data.rank(method="first") / len(data) + 1 / len(data)
    else:
        return data.rank(method="first") / len(data)


def _display_clicks(div, attributes=[], style="float:left;clear:left;font_size=0.5pt"):
    """Build a suitable CustomJS to display the current event
    in the div model."""
    return bokeh.models.CustomJS(
        args=dict(div=div),
        code="""
        var attrs = %s; var args = [];
        for (var i=0; i<attrs.length; i++ ) {
            args.push(Number(cb_obj[attrs[i]]).toFixed(4));
        }
        var line = "<span style=%r>[" + args.join(", ") + "], </span>\\n";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if ( lines.length > 35 ) { lines.shift(); }
        div.text = lines.join("\\n");
    """
        % (attributes, style),
    )


def _data_range(df, x, y, margin=0.02):
    x_range = df[x].max() - df[x].min()
    y_range = df[y].max() - df[y].min()
    return (
        [df[x].min() - x_range * margin, df[x].max() + x_range * margin],
        [df[y].min() - y_range * margin, df[y].max() + y_range * margin],
    )


def _create_points_image(x_range, y_range, w, h, df, x, y, cmap):
    cvs = ds.Canvas(
        x_range=x_range, y_range=y_range, plot_height=int(h), plot_width=int(w)
    )
    agg = cvs.points(df, x, y, agg=ds.reductions.count())
    return ds.transfer_functions.dynspread(
        ds.transfer_functions.shade(agg, cmap=cmap, how="linear")
    )


def _create_line_image(x_range, y_range, w, h, df, x, y, cmap=None):
    cvs = ds.Canvas(
        x_range=x_range, y_range=y_range, plot_height=int(h), plot_width=int(w)
    )
    agg = cvs.line(df, x, y)
    return ds.transfer_functions.dynspread(ds.transfer_functions.shade(agg, cmap=cmap))


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
    c = matplotlib._contour.QuadContourGenerator(X, Y, Z, None, True, 0)
    xs = []
    ys = []
    for level in V:
        paths = c.create_contour(level)
        for line in paths:
            xs.append(line[:, 0])
            ys.append(line[:, 1])

    return xs, ys


def contour_lines_from_samples(
    x, y, smooth=0.02, levels=None, bins=50, weights=None, extend_domain=False
):
    """
    Get lines for a contour plot from (x, y) samples.

    Parameters
    ----------
    x : array_like, shape (n,)
        x-values of samples.
    y : array_like, shape (n,)
        y-values of samples.
    smooth : float, default 0.02
        Smoothing parameter for Gaussian smoothing of contour. A
        Gaussian filter is applied with standard deviation given by
        `smooth * bins`. If None, no smoothing is done.
    levels : float, list of floats, or None
        The levels of the contours. To enclose 95% of the samples, use
        `levels=0.95`. If provided as a list, multiple levels are used.
        If None, `levels` is approximated [0.12, 0.39, 0.68, 0.86].
    bins : int, default 50
        Binning of samples into square bins is necessary to construct
        the contours. `bins` gives the number of bins in each direction.
    weights : array_like, shape (n,), default None
        Weights to apply to each sample in constructing the histogram.
        Default is `None`, such that all samples are equally weighted.
    extend_domain : bool, default False
        If True, extend the domain of the contours beyond the domain
        of the min and max of the samples. This can be useful if the
        contours might clash with the edges of a plot.

    Returns
    -------
    xs : list of arrays
        Each array is the x-values for a plotted contour
    ys : list of arrays
        Each array is the y-values for a plotted contour

    Notes
    -----
    The method proceeds as follows: the samples are binned. The
    counts of samples landing in bins are thought of as values of a
    function f(xb, yb), where (xb, yb) denotes the center of the
    respective bins. This function is then optionally smoothed using
    a Gaussian blur, and then the result is used to construct a
    contour plot.

    Based heavily on code from the corner package by Dan Forman-Mackey.
    """
    # The code in this function is based on the corner package by Dan Forman-Mackey.
    # Following is the copyright notice from that pacakge.
    #
    # Copyright (c) 2013-2016 Daniel Foreman-Mackey
    # All rights reserved.

    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:

    # 1. Redistributions of source code must retain the above copyright notice, this
    #    list of conditions and the following disclaimer.
    # 2. Redistributions in binary form must reproduce the above copyright notice,
    #    this list of conditions and the following disclaimer in the documentation
    #    and/or other materials provided with the distribution.

    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    # The views and conclusions contained in the software and documentation are those
    # of the authors and should not be interpreted as representing official policies,
    # either expressed or implied, of the FreeBSD Project.
    if type(bins) != int or bins <= 0:
        raise ValueError("`bins` must be a positive integer.")

    data_range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    elif type(levels) not in [list, tuple, np.ndarray]:
        levels = [levels]

    for level in levels:
        if level <= 0 or level > 1:
            raise ValueError("All level values must be between zero and one.")

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            range=list(map(np.sort, data_range)),
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "2D histogram generation failed. It could be that one of your sampling ranges has no dynamic range."
        )

    if smooth is not None:
        H = scipy.ndimage.gaussian_filter(H, smooth * bins)

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
        X2 = np.concatenate(
            [
                X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                X1,
                X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
            ]
        )
        Y2 = np.concatenate(
            [
                Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                Y1,
                Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
            ]
        )
        X2, Y2 = np.meshgrid(X2, Y2)
    else:
        X2, Y2 = np.meshgrid(X1, Y1)
        H2 = H

    return _contour_lines(X2, Y2, H2.transpose(), levels)
