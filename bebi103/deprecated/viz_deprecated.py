def bokeh_matplot(df, i_col, j_col, data_col, data_range=None, n_colors=21,
                  label_ticks=True, colormap='RdBu_r', plot_width=1000,
                  plot_height=1000, x_axis_location='auto',
                  toolbar_location='left',
                  tools='reset,resize,hover,save,pan,box_zoom,wheel_zoom',
                  **kwargs):
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
    x_axis_location : str, default = None
        Location of the x-axis around the plot. If 'auto' and first
        element of `df[i_col]` is numerical, x-axis will be placed below
        with the lower left corner as the origin. Otherwise, above
        with the upper left corner as the origin.
    toolbar_location : str, default = 'left'
        Location of the Bokeh toolbar around the plot
    tools : str, default = 'reset,resize,hover,save,pan,box_zoom,wheel_zoom'
        Tools to show in the Bokeh toolbar
    **kwargs
        Arbitrary keyword arguments passed to bokeh.plotting.figure

    Returns
    -------
    Bokeh plotting object


    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
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

    # only reverse the y-axis and put the x-axis on top
    # if the x-axis is categorical:
    if x_axis_location == 'auto':
        if isinstance(df[j_col].iloc[0], numbers.Number):
            y_range = list(df_[i_col].unique())
            x_axis_location = 'below'
        else:
            y_range = list(reversed(list(df_[i_col].unique())))
            x_axis_location = 'above'
    elif x_axis_location == 'above':
        y_range = list(reversed(list(df_[i_col].unique())))
    elif x_axis_location == 'below':
        y_range = list(df_[i_col].unique())

    # Set up figure
    p = bokeh.plotting.figure(x_range=list(df_[j_col].unique()),
                              y_range=y_range,
                              x_axis_location=x_axis_location,
                              plot_width=plot_width,
                              plot_height=plot_height,
                              toolbar_location=toolbar_location,
                              tools=tools, **kwargs)

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
    p.xaxis.major_label_orientation = np.pi / 3

    # Build hover tool
    hover = p.select(dict(type=bokeh.models.HoverTool))
    hover.tooltips = collections.OrderedDict([('i', '  @' + i_col),
                                              ('j', '  @' + j_col),
                                              (data_col, '  @' + data_col)])

    return p


def bokeh_boxplot(df, value, label, ylabel=None, sort=True, plot_width=650,
                  plot_height=450, box_fill_color='medium_purple',
                  background_fill_color='#DFDFE5',
                  tools='reset,resize,hover,save,pan,box_zoom,wheel_zoom',
                  **kwargs):
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
    background_fill_color : str, default = '#DFDFE5'
        Fill color of the plot background
    tools : str, default = 'reset,resize,hover,save,pan,box_zoom,wheel_zoom'
        Tools to show in the Bokeh toolbar
    **kwargs
        Arbitrary keyword arguments passed to bokeh.plotting.figure

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
    upper_cutoff = q3 + 1.5 * iqr
    lower_cutoff = q1 - 1.5 * iqr

    # Find the outliers for each category
    def outliers(group):
        cat = group.name
        outlier_inds = (group[value] > upper_cutoff[cat]) | \
                       (group[value] < lower_cutoff[cat])
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
    p = bokeh.plotting.figure(x_range=cats,
                              background_fill_color=background_fill_color,
                              plot_width=plot_width, plot_height=plot_height,
                              tools=tools,
                              **kwargs)
    p.ygrid.grid_line_color = 'white'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_width = 2
    p.yaxis.axis_label = ylabel

    # stems
    p.segment(cats, upper, cats, q3, line_width=2, line_color="black")
    p.segment(cats, lower, cats, q1, line_width=2, line_color="black")

    # boxes
    p.rect(cats, (q3 + q1) / 2, 0.5, q3 - q1, fill_color="mediumpurple",
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


def _outliers(data):
    bottom, middle, top = np.percentile(data, [25, 50, 75])
    iqr = top - bottom
    outliers = data[(data > top + 1.5 * iqr) | (data < bottom - 1.5 * iqr)]
    return outliers


def _box_and_whisker(data):
    middle = data.median()
    bottom = data.quantile(0.25)
    top = data.quantile(0.75)
    iqr = top - bottom
    top_whisker = data[data <= top + 1.5 * iqr].max()
    bottom_whisker = data[data >= bottom - 1.5 * iqr].min()
    return pd.Series(
        {
            "middle": middle,
            "bottom": bottom,
            "top": top,
            "top_whisker": top_whisker,
            "bottom_whisker": bottom_whisker,
        }
    )


def _box_source(df, cats, val, cols):
    """Construct a data frame for making box plot."""

    # Need to reset index for use in slicing outliers
    df_source = df.reset_index(drop=True)

    if type(cats) in [list, tuple]:
        level = list(range(len(cats)))
    else:
        level = 0

    if cats is None:
        grouped = df_source
    else:
        grouped = df_source.groupby(cats)

    # Data frame for boxes and whiskers
    df_box = grouped[val].apply(_box_and_whisker).unstack().reset_index()
    source_box = _cat_source(
        df_box, cats, ["middle", "bottom", "top", "top_whisker", "bottom_whisker"], None
    )

    # Data frame for outliers
    df_outliers = grouped[val].apply(_outliers).reset_index(level=level)
    df_outliers[cols] = df_source.loc[df_outliers.index, cols]
    source_outliers = _cat_source(df_outliers, cats, cols, None)

    return source_box, source_outliers


def _check_cat_input(df, cats, val, color_column, tooltips, palette, kwargs):
    if df is None:
        raise RuntimeError("`df` argument must be provided.")
    if cats is None:
        raise RuntimeError("`cats` argument must be provided.")
    if val is None:
        raise RuntimeError("`val` argument must be provided.")

    if type(palette) not in [list, tuple]:
        raise RuntimeError("`palette` must be a list or tuple.")

    if val not in df.columns:
        raise RuntimeError(f"{val} is not a column in the inputted data frame")

    cats_array = type(cats) in [list, tuple]

    if cats_array:
        for cat in cats:
            if cat not in df.columns:
                raise RuntimeError(f"{cat} is not a column in the inputted data frame")
    else:
        if cats not in df.columns:
            raise RuntimeError(f"{cats} is not a column in the inputted data frame")

    if color_column is not None and color_column not in df.columns:
        raise RuntimeError(f"{color_column} is not a column in the inputted data frame")

    cols = _cols_to_keep(cats, val, color_column, tooltips)

    for col in cols:
        if col not in df.columns:
            raise RuntimeError(f"{col} is not a column in the inputted data frame")

    bad_kwargs = ["x", "y", "source", "cat", "legend"]
    if kwargs is not None and any([key in kwargs for key in bad_kwargs]):
        raise RuntimeError(", ".join(bad_kwargs) + " are not allowed kwargs.")

    if val == "cat":
        raise RuntimeError("`'cat'` cannot be used as `val`.")

    if val == "__label" or (cats == "__label" or (cats_array and "__label" in cats)):
        raise RuntimeError("'__label' cannot be used for `val` or `cats`.")

    return cols

def _get_cat_range(df, grouped, order, color_column, horizontal):
    if order is None:
        if isinstance(list(grouped.groups.keys())[0], tuple):
            factors = tuple(
                [tuple([str(k) for k in key]) for key in grouped.groups.keys()]
            )
        else:
            factors = tuple([str(key) for key in grouped.groups.keys()])
    else:
        if type(order[0]) in [list, tuple]:
            factors = tuple([tuple([str(k) for k in key]) for key in order])
        else:
            factors = tuple([str(entry) for entry in order])

    if horizontal:
        cat_range = bokeh.models.FactorRange(*(factors[::-1]))
    else:
        cat_range = bokeh.models.FactorRange(*factors)

    if color_column is None:
        color_factors = factors
    else:
        color_factors = tuple(sorted(list(df[color_column].unique().astype(str))))

    return cat_range, factors, color_factors


def _cat_figure(
    df,
    grouped,
    plot_height,
    plot_width,
    x_axis_label,
    y_axis_label,
    title,
    order,
    color_column,
    tooltips,
    horizontal,
    val_axis_type,
):
    fig_kwargs = dict(
        plot_height=plot_height,
        plot_width=plot_width,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        title=title,
        tooltips=tooltips,
    )

    cat_range, factors, color_factors = _get_cat_range(
        df, grouped, order, color_column, horizontal
    )

    if horizontal:
        fig_kwargs["y_range"] = cat_range
        fig_kwargs["x_axis_type"] = val_axis_type
    else:
        fig_kwargs["x_range"] = cat_range
        fig_kwargs["y_axis_type"] = val_axis_type

    return bokeh.plotting.figure(**fig_kwargs), factors, color_factors


def _cat_source(df, cats, cols, color_column):
    if type(cats) in [list, tuple]:
        cat_source = list(zip(*tuple([df[cat].astype(str) for cat in cats])))
        labels = [", ".join(cat) for cat in cat_source]
    else:
        cat_source = list(df[cats].astype(str).values)
        labels = cat_source

    if type(cols) in [list, tuple, pd.core.indexes.base.Index]:
        source_dict = {col: list(df[col].values) for col in cols}
    else:
        source_dict = {cols: list(df[cols].values)}

    source_dict["cat"] = cat_source
    if color_column in [None, "cat"]:
        source_dict["__label"] = labels
    else:
        source_dict["__label"] = list(df[color_column].astype(str).values)
        source_dict[color_column] = list(df[color_column].astype(str).values)

    return bokeh.models.ColumnDataSource(source_dict)


def _tooltip_cols(tooltips):
    if tooltips is None:
        return []
    if type(tooltips) not in [list, tuple]:
        raise RuntimeError("`tooltips` must be a list or tuple of two-tuples.")

    cols = []
    for tip in tooltips:
        if type(tip) not in [list, tuple] or len(tip) != 2:
            raise RuntimeError("Invalid tooltip.")
        if tip[1][0] == "@":
            if tip[1][1] == "{":
                cols.append(tip[1][2 : tip[1].find("}")])
            elif "{" in tip[1]:
                cols.append(tip[1][1 : tip[1].find("{")])
            else:
                cols.append(tip[1][1:])

    return cols


def _cols_to_keep(cats, val, color_column, tooltips):
    cols = _tooltip_cols(tooltips)
    cols += [val]

    if type(cats) in [list, tuple]:
        cols += list(cats)
    else:
        cols += [cats]

    if color_column is not None:
        cols += [color_column]

    return list(set(cols))

def _point_ecdf_source(data, val, cats, cols, complementary, colored):
    """DataFrame for making point-wise ECDF."""
    df = data.copy()

    if complementary:
        col = "__ECCDF"
    else:
        col = "__ECDF"

    if cats is None or colored:
        df[col] = _ecdf_y(df[val], complementary)
    else:
        df[col] = df.groupby(cats)[val].transform(_ecdf_y, complementary)

    cols += [col]

    return _cat_source(df, cats, cols, None)


def _ecdf_collection_dots(
    df, val, cats, cols, complementary, order, palette, show_legend, y, p, **kwargs
):
    _, _, color_factors = _get_cat_range(df, df.groupby(cats), order, None, False)

    source = _point_ecdf_source(df, val, cats, cols, complementary, False)

    if "color" not in kwargs:
        kwargs["color"] = bokeh.transform.factor_cmap(
            "cat", palette=palette, factors=color_factors
        )

    if show_legend:
        kwargs["legend"] = "__label"

    p.circle(source=source, x=val, y=y, **kwargs)

    return p


def _ecdf_collection_staircase(
    df, val, cats, complementary, order, palette, show_legend, p, **kwargs
):
    grouped = df.groupby(cats)

    color_not_in_kwargs = "color" not in kwargs

    if order is None:
        order = list(grouped.groups.keys())
    grouped_iterator = [
        (order_val, grouped.get_group(order_val)) for order_val in order
    ]

    for i, g in enumerate(grouped_iterator):
        if show_legend:
            if type(g[0]) == tuple:
                legend = ", ".join([str(c) for c in g[0]])
            else:
                legend = str(g[0])
        else:
            legend = None

        if color_not_in_kwargs:
            kwargs["color"] = palette[i % len(palette)]

        _ecdf(
            g[1][val],
            staircase=True,
            p=p,
            legend=legend,
            complementary=complementary,
            **kwargs,
        )

    return p


def bokeh_imrgb(im, plot_height=400, plot_width=None,
                tools='pan,box_zoom,wheel_zoom,reset,resize'):
    """
    Make a Bokeh Figure instance displaying an RGB image.
    If the image is already 32 bit, just display it
    """
    # Make 32 bit image
    if len(im.shape) == 2 and im.dtype == np.uint32:
        im_disp = im
    else:
        im_disp = rgb_to_rgba32(im)

    # Get shape
    n, m = im_disp.shape

    # Determine plot height and width
    if plot_height is not None and plot_width is None:
        plot_width = int(m/n * plot_height)
    elif plot_height is None and plot_width is not None:
        plot_height = int(n/m * plot_width)
    elif plot_height is None and plot_width is None:
        plot_heigt = 400
        plot_width = int(m/n * plot_height)

    # Set up figure with appropriate dimensions
    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width,
                              x_range=[0, m], y_range=[0, n], tools=tools)

    # Display the image, setting the origin and heights/widths properly
    p.image_rgba(image=[im_disp], x=0, y=0, dw=m, dh=n)

    return p


def bokeh_im(im, plot_height=400, plot_width=None,
             color_palette=bokeh.palettes.gray(256),
             tools='pan,box_zoom,wheel_zoom,reset,resize'):
    """
    """
    # Get shape
    n, m = im.shape

    # Determine plot height and width
    if plot_height is not None and plot_width is None:
        plot_width = int(m/n * plot_height)
    elif plot_height is None and plot_width is not None:
        plot_height = int(n/m * plot_width)
    elif plot_height is None and plot_width is None:
        plot_heigt = 400
        plot_width = int(m/n * plot_height)

    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width,
                              x_range=[0, m], y_range=[0, n], tools=tools)

    # Set color mapper
    color = bokeh.models.LinearColorMapper(color_palette)

    # Display the image
    p.image(image=[im], x=0, y=0, dw=m, dh=n, color_mapper=color)

    return p


def distribution_plot_app(
    x_min=None,
    x_max=None,
    scipy_dist=None,
    transform=None,
    custom_pdf=None,
    custom_pmf=None,
    custom_cdf=None,
    params=None,
    n=400,
    plot_height=200,
    plot_width=300,
    x_axis_label="x",
    title=None,
):
    """
    Build interactive Bokeh app displaying a univariate
    probability distribution.

    Parameters
    ----------
    x_min : float
        Minimum value that the random variable can take in plots.
    x_max : float
        Maximum value that the random variable can take in plots.
    scipy_dist : scipy.stats distribution
        Distribution to use in plotting.
    transform : function or None (default)
        A function of call signature `transform(*params)` that takes
        a tuple or Numpy array of parameters and returns a tuple of
        the same length with transformed parameters.
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
    title : str, default None
        Title to be displayed above the PDF or PMF plot.

    Returns
    -------
    output : Bokeh app
        An app to visualize the PDF/PMF and CDF. It can be displayed
        with bokeh.io.show(). If it is displayed in a notebook, the
        notebook_url kwarg should be specified.
    """
    if None in [x_min, x_max]:
        raise RuntimeError("`x_min` and `x_max` must be specified.")

    if scipy_dist is None:
        fun_c = custom_cdf
        if (custom_pdf is None and custom_pmf is None) or custom_cdf is None:
            raise RuntimeError(
                "For custom distributions, both PDF/PMF and" + " CDF must be specified."
            )
        if custom_pdf is not None and custom_pmf is not None:
            raise RuntimeError("Can only specify custom PMF or PDF.")
        if custom_pmf is None:
            discrete = False
            fun_p = custom_pdf
        else:
            discrete = True
            fun_p = custom_pmf
    elif custom_pdf is not None or custom_pmf is not None or custom_cdf is not None:
        raise RuntimeError("Can only specify either custom or scipy distribution.")
    else:
        fun_c = scipy_dist.cdf
        if hasattr(scipy_dist, "pmf"):
            discrete = True
            fun_p = scipy_dist.pmf
        else:
            discrete = False
            fun_p = scipy_dist.pdf

    if discrete:
        p_y_axis_label = "PMF"
    else:
        p_y_axis_label = "PDF"

    if params is None:
        raise RuntimeError("`params` must be specified.")

    def _plot_app(doc):
        p_p = bokeh.plotting.figure(
            plot_height=plot_height,
            plot_width=plot_width,
            x_axis_label=x_axis_label,
            y_axis_label=p_y_axis_label,
            title=title,
        )
        p_c = bokeh.plotting.figure(
            plot_height=plot_height,
            plot_width=plot_width,
            x_axis_label=x_axis_label,
            y_axis_label="CDF",
        )

        # Link the axes
        p_c.x_range = p_p.x_range

        # Make sure CDF y_range is zero to one
        p_c.y_range = bokeh.models.Range1d(-0.05, 1.05)

        # Make array of parameter values
        param_vals = np.array([param["value"] for param in params])
        if transform is not None:
            param_vals = transform(*param_vals)

        # Set up data for plot
        if discrete:
            x = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1)
            x_size = x[-1] - x[0]
            x_c = np.empty(2 * len(x))
            x_c[::2] = x
            x_c[1::2] = x
            x_c = np.concatenate(
                (
                    (max(x[0] - 0.05 * x_size, x[0] - 0.95),),
                    x_c,
                    (min(x[-1] + 0.05 * x_size, x[-1] + 0.95),),
                )
            )
            x_cdf = np.concatenate(((x_c[0],), x))
        else:
            x = np.linspace(x_min, x_max, n)
            x_c = x_cdf = x

        # Compute PDF and CDF
        y_p = fun_p(x, *param_vals)
        y_c = fun_c(x_cdf, *param_vals)
        if discrete:
            y_c_plot = np.empty_like(x_c)
            y_c_plot[::2] = y_c
            y_c_plot[1::2] = y_c
            y_c = y_c_plot

        # Set up data sources
        source_p = bokeh.models.ColumnDataSource(data={"x": x, "y_p": y_p})
        source_c = bokeh.models.ColumnDataSource(data={"x": x_c, "y_c": y_c})

        # Plot PDF and CDF
        p_c.line("x", "y_c", source=source_c, line_width=2)
        if discrete:
            p_p.circle("x", "y_p", source=source_p, size=5)
            p_p.segment(x0="x", x1="x", y0=0, y1="y_p", source=source_p, line_width=2)
        else:
            p_p.line("x", "y_p", source=source_p, line_width=2)

        def _callback(attr, old, new):
            param_vals = tuple([slider.value for slider in sliders])
            if transform is not None:
                param_vals = transform(*param_vals)

            # Compute PDF and CDF
            source_p.data["y_p"] = fun_p(x, *param_vals)
            y_c = fun_c(x_cdf, *param_vals)
            if discrete:
                y_c_plot = np.empty_like(x_c)
                y_c_plot[::2] = y_c
                y_c_plot[1::2] = y_c
                y_c = y_c_plot
            source_c.data["y_c"] = y_c

        sliders = [
            bokeh.models.Slider(
                start=param["start"],
                end=param["end"],
                value=param["value"],
                step=param["step"],
                title=param["name"],
            )
            for param in params
        ]
        for slider in sliders:
            slider.on_change("value", _callback)

        # Add the plot to the app
        widgets = bokeh.layouts.widgetbox(sliders)
        grid = bokeh.layouts.gridplot([p_p, p_c], ncols=2)
        doc.add_root(bokeh.layouts.column(widgets, grid))

    handler = bokeh.application.handlers.FunctionHandler(_plot_app)
    return bokeh.application.Application(handler)


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
        raise RuntimeError("Can only adjust range if data is Pandas DataFrame.")

    # Adjust ranges of kdims
    for i, dim in enumerate(element.kdims):
        if element.data[dim.name].dtype in [float, int]:
            data_range = (element.data[dim.name].min(), element.data[dim.name].max())
            if data_range[1] - data_range[0] > 0:
                buff = buffer * (data_range[1] - data_range[0])
                element.kdims[i].soft_range = (
                    data_range[0] - buff,
                    data_range[1] + buff,
                )

    # Adjust ranges of vdims
    for i, dim in enumerate(element.vdims):
        if element.data[dim.name].dtype in [float, int]:
            data_range = (element.data[dim.name].min(), element.data[dim.name].max())
            if data_range[1] - data_range[0] > 0:
                buff = buffer * (data_range[1] - data_range[0])
                element.vdims[i].soft_range = (
                    data_range[0] - buff,
                    data_range[1] + buff,
                )

    return element

