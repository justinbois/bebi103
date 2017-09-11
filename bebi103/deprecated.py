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
