import warnings

import numpy as np
import pandas as pd

import scipy.odr

import numba

import skimage.io
import skimage.measure

import colorcet

import bokeh.models
import bokeh.palettes
import bokeh.plotting

from matplotlib import path

from . import viz
from . import utils

def imshow(
    im,
    cmap=None,
    frame_height=400,
    frame_width=None,
    length_units="pixels",
    interpixel_distance=1.0,
    x_range=None,
    y_range=None,
    colorbar=False,
    no_ticks=False,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    flip=True,
    return_im=False,
    saturate_channels=True,
    min_intensity=None,
    max_intensity=None,
    display_clicks=False,
    record_clicks=False,
):
    """
    Display an image in a Bokeh figure.

    Parameters
    ----------
    im : Numpy array
        If 2D, intensity image to be displayed. If 3D, first two
        dimensions are pixel values. Last dimension can be of length
        1, 2, or 3, which specify colors.
    cmap : str or list of hex colors, default None
        If `im` is an intensity image, `cmap` is a mapping of
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `cmap` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    frame_height : int
        Height of the plot in pixels. The width is scaled so that the
        x and y distance between pixels is the same.
    frame_width : int or None (default)
        If None, the width is scaled so that the x and y distance
        between pixels is approximately the same. Otherwise, the width
        of the plot in pixels.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    x_range : bokeh.models.Range1d instance, default None
        Range of x-axis. If None, determined automatically.
    y_range : bokeh.models.Range1d instance, default None
        Range of y-axis. If None, determined automatically.
    colorbar : bool, default False
        If True, include a colorbar.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    x_axis_label : str, default None
        Label for the x-axis. If None, labeled with `length_units`.
    y_axis_label : str, default None
        Label for the y-axis. If None, labeled with `length_units`.
    title : str, default None
        The title of the plot.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
    return_im : bool, default False
        If True, return the GlyphRenderer instance of the image being
        displayed.
    saturate_channels : bool, default True
        If True, each of the channels have their displayed pixel values
        extended to range from 0 to 255 to show the full dynamic range.
    min_intensity : int or float, default None
        Minimum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    max_intensity : int or float, default None
        Maximum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    display_clicks : bool, default False
        If True, display clicks to the right of the plot using
        JavaScript. The clicks are not recorded nor stored, just
        printed. If you want to store the clicks, use the
        `record_clicks()` or `draw_rois()` functions.
    record_clicks : bool, default False
        Deprecated. Use `display_clicks`.

    Returns
    -------
    p : bokeh.plotting.figure instance
        Bokeh plot with image displayed.
    im : bokeh.models.renderers.GlyphRenderer instance (optional)
        The GlyphRenderer instance of the image being displayed. This is
        only returned if `return_im` is True.
    """
    if record_clicks:
        warnings.warn(
            "`record_clicks` is deprecated. Use the `bebi103.viz.record_clicks()` function to store clicks. Otherwise use the `display_clicks` kwarg to print the clicks to the right of the displayed image.",
            DeprecationWarning,
        )

    # If a single channel in 3D image, flatten and check shape
    if im.ndim == 3:
        if im.shape[2] == 1:
            im = im[:, :, 0]
        elif im.shape[2] not in [2, 3]:
            raise RuntimeError("Can only display 1, 2, or 3 channels.")

    # If binary image, make sure it's int
    if im.dtype == bool:
        im = im.astype(np.uint8)

    # Get color mapper
    if im.ndim == 2:
        if cmap is None:
            color_mapper = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))
        elif type(cmap) == str and cmap.lower() in ["rgb", "cmy"]:
            raise RuntimeError("Cannot use rgb or cmy colormap for intensity image.")
        else:
            color_mapper = bokeh.models.LinearColorMapper(cmap)

        if min_intensity is None:
            color_mapper.low = im.min()
        else:
            color_mapper.low = min_intensity
        if max_intensity is None:
            color_mapper.high = im.max()
        else:
            color_mapper.high = max_intensity
    elif im.ndim == 3:
        if cmap is None or cmap.lower() == "cmy":
            im = im_merge(
                *np.rollaxis(im, 2),
                cmy=True,
                im_0_min=min_intensity,
                im_1_min=min_intensity,
                im_2_min=min_intensity,
                im_0_max=max_intensity,
                im_1_max=max_intensity,
                im_2_max=max_intensity,
            )
        elif cmap.lower() == "rgb":
            im = im_merge(
                *np.rollaxis(im, 2),
                cmy=False,
                im_0_min=min_intensity,
                im_1_min=min_intensity,
                im_2_min=min_intensity,
                im_0_max=max_intensity,
                im_1_max=max_intensity,
                im_2_max=max_intensity,
            )
        else:
            raise RuntimeError("Invalid color mapper for color image.")
    else:
        raise RuntimeError("Input image array must have either 2 or 3 dimensions.")

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
    if frame_width is None:
        frame_width = int(m / n * frame_height)
    if colorbar:
        toolbar_location = "above"
    else:
        toolbar_location = "right"
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=x_range,
        y_range=y_range,
        title=title,
        toolbar_location=toolbar_location,
        tools="pan,box_zoom,wheel_zoom,save,reset",
    )
    if no_ticks:
        p.xaxis.major_label_text_font_size = "0pt"
        p.yaxis.major_label_text_font_size = "0pt"
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
            im = im[::-1, :]
        im_bokeh = p.image(
            image=[im],
            x=x_range[0],
            y=y_range[0],
            dw=dw,
            dh=dh,
            color_mapper=color_mapper,
        )
    else:
        im_bokeh = p.image_rgba(
            image=[rgb_to_rgba32(im, flip=flip)],
            x=x_range[0],
            y=y_range[0],
            dw=dw,
            dh=dh,
        )

    # Make a colorbar
    if colorbar:
        if im.ndim == 3:
            warnings.warn("No colorbar display for RGB images.")
        else:
            color_bar = bokeh.models.ColorBar(
                color_mapper=color_mapper,
                label_standoff=12,
                border_line_color=None,
                location=(0, 0),
            )
            p.add_layout(color_bar, "right")

    if record_clicks or display_clicks:
        div = bokeh.models.Div(width=200)
        layout = bokeh.layouts.row(p, div)
        p.js_on_event(bokeh.events.Tap, viz._display_clicks(div, attributes=["x", "y"]))
        if return_im:
            return layout, im_bokeh
        else:
            return layout

    if return_im:
        return p, im_bokeh
    return p


def record_clicks(
    im,
    notebook_url="localhost:8888",
    point_size=3,
    table_height=200,
    crosshair_alpha=0.5,
    point_color="white",
    cmap=None,
    frame_height=400,
    frame_width=None,
    length_units="pixels",
    interpixel_distance=1.0,
    x_range=None,
    y_range=None,
    colorbar=False,
    no_ticks=False,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    flip=False,
    saturate_channels=True,
    min_intensity=None,
    max_intensity=None,
):
    """Display and record mouse clicks on a Bokeh plot of an image.

    Parameters
    ----------
    im : 2D Numpy array
        Image to display while clicking.
    notebook_url : str, default 'localhost:8888'
        URL of notebook for display.
    point_size : int, default 3
        Size of points to display when clicking.
    table_height : int, default 200
        Height, in pixels, of table displaying mouse click locations.
    crosshair_alpha : float, default 0.5
        Opacity value for crosshairs when using the crosshair tool.
    point_color : str, default 'white'
        Color of the points displaying clicks.
    cmap : str or list of hex colors, default None
        If `im` is an intensity image, `cmap` is a mapping of
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `cmap` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    frame_height : int
        Height of the plot in pixels. The width is scaled so that the
        x and y distance between pixels is the same.
    frame_width : int or None (default)
        If None, the width is scaled so that the x and y distance
        between pixels is approximately the same. Otherwise, the width
        of the plot in pixels.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    x_range : bokeh.models.Range1d instance, default None
        Range of x-axis. If None, determined automatically.
    y_range : bokeh.models.Range1d instance, default None
        Range of y-axis. If None, determined automatically.
    colorbar : bool, default False
        If True, include a colorbar.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    x_axis_label : str, default None
        Label for the x-axis. If None, labeled with `length_units`.
    y_axis_label : str, default None
        Label for the y-axis. If None, labeled with `length_units`.
    title : str, default None
        The title of the plot.
    flip : bool, default False
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
        If you are going to use the clicks you record in further image
        processing applicaitons, you should have `flip` set to False.
    saturate_channels : bool, default True
        If True, each of the channels have their displayed pixel values
        extended to range from 0 to 255 to show the full dynamic range.
    min_intensity : int or float, default None
        Minimum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    max_intensity : int or float, default None
        Maximum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.

    Returns
    -------
    output : Bokeh ColumnDataSource
        A Bokeh ColumnDataSource instance. This can be immediately
        converted to a Pandas DataFrame using the `to_df()` method. For
        example, `output.to_df()`.
    """
    points_source = bokeh.models.ColumnDataSource({"x": [], "y": []})

    def modify_doc(doc):
        p = imshow(
            im,
            cmap=cmap,
            frame_height=frame_height,
            frame_width=frame_width,
            length_units=length_units,
            interpixel_distance=interpixel_distance,
            x_range=x_range,
            y_range=y_range,
            colorbar=colorbar,
            no_ticks=no_ticks,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            title=title,
            flip=flip,
            return_im=False,
            saturate_channels=saturate_channels,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
        )

        view = bokeh.models.CDSView(source=points_source)

        renderer = p.scatter(
            x="x",
            y="y",
            source=points_source,
            view=view,
            color=point_color,
            size=point_size,
        )

        columns = [
            bokeh.models.TableColumn(field="x", title="x"),
            bokeh.models.TableColumn(field="y", title="y"),
        ]

        table = bokeh.models.DataTable(
            source=points_source, columns=columns, editable=True, height=table_height
        )

        draw_tool = bokeh.models.PointDrawTool(renderers=[renderer])
        p.add_tools(draw_tool)
        p.add_tools(bokeh.models.CrosshairTool(line_alpha=crosshair_alpha))
        p.toolbar.active_tap = draw_tool

        doc.add_root(bokeh.layouts.column(p, table))

    bokeh.io.show(modify_doc, notebook_url=notebook_url)

    return points_source


def draw_rois(
    im,
    notebook_url="localhost:8888",
    table_height=100,
    crosshair_tool_alpha=0.5,
    color="white",
    fill_alpha=0.1,
    vertex_color="red",
    vertex_size=10,
    cmap=None,
    frame_height=400,
    frame_width=None,
    length_units="pixels",
    interpixel_distance=1.0,
    x_range=None,
    y_range=None,
    colorbar=False,
    no_ticks=False,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    flip=False,
    saturate_channels=True,
    min_intensity=None,
    max_intensity=None,
):
    """Draw and record polygonal regions of interest on a plot of a
    Bokeh image.

    Parameters
    ----------
    im : 2D Numpy array
        Image to display while clicking.
    notebook_url : str, default 'localhost:8888'
        URL of notebook for display.
    table_height : int, default 200
        Height, in pixels, of table displaying polygon vertex locations.
    crosshair_alpha : float, default 0.5
        Opacity value for crosshairs when using the crosshair tool.
    color : str, default 'white'
        Color of the ROI polygons (lines and fill).
    fill_alpha : float, default 0.1
        Opacity of drawn ROI polygons.
    vertex_color : str, default 'red'
        Color of vertices of the ROI polygons while using the polygon
        edit tool.
    vertex_size: int, default 10
        Size, in pixels, of vertices of the ROI polygons while using the
        polygon edit tool.
    cmap : str or list of hex colors, default None
        If `im` is an intensity image, `cmap` is a mapping of
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `cmap` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    frame_height : int
        Height of the plot in pixels. The width is scaled so that the
        x and y distance between pixels is the same.
    frame_width : int or None (default)
        If None, the width is scaled so that the x and y distance
        between pixels is approximately the same. Otherwise, the width
        of the plot in pixels.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    x_range : bokeh.models.Range1d instance, default None
        Range of x-axis. If None, determined automatically.
    y_range : bokeh.models.Range1d instance, default None
        Range of y-axis. If None, determined automatically.
    colorbar : bool, default False
        If True, include a colorbar.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    x_axis_label : str, default None
        Label for the x-axis. If None, labeled with `length_units`.
    y_axis_label : str, default None
        Label for the y-axis. If None, labeled with `length_units`.
    title : str, default None
        The title of the plot.
    flip : bool, default False
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
        If you are going to use the clicks you record in further image
        processing applicaitons, you should have `flip` set to False.
    saturate_channels : bool, default True
        If True, each of the channels have their displayed pixel values
        extended to range from 0 to 255 to show the full dynamic range.
    min_intensity : int or float, default None
        Minimum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    max_intensity : int or float, default None
        Maximum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.

    Returns
    -------
    output : Bokeh ColumnDataSource
        A Bokeh ColumnDataSource instance. This can be immediately
        converted to a Pandas DataFrame `roicds_to_df()` function. For
        example, `bebi103.viz.roicds_to_df(output)`.

    Notes
    -----
    The displayed table is not particularly useful because it
    displays a list of points. It helps to make sure your clicks are
    getting registered and to select which ROI number is which
    polygon.
    """

    poly_source = bokeh.models.ColumnDataSource({"xs": [], "ys": []})

    def modify_doc(doc):
        p = imshow(
            im,
            cmap=cmap,
            frame_height=frame_height,
            frame_width=frame_width,
            length_units=length_units,
            interpixel_distance=interpixel_distance,
            x_range=x_range,
            y_range=y_range,
            colorbar=colorbar,
            no_ticks=no_ticks,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            title=title,
            flip=flip,
            return_im=False,
            saturate_channels=saturate_channels,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
        )

        view = bokeh.models.CDSView(source=poly_source)
        renderer = p.patches(
            xs="xs",
            ys="ys",
            source=poly_source,
            view=view,
            fill_alpha=fill_alpha,
            color=color,
        )
        vertex_renderer = p.circle([], [], size=vertex_size, color="red")

        columns = [
            bokeh.models.TableColumn(field="xs", title="xs"),
            bokeh.models.TableColumn(field="ys", title="ys"),
        ]

        table = bokeh.models.DataTable(
            source=poly_source, index_header="roi", columns=columns, height=table_height
        )
        draw_tool = bokeh.models.PolyDrawTool(renderers=[renderer])
        edit_tool = bokeh.models.PolyEditTool(
            renderers=[renderer], vertex_renderer=vertex_renderer
        )
        p.add_tools(draw_tool)
        p.add_tools(edit_tool)
        p.add_tools(bokeh.models.CrosshairTool(line_alpha=crosshair_tool_alpha))
        p.toolbar.active_tap = draw_tool

        doc.add_root(bokeh.layouts.column(p, table))

    bokeh.io.show(modify_doc, notebook_url=notebook_url)

    return poly_source


def roicds_to_df(cds):
    """Convert a ColumnDataSource outputted by `draw_rois()` to a Pandas
    DataFrame.

    Parameters
    ----------
    cds : Bokeh ColumnDataSource
        ColumnDataSource outputted by `draw_rois()`

    Returns
    -------
    output : Pandas DataFrame
        DataFrame with columns ['roi', 'x', 'y'] containing the
        positions of the vertices of the respective polygonal ROIs.
    """
    roi = np.concatenate([[i] * len(x_data) for i, x_data in enumerate(cds.data["xs"])])
    x = np.concatenate(cds.data["xs"])
    y = np.concatenate(cds.data["ys"])

    return pd.DataFrame(data=dict(roi=roi, x=x, y=y))


def im_merge(
    im_0,
    im_1,
    im_2=None,
    im_0_max=None,
    im_1_max=None,
    im_2_max=None,
    im_0_min=None,
    im_1_min=None,
    im_2_min=None,
    cmy=True,
):
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
    if (
        im_0_max < im_0.max()
        or im_1_max < im_1.max()
        or (im_2 is not None and im_2_max < im_2.max())
    ):
        raise RuntimeError("Inputted max of channel < max of inputted channel.")

    # Make sure mins are ok
    if (
        im_0_min > im_0.min()
        or im_1_min > im_1.min()
        or (im_2 is not None and im_2_min > im_2.min())
    ):
        raise RuntimeError("Inputted min of channel > min of inputted channel.")

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
            im_rgb[:, :, i] /= im_rgb[:, :, i].max()
    else:
        im_rgb = np.empty((*im_0.shape, 3))
        im_rgb[:, :, 0] = im_0
        im_rgb[:, :, 1] = im_1
        im_rgb[:, :, 2] = im_2

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
    if im.ndim != 3 or im.shape[2] != 3:
        raise RuntimeError("Input image is not RGB.")

    # Make sure all entries between zero and one
    if (im < 0).any() or (im > 1).any():
        raise RuntimeError("All pixel values must be between 0 and 1.")

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im_8 = skimage.img_as_ubyte(im)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.stack(
        (*np.rollaxis(im_8, 2), 255 * np.ones((n, m), dtype=np.uint8)), axis=2
    )

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
        raise RuntimeError("`rgb_frac` must have exactly three entries.")

    if (np.array(rgb_frac) < 0).any() or (np.array(rgb_frac) > 1).any():
        raise RuntimeError("RGB values must be between 0 and 1.")

    return "#{0:02x}{1:02x}{2:02x}".format(
        int(rgb_frac[0] * 255), int(rgb_frac[1] * 255), int(rgb_frac[2] * 255)
    )


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
    Any keyword arguments except those listed above are passed into
    load_func as kwargs.

    This is a much simplified (and therefore faster) version of
    skimage.io.ImageCollection.
    """

    def __init__(
        self,
        load_pattern,
        load_func=skimage.io.imread,
        conserve_memory=True,
        **load_func_kwargs,
    ):
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


def simple_image_collection(
    im_glob, load_func=skimage.io.imread, conserve_memory=True, **load_func_kwargs
):
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
    Any keyword arguments except those listed above are passed into
    load_func as kwargs.

    This is a much simplified (and therefore faster) version of
    skimage.io.ImageCollection.
    """
    return SimpleImageCollection(
        im_glob,
        load_func=load_func,
        conserve_memory=conserve_memory,
        **load_func_kwargs,
    )


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
    roi_bbox = np.s_[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]

    # Get ROI mask for just within bounding box
    roi_box = roi[roi_bbox]

    # Return boolean in same shape as image
    return (roi, roi_bbox, roi_box)


class _CostesColocalization(object):
    """
    Generic class just to store attributes.
    """

    def __init__(self, **kw):
        self.__dict__ = kw


def costes_coloc(
    im_1,
    im_2,
    psf_width=3,
    n_scramble=1000,
    thresh_r=0.0,
    roi=None,
    roi_method="all",
    do_manders=True,
):
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

    # Make float mirrored boundaries in preparation for scrambling
    im_1_mirror = _mirror_edges(im_1, psf_width).astype(float)
    im_2_mirror = _mirror_edges(im_2, psf_width).astype(float)

    # Set up ROI
    if roi is None:
        roi = np.ones_like(im_1, dtype="bool")

    # Rename images to be sliced ROI and convert to float
    im_1 = im_1[roi].astype(float)
    im_2 = im_2[roi].astype(float)

    # Mirror ROI at edges
    roi_mirror = _mirror_edges(roi, psf_width)

    # Compute the blocks that we'll scramble
    blocks_1 = _im_to_blocks(im_1_mirror, psf_width, roi_mirror, roi_method)
    blocks_2 = _im_to_blocks(im_2_mirror, psf_width, roi_mirror, roi_method)

    # Compute the Pearson coefficient
    pearson_r = utils._pearson_r(blocks_1.ravel(), blocks_2.ravel())

    # Do image scrambling and r calculations
    r_scr = _scrambled_r(blocks_1, blocks_2, n=n_scramble)

    # Compute percent chance of coloc
    p_coloc = (r_scr < pearson_r).sum() / n_scramble

    # Now do work to compute adjusted Manders's coefficients
    if do_manders:
        # Get the linear relationship between im_2 and im_1
        a, b = _odr_linear(im_1.ravel(), im_2.ravel())

        # Perform threshold calculation
        thresh_1 = _find_thresh(im_1, im_2, a, b, thresh_r=thresh_r)
        thresh_2 = a * thresh_1 + b

        # Compute Costes's update to the Manders's coefficients
        inds = (im_1 > thresh_1) & (im_2 > thresh_2)
        M_1 = im_1[inds].sum() / im_1.sum()
        M_2 = im_2[inds].sum() / im_2.sum()

        # Toss results into class for returning
        return _CostesColocalization(
            im_1=im_1,
            im_2=im_2,
            roi=roi,
            roi_method=roi_method,
            psf_width=psf_width,
            n_scramble=n_scramble,
            thresh_r=thresh_r,
            thresh_1=thresh_1,
            thresh_2=thresh_2,
            a=a,
            b=b,
            M_1=M_1,
            M_2=M_2,
            r_scr=r_scr,
            pearson_r=pearson_r,
            p_coloc=p_coloc,
        )
    else:
        return _CostesColocalization(
            im_1=im_1,
            im_2=im_2,
            roi=roi,
            roi_method=roi_method,
            psf_width=psf_width,
            n_scramble=n_scramble,
            thresh_r=None,
            thresh_1=None,
            thresh_2=None,
            a=None,
            b=None,
            M_1=None,
            M_2=None,
            r_scr=r_scr,
            pearson_r=pearson_r,
            p_coloc=p_coloc,
        )


@numba.jit(nopython=True)
def _scrambled_r(blocks_1, blocks_2, n=200):
    """
    Scrambles blocks_1 n_scramble times and returns the Pearson r values.

    Parameters
    ----------
    blocks_1 : n x n_p x n_p Numpy array
        First index corresponds to block ID, and second and third
        indices have pixel values in blocks.
    blocks_2 : n x n_p x n_p Numpy array
        First index corresponds to block ID, and second and third
        indices have pixel values in blocks.
    n : int
        Number of scrambled Pearson r values to compute.

    Returns
    -------
    output : ndarray
        Numpy array with Pearson r values computed by scrambling the
        first set of blocks.
    """
    # Indicies of blocks
    block_inds = np.arange(blocks_1.shape[0])

    # Flatten blocks 2
    blocks_2_flat = blocks_2.flatten()

    r_scr = np.empty(n)
    for i in range(n):
        np.random.shuffle(block_inds)
        r = utils._pearson_r(blocks_1[block_inds].ravel(), blocks_2_flat)
        r_scr[i] = r
    return r_scr


def _odr_linear(x, y, intercept=None, beta0=None):
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
        raise scipy.odr.odr_error("ORD failed.")

    return result.beta


def _find_thresh(im_1, im_2, a, b, thresh_r=0.0):
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
    To determine which pixels are colocalized in two images, we
    do the following:
        1. Perform a regression based on all points of to give I_2 = a * I_1 + b.
        2. Define T = I_1.max().
        3. Compute the Pearson r value considering all pixels with I_1 < T and I_2 < a * T + b.
        4. If r <= thresh_r decrement T and goto 3.  Otherwise, save $T_1 = T$ and $T_2 = a * T + b.
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
    r = _pearsonr_below_thresh(thresh, im_1, im_2, a, b)
    min_r = r
    min_thresh = thresh
    while thresh > thresh_min and r > thresh_r:
        thresh -= incr
        r = _pearsonr_below_thresh(thresh, im_1, im_2, a, b)
        if min_r > r:
            min_r = r
            min_thresh = thresh

    if thresh == thresh_min:
        thresh = min_thresh

    return thresh


def _pearsonr_below_thresh(thresh, im_1, im_2, a, b):
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
    r = utils._pearson_r(im_1[inds], im_2[inds])
    return r


def _mirror_edges(im, psf_width):
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
    return np.pad(im, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")


def _im_to_blocks(im, width, roi=None, roi_method="all"):
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
    if roi_method == "all":
        roi_test = np.all
    else:
        roi_test = np.any

    # Construct list of blocks
    return np.array(
        [
            im[i : i + width, j : j + width]
            for i in range(0, im.shape[0], width)
            for j in range(0, im.shape[1], width)
            if roi_test(roi[i : i + width, j : j + width])
        ]
    )
