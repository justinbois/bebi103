import bokeh.palettes
import colorcet
import holoviews as hv

datashader_blues = bokeh.palettes.Blues9[:6][::-1]
datashader_greys = bokeh.palettes.Greys9[:6][::-1]
datashader_purples = bokeh.palettes.Purples9[:6][::-1]
datashader_reds = bokeh.palettes.Reds9[:6][::-1]


def no_xgrid_hook(plot, element):
    """Hook for disabling x-grid lines.

    This function is never called, but used as one of the hooks passed
    as the hooks kwarg, e.g. as
    `hv.opts.BoxWhisker(hooks=[no_xgrid_hook])`.

    """
    plot.handles["plot"].xgrid.grid_line_color = None


def no_ygrid_hook(plot, element):
    """Hook for disabling x-grid lines.

    This function is never called, but used as one of the hooks passed
    as the hooks kwarg, e.g. as
    `hv.opts.BoxWhisker(hooks=[no_ygrid_hook])`.

    """
    plot.handles["plot"].ygrid.grid_line_color = None


default_categorical_cmap = colorcet.b_glasbey_category10
default_sequential_cmap = list(bokeh.palettes.Viridis256)
default_diverging_cmap = colorcet.b_diverging_bwr_20_95_c54


def set_defaults():
    """
    Set convenient HoloViews defaults

    Called without arguments, sets default visual plotting options for
    HoloViews.
    """
    hv.opts.defaults(
        hv.opts.Bars(
            alpha=0.9,
            bar_width=0.6,
            cmap=default_categorical_cmap,
            color=default_categorical_cmap[0],
            frame_height=300,
            frame_width=400,
            hooks=[no_xgrid_hook],
            legend_offset=(10, 100),
            legend_position="right",
            line_alpha=0,
            padding=0.05,
            show_grid=True,
            show_legend=False,
            ylim=(0, None),
        )
    )

    hv.opts.defaults(
        hv.opts.BoxWhisker(
            box_cmap=default_categorical_cmap,
            box_fill_alpha=0.75,
            box_fill_color="lightgray",
            box_line_color="#222222",
            box_width=0.4,
            cmap=default_categorical_cmap,
            frame_height=300,
            frame_width=400,
            hooks=[no_xgrid_hook],
            legend_offset=(10, 100),
            legend_position="right",
            outlier_alpha=0.75,
            outlier_line_color=None,
            padding=0.05,
            show_grid=True,
            show_legend=False,
            toolbar="above",
            whisker_color="#222222",
            whisker_line_width=1,
        )
    )

    hv.opts.defaults(
        hv.opts.Curve(
            color=hv.Cycle(default_categorical_cmap),
            frame_height=300,
            frame_width=400,
            line_width=2,
            muted_line_alpha=0.1,
            padding=0.05,
            show_grid=True,
            toolbar="above",
        )
    )

    hv.opts.defaults(hv.opts.HeatMap(cmap=default_sequential_cmap))

    hv.opts.defaults(
        hv.opts.HexTiles(
            cmap=default_sequential_cmap, padding=0.05, show_grid=True, toolbar="above"
        )
    )

    hv.opts.defaults(
        hv.opts.Histogram(
            fill_alpha=0.3,
            fill_color=hv.Cycle(default_categorical_cmap),
            frame_height=300,
            frame_width=450,
            line_alpha=0,
            line_width=2,
            padding=0.05,
            show_grid=True,
            show_legend=True,
            toolbar="above",
            ylim=(0, None),
        )
    )

    hv.opts.defaults(hv.opts.Image(cmap=default_sequential_cmap))

    hv.opts.defaults(
        hv.opts.NdOverlay(
            click_policy="hide",
            fontsize=dict(legend=8, title=12),
            legend_offset=(10, 100),
            legend_position="right",
            padding=0.05,
            show_grid=True,
            show_legend=True,
            toolbar="above",
        )
    )

    hv.opts.defaults(
        hv.opts.Overlay(
            click_policy="hide",
            fontsize=dict(legend=8),
            legend_offset=(10, 100),
            legend_position="right",
            padding=0.05,
            show_grid=True,
            show_legend=True,
            toolbar="above",
        )
    )

    hv.opts.defaults(
        hv.opts.Path(
            color=hv.Cycle(default_categorical_cmap),
            frame_height=300,
            frame_width=300,
            line_width=2,
            muted_line_alpha=0.1,
            padding=0.05,
            show_grid=True,
            toolbar="above",
        )
    )

    hv.opts.defaults(
        hv.opts.Points(
            alpha=0.75,
            cmap=default_categorical_cmap,
            color=hv.Cycle(default_categorical_cmap),
            fontsize=dict(legend=8),
            frame_height=300,
            frame_width=300,
            legend_offset=(10, 100),
            legend_position="right",
            padding=0.05,
            show_grid=True,
            show_legend=True,
            size=3,
            toolbar="above",
        )
    )

    hv.opts.defaults(
        hv.opts.Scatter(
            alpha=0.75,
            cmap=default_categorical_cmap,
            color=hv.Cycle(default_categorical_cmap),
            fontsize=dict(legend=8),
            frame_height=300,
            frame_width=400,
            legend_offset=(10, 100),
            legend_position="right",
            muted_line_alpha=0.1,
            padding=0.05,
            show_grid=True,
            size=3,
            toolbar="above",
        )
    )

    hv.opts.defaults(
        hv.opts.Spikes(
            cmap=default_categorical_cmap,
            color=hv.Cycle(default_categorical_cmap),
            fontsize=dict(legend=8),
            frame_height=300,
            frame_width=400,
            hooks=[no_xgrid_hook],
            line_width=2,
            muted_line_alpha=0.1,
            padding=0.05,
            show_grid=True,
            toolbar="above",
        )
    )
