"""Test suite for bebi103.viz.

Plotting functions are tested by inspecting the returned Bokeh objects
(figure/layout types, glyph renderers, data sources) and by exercising
error paths. Numerical helpers (ECDF values, histogram binning, contour
lines, color mapping) are tested for correctness directly.

Not covered: functions requiring interactive display or notebook
integration (_display_clicks), and the deprecated/unimplemented paths
beyond checking that they raise (datashade, filled contours).
"""

import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import bokeh.layouts
import bokeh.models
import bokeh.plotting

import bebi103.stan
import bebi103.viz as viz


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_CHAINS = 4
N_DRAWS = 100


def make_samples_dict(n_chains=N_CHAINS, n_draws=N_DRAWS, seed=0, divergences=False):
    """A validated samples dict with expectands and HMC diagnostics."""
    rng = np.random.default_rng(seed)
    expectands = {
        "mu": rng.normal(size=(n_chains, n_draws)),
        "sigma": np.abs(rng.normal(size=(n_chains, n_draws))) + 1.0,
        "alpha[1]": rng.normal(2.0, 1.0, size=(n_chains, n_draws)),
        "alpha[2]": rng.normal(-2.0, 1.0, size=(n_chains, n_draws)),
        "y_ppc[1]": rng.normal(size=(n_chains, n_draws)),
        "log_lik[1]": rng.normal(size=(n_chains, n_draws)),
    }
    divergent = np.zeros((n_chains, n_draws))
    if divergences:
        divergent[0, :5] = 1.0
    diagnostics = {
        "lp__": rng.normal(size=(n_chains, n_draws)),
        "accept_stat__": np.full((n_chains, n_draws), 0.9),
        "stepsize__": np.full((n_chains, n_draws), 0.1),
        "treedepth__": np.full((n_chains, n_draws), 3.0),
        "n_leapfrog__": np.full((n_chains, n_draws), 7.0),
        "energy__": rng.normal(size=(n_chains, n_draws)),
        "divergent__": divergent,
    }
    return {**expectands, **diagnostics}


@pytest.fixture
def samples_dict():
    return make_samples_dict()


@pytest.fixture
def samples_dict_div():
    return make_samples_dict(divergences=True, seed=1)


@pytest.fixture(
    params=["dict", "pandas", "polars", "plain_pandas", "plain_polars"]
)
def samples_any_format(request, samples_dict):
    """Samples in every input format accepted by corner/parcoord/trace."""
    if request.param == "dict":
        return samples_dict
    df = bebi103.stan.convert_samples(samples_dict, "pandas")
    if request.param == "pandas":
        return df
    if request.param == "polars":
        return pl.from_pandas(df)
    plain = df[["mu", "sigma", "alpha[1]", "alpha[2]"]].reset_index(drop=True)
    if request.param == "plain_pandas":
        return plain
    return pl.from_pandas(plain)


def glyphs(p, glyph_type):
    """All glyphs of a given type on a figure."""
    return [
        r.glyph
        for r in p.renderers
        if isinstance(r, bokeh.models.GlyphRenderer) and isinstance(r.glyph, glyph_type)
    ]


def collect_figures(obj):
    """Recursively collect all figures in a Bokeh layout."""
    figs = []
    if isinstance(obj, bokeh.plotting.figure):
        figs.append(obj)
    for child in getattr(obj, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        figs.extend(collect_figures(item))
    return figs


# ---------------------------------------------------------------------------
# to_ColumnDataSource
# ---------------------------------------------------------------------------


class TestToColumnDataSource:
    def test_pandas(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        cds = viz.to_ColumnDataSource(df)
        assert isinstance(cds, bokeh.models.ColumnDataSource)
        assert list(cds.data["a"]) == [1, 2]

    def test_polars(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        cds = viz.to_ColumnDataSource(df)
        assert isinstance(cds, bokeh.models.ColumnDataSource)
        assert list(cds.data["a"]) == [1, 2]

    def test_dict(self):
        cds = viz.to_ColumnDataSource({"a": [1, 2]})
        assert isinstance(cds, bokeh.models.ColumnDataSource)

    def test_invalid(self):
        with pytest.raises(RuntimeError):
            viz.to_ColumnDataSource(42)


# ---------------------------------------------------------------------------
# confints
# ---------------------------------------------------------------------------


class TestConfints:
    summaries = [
        dict(estimate=1.0, conf_int=(0.5, 1.5), label="a"),
        dict(estimate=2.0, conf_int=(1.5, 2.5), label="b"),
        dict(estimate=3.0, conf_int=(2.5, 3.5), label="c"),
    ]

    def test_basic(self):
        p = viz.confints(self.summaries)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Scatter)) == 3
        assert len(glyphs(p, bokeh.models.Line)) == 3

    def test_y_range_labels_reversed(self):
        p = viz.confints(self.summaries)
        assert list(p.y_range.factors) == ["c", "b", "a"]

    def test_string_palette(self):
        p = viz.confints(self.summaries, palette="black")
        assert isinstance(p, bokeh.plotting.figure)

    def test_list_palette(self):
        p = viz.confints(self.summaries, palette=["#ff0000", "#00ff00"])
        assert isinstance(p, bokeh.plotting.figure)

    def test_palette_color_clash(self):
        with pytest.raises(RuntimeError):
            viz.confints(
                self.summaries, palette="black", marker_kwargs=dict(color="red")
            )

    def test_existing_figure(self):
        p0 = bokeh.plotting.figure(y_range=["c", "b", "a"])
        p = viz.confints(self.summaries, p=p0)
        assert p is p0


# ---------------------------------------------------------------------------
# fill_between
# ---------------------------------------------------------------------------


class TestFillBetween:
    x = np.linspace(0.0, 1.0, 10)

    def test_basic(self):
        p = viz.fill_between(self.x, self.x**2, self.x, self.x)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Patch)) == 1
        assert len(glyphs(p, bokeh.models.Line)) == 2

    def test_no_line(self):
        p = viz.fill_between(self.x, self.x**2, self.x, self.x, show_line=False)
        assert len(glyphs(p, bokeh.models.Patch)) == 1
        assert len(glyphs(p, bokeh.models.Line)) == 0

    def test_existing_figure(self):
        p0 = bokeh.plotting.figure()
        p = viz.fill_between(self.x, self.x**2, self.x, self.x, p=p0)
        assert p is p0

    def test_patch_vertices(self):
        p = viz.fill_between(self.x, self.x**2, self.x, self.x, show_line=False)
        renderer = [
            r for r in p.renderers if isinstance(r.glyph, bokeh.models.Patch)
        ][0]
        assert len(renderer.data_source.data["x"]) == 2 * len(self.x)


# ---------------------------------------------------------------------------
# qqplot
# ---------------------------------------------------------------------------


class TestQqplot:
    rng = np.random.default_rng(8)
    samples = rng.normal(size=(150, 30))
    data = rng.normal(size=30)

    def test_basic(self):
        p = viz.qqplot(self.samples, self.data)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Patch)) == 1
        # Two envelope edge lines plus the diagonal reference line.
        assert len(glyphs(p, bokeh.models.Line)) == 3

    def test_samples_must_be_numpy(self):
        with pytest.raises(RuntimeError):
            viz.qqplot(list(self.samples), self.data)

    def test_samples_must_be_2d(self):
        with pytest.raises(RuntimeError):
            viz.qqplot(self.samples.flatten(), self.data)

    def test_shape_mismatch(self):
        with pytest.raises(RuntimeError):
            viz.qqplot(self.samples, self.data[:-1])

    def test_few_samples_warns(self):
        with pytest.warns(UserWarning, match="very few samples"):
            viz.qqplot(self.samples[:20], self.data)


# ---------------------------------------------------------------------------
# ECDF and histogram computation helpers
# ---------------------------------------------------------------------------


class TestEcdfHelpers:
    def test_ecdf_vals_dots(self):
        x, y = viz._ecdf_vals(np.array([3.0, 1.0, 2.0]))
        assert np.allclose(x, [1.0, 2.0, 3.0])
        assert np.allclose(y, [1 / 3, 2 / 3, 1.0])

    def test_ecdf_vals_complementary_dots(self):
        _, y = viz._ecdf_vals(np.array([3.0, 1.0, 2.0]), complementary=True)
        assert np.allclose(y, [1.0, 2 / 3, 1 / 3])

    def test_ecdf_vals_staircase(self):
        x, y = viz._ecdf_vals(np.array([3.0, 1.0, 2.0]), staircase=True)
        assert np.allclose(x, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        assert np.allclose(y, [0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0])

    def test_cdf_to_staircase(self):
        x_s, y_s = viz.cdf_to_staircase(
            np.array([1.0, 2.0, 3.0]), np.array([1 / 3, 2 / 3, 1.0])
        )
        assert np.allclose(x_s, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        assert np.allclose(y_s, [0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0])

    def test_ecdf_arbitrary_points(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.0, 1.5, 2.0, 5.0])
        y = viz._ecdf_arbitrary_points(data, x)
        assert np.allclose(y, [0.0, 0.25, 0.5, 1.0])


class TestBinsToNp:
    def test_int_passthrough(self):
        assert viz._bins_to_np(np.array([1.0, 2.0, 3.0]), 10) == 10

    def test_array_passthrough(self):
        bins = np.array([0.0, 1.0, 2.0])
        assert viz._bins_to_np(np.array([0.5, 1.5]), bins) is bins

    def test_exact(self):
        bins = viz._bins_to_np(np.array([1.0, 2.0, 2.0, 4.0]), "exact")
        assert np.allclose(bins, [0.5, 1.5, 3.0, 5.0])

    def test_exact_single_value(self):
        bins = viz._bins_to_np(np.array([2.0, 2.0]), "exact")
        assert np.allclose(bins, [1.5, 2.5])

    def test_integer(self):
        bins = viz._bins_to_np(np.array([1.0, 2.0, 3.0]), "integer")
        assert np.allclose(bins, np.arange(0, 4) + 0.5)

    def test_integer_noninteger_data(self):
        with pytest.raises(RuntimeError):
            viz._bins_to_np(np.array([1.5, 2.0]), "integer")

    def test_sqrt(self):
        assert viz._bins_to_np(np.arange(100, dtype=float), "sqrt") == 10

    def test_freedman_diaconis(self):
        rng = np.random.default_rng(0)
        bins = viz._bins_to_np(rng.normal(size=1000), "freedman-diaconis")
        assert isinstance(bins, int) and bins > 0

    def test_freedman_diaconis_no_iqr(self):
        assert viz._bins_to_np(np.ones(100), "freedman-diaconis") == 3

    def test_invalid_string(self):
        with pytest.raises(RuntimeError):
            viz._bins_to_np(np.array([1.0]), "not-a-rule")


class TestComputeHistogram:
    def test_counts_match_numpy(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=500)
        e0, f0 = viz._compute_histogram(data, 20, density=False)
        f, e = np.histogram(data, bins=20)
        assert len(e0) == 2 * len(e)
        assert f0[0] == 0 and f0[-1] == 0
        assert np.allclose(f0[1:-1:2], f)
        assert np.allclose(e0[::2], e)

    def test_density_integrates_to_one(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=500)
        e0, f0 = viz._compute_histogram(data, 20, density=True)
        widths = np.diff(e0[::2])
        assert np.isclose(np.sum(widths * f0[1:-1:2]), 1.0)


# ---------------------------------------------------------------------------
# predictive_ecdf
# ---------------------------------------------------------------------------


class TestPredictiveEcdf:
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(150, 40))
    data = rng.normal(size=40)

    def test_basic(self):
        p = viz.predictive_ecdf(self.samples)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Patch)) == 2  # 95 and 68 envelopes

    @pytest.mark.parametrize("diff", [None, "ecdf", "iecdf", "eppf"])
    def test_diff_modes(self, diff):
        p = viz.predictive_ecdf(self.samples, data=self.data, diff=diff)
        assert isinstance(p, bokeh.plotting.figure)

    @pytest.mark.parametrize("data_staircase", [True, False])
    def test_data_overlay(self, data_staircase):
        p = viz.predictive_ecdf(
            self.samples, data=self.data, data_staircase=data_staircase
        )
        assert isinstance(p, bokeh.plotting.figure)

    def test_boolean_diff_deprecated(self):
        with pytest.warns(DeprecationWarning):
            viz.predictive_ecdf(self.samples, diff=True)
        with pytest.warns(DeprecationWarning):
            viz.predictive_ecdf(self.samples, diff=False)

    def test_single_percentile(self):
        p = viz.predictive_ecdf(self.samples, percentiles=(95,))
        assert len(glyphs(p, bokeh.models.Patch)) == 1

    def test_too_many_percentiles(self):
        with pytest.raises(RuntimeError):
            viz.predictive_ecdf(self.samples, percentiles=(99, 95, 90, 80, 60))

    def test_invalid_color(self):
        with pytest.raises(RuntimeError):
            viz.predictive_ecdf(self.samples, color="mauve")

    def test_samples_must_be_2d(self):
        with pytest.raises(RuntimeError):
            viz.predictive_ecdf(self.samples.flatten())

    def test_samples_bad_type(self):
        with pytest.raises(RuntimeError):
            viz.predictive_ecdf(list(self.samples))

    def test_data_shape_mismatch(self):
        with pytest.raises(RuntimeError):
            viz.predictive_ecdf(self.samples, data=self.data[:-1])

    def test_few_samples_warns(self):
        with pytest.warns(UserWarning, match="very few samples"):
            viz.predictive_ecdf(self.samples[:20])


# ---------------------------------------------------------------------------
# predictive_regression
# ---------------------------------------------------------------------------


class TestPredictiveRegression:
    rng = np.random.default_rng(5)
    samples_x = np.linspace(0.0, 1.0, 20)
    samples = samples_x + rng.normal(size=(150, 20))
    data = np.column_stack((samples_x, samples_x + rng.normal(size=20)))

    def test_basic(self):
        p = viz.predictive_regression(self.samples, self.samples_x)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Patch)) == 2

    def test_data_overlay(self):
        p = viz.predictive_regression(self.samples, self.samples_x, data=self.data)
        assert len(glyphs(p, bokeh.models.Scatter)) == 1

    def test_data_as_tuple(self):
        p = viz.predictive_regression(
            self.samples, self.samples_x, data=(self.data[:, 0], self.data[:, 1])
        )
        assert isinstance(p, bokeh.plotting.figure)

    def test_diff(self):
        p = viz.predictive_regression(
            self.samples, self.samples_x, data=self.data, diff=True
        )
        assert isinstance(p, bokeh.plotting.figure)

    def test_diff_x_mismatch(self):
        data = self.data.copy()
        data[:, 0] += 0.1
        with pytest.raises(ValueError):
            viz.predictive_regression(self.samples, self.samples_x, data=data, diff=True)

    def test_samples_x_shape_mismatch(self):
        with pytest.raises(ValueError):
            viz.predictive_regression(self.samples, self.samples_x[:-1])

    def test_samples_bad_type(self):
        with pytest.raises(RuntimeError):
            viz.predictive_regression(list(self.samples), self.samples_x)

    def test_too_many_percentiles(self):
        with pytest.raises(RuntimeError):
            viz.predictive_regression(
                self.samples, self.samples_x, percentiles=[99, 95, 90, 80, 60]
            )

    def test_invalid_color(self):
        with pytest.raises(RuntimeError):
            viz.predictive_regression(self.samples, self.samples_x, color="mauve")


# ---------------------------------------------------------------------------
# sbc_rank_ecdf
# ---------------------------------------------------------------------------


def make_sbc_df(L=100, n=150, seed=6):
    rng = np.random.default_rng(seed)
    dfs = []
    for param in ["alpha", "beta"]:
        dfs.append(
            pd.DataFrame(
                {
                    "parameter": param,
                    "rank_statistic": rng.integers(0, L, size=n),
                    "warning_code": rng.choice([0, 1], size=n),
                    "L": L,
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


class TestSbcRankEcdf:
    @pytest.mark.parametrize("diff", [True, False])
    def test_basic(self, diff):
        p = viz.sbc_rank_ecdf(make_sbc_df(), diff=diff)
        assert isinstance(p, bokeh.plotting.figure)
        # One envelope patch and one scatter set per parameter.
        assert len(glyphs(p, bokeh.models.Patch)) == 1
        assert len(glyphs(p, bokeh.models.Scatter)) == 2

    def test_polars_input(self):
        p = viz.sbc_rank_ecdf(pl.from_pandas(make_sbc_df()))
        assert isinstance(p, bokeh.plotting.figure)

    def test_staircase(self):
        p = viz.sbc_rank_ecdf(make_sbc_df(), staircase=True)
        assert len(glyphs(p, bokeh.models.Line)) >= 2

    def test_parameter_subset(self):
        p = viz.sbc_rank_ecdf(make_sbc_df(), parameters=["alpha"])
        assert len(glyphs(p, bokeh.models.Scatter)) == 1

    def test_no_envelope(self):
        p = viz.sbc_rank_ecdf(make_sbc_df(), show_envelope=False)
        assert len(glyphs(p, bokeh.models.Patch)) == 0

    def test_color_by_warning_code(self):
        p = viz.sbc_rank_ecdf(make_sbc_df(), color_by_warning_code=True)
        assert len(glyphs(p, bokeh.models.Scatter)) == 2  # two warning codes

    def test_color_by_warning_code_marker_color_clash(self):
        with pytest.raises(RuntimeError):
            viz.sbc_rank_ecdf(
                make_sbc_df(),
                color_by_warning_code=True,
                marker_kwargs=dict(color="red"),
            )

    def test_color_by_warning_code_staircase_clash(self):
        with pytest.raises(RuntimeError):
            viz.sbc_rank_ecdf(make_sbc_df(), color_by_warning_code=True, staircase=True)

    def test_missing_input(self):
        with pytest.raises(RuntimeError):
            viz.sbc_rank_ecdf()


# ---------------------------------------------------------------------------
# parcoord
# ---------------------------------------------------------------------------


class TestParcoord:
    def test_all_input_formats(self, samples_any_format):
        p = viz.parcoord(samples_any_format)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.MultiLine)) == 1

    def test_default_screens_ppc_and_log_lik(self, samples_dict):
        p = viz.parcoord(samples_dict)
        factors = list(p.x_range.factors)
        assert factors == ["mu", "sigma", "alpha[1]", "alpha[2]"]

    def test_include_ppc_and_log_lik(self, samples_dict):
        p = viz.parcoord(samples_dict, include_ppc=True, include_log_lik=True)
        factors = list(p.x_range.factors)
        assert "y_ppc[1]" in factors and "log_lik[1]" in factors

    def test_parameter_labels(self, samples_dict):
        p = viz.parcoord(samples_dict, parameters=[("mu", "µ"), "sigma"])
        assert list(p.x_range.factors) == ["µ", "sigma"]

    def test_divergences_plotted(self, samples_dict_div):
        p = viz.parcoord(samples_dict_div)
        assert len(glyphs(p, bokeh.models.MultiLine)) == 2

    @pytest.mark.parametrize("transformation", ["minmax", "rank", np.exp])
    def test_transformations(self, samples_dict, transformation):
        p = viz.parcoord(samples_dict, transformation=transformation)
        assert isinstance(p, bokeh.plotting.figure)

    def test_transformation_dict(self, samples_dict):
        p = viz.parcoord(
            samples_dict,
            parameters=["mu", "sigma"],
            transformation={"sigma": np.log},
        )
        assert isinstance(p, bokeh.plotting.figure)

    def test_color_by_chain(self, samples_dict):
        p = viz.parcoord(samples_dict, color_by_chain=True)
        assert isinstance(p, bokeh.plotting.figure)

    def test_parameters_and_omit_clash(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.parcoord(samples_dict, parameters=["mu"], omit="sigma")

    def test_x_range_not_allowed(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.parcoord(samples_dict, x_range=[0, 1])

    def test_color_and_color_by_chain_clash(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.parcoord(
                samples_dict, color_by_chain=True, line_kwargs=dict(color="red")
            )


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_all_input_formats(self, samples_any_format):
        out = viz.trace(samples_any_format)
        figs = collect_figures(out)
        assert len(figs) == 4  # mu, sigma, alpha[1], alpha[2]

    def test_single_parameter_returns_figure(self, samples_dict):
        p = viz.trace(samples_dict, parameters=["mu"])
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Line)) == N_CHAINS
        assert p.yaxis.axis_label == "mu"

    def test_parameter_labels(self, samples_dict):
        p = viz.trace(samples_dict, parameters=[("mu", "µ")])
        assert p.yaxis.axis_label == "µ"

    def test_multiple_parameters_layout(self, samples_dict):
        out = viz.trace(samples_dict, parameters=["mu", "sigma"])
        figs = collect_figures(out)
        assert len(figs) == 2
        # x-ranges of all trace plots are linked.
        assert figs[0].x_range is figs[1].x_range

    def test_omit(self, samples_dict):
        # omit patterns match against the base (non-indexed) variable name,
        # so 'alpha' screens out both alpha[1] and alpha[2].
        out = viz.trace(samples_dict, omit=[re.compile("alpha")])
        figs = collect_figures(out)
        assert len(figs) == 2  # mu and sigma

    def test_parameters_and_omit_clash(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.trace(samples_dict, parameters=["mu"], omit="sigma")

    def test_y_axis_label_not_allowed(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.trace(samples_dict, y_axis_label="nope")

    def test_color_not_allowed(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.trace(samples_dict, line_kwargs=dict(color="red"))


# ---------------------------------------------------------------------------
# corner
# ---------------------------------------------------------------------------


class TestCorner:
    def test_all_input_formats(self, samples_any_format):
        out = viz.corner(samples_any_format)
        figs = collect_figures(out)
        assert len(figs) == 10  # lower triangle of 4 parameters

    def test_single_parameter_histogram(self, samples_dict):
        p = viz.corner(samples_dict, parameters=["mu"])
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Line)) == 1

    def test_single_parameter_ecdf_dots(self, samples_dict):
        p = viz.corner(samples_dict, parameters=["mu"], plot_ecdf=True)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.Scatter)) == 2  # samples + divergences

    def test_single_parameter_ecdf_staircase(self, samples_dict):
        p = viz.corner(
            samples_dict, parameters=["mu"], plot_ecdf=True, ecdf_staircase=True
        )
        assert isinstance(p, bokeh.plotting.figure)

    def test_two_parameters(self, samples_dict):
        out = viz.corner(samples_dict, parameters=["mu", "sigma"])
        figs = collect_figures(out)
        assert len(figs) == 3

    def test_plot_ecdf_grid(self, samples_dict):
        out = viz.corner(samples_dict, parameters=["mu", "sigma"], plot_ecdf=True)
        figs = collect_figures(out)
        assert len(figs) == 3

    def test_parameter_labels(self, samples_dict):
        out = viz.corner(samples_dict, parameters=[("mu", "µ"), ("sigma", "σ")])
        figs = collect_figures(out)
        labels = [f.xaxis.axis_label for f in figs]
        assert "µ" in labels and "σ" in labels

    def test_show_contours(self, samples_dict):
        out = viz.corner(
            samples_dict, parameters=["mu", "sigma"], show_contours=True, bins_2d=20
        )
        figs = collect_figures(out)
        assert any(len(glyphs(f, bokeh.models.MultiLine)) > 0 for f in figs)

    def test_divergences(self, samples_dict_div):
        out = viz.corner(samples_dict_div, parameters=["mu", "sigma"])
        assert len(collect_figures(out)) == 3

    def test_color_by_chain(self, samples_dict):
        out = viz.corner(samples_dict, parameters=["mu", "sigma"], color_by_chain=True)
        assert len(collect_figures(out)) == 3

    def test_max_plotted(self, samples_dict):
        p = viz.corner(
            samples_dict, parameters=["mu"], plot_ecdf=True, max_plotted=50
        )
        scatter_renderers = [
            r for r in p.renderers if isinstance(r.glyph, bokeh.models.Scatter)
        ]
        assert len(scatter_renderers[0].data_source.data["mu"]) == 50

    def test_numpy_input(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(500, 3))
        out = viz.corner(samples)
        figs = collect_figures(out)
        assert len(figs) == 6
        labels = [f.xaxis.axis_label for f in figs]
        assert "0" in labels and "2" in labels

    def test_numpy_input_index_subset(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(500, 3))
        out = viz.corner(samples, parameters=[0, 2])
        assert len(collect_figures(out)) == 3

    def test_numpy_input_bad_parameters(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(500, 3))
        with pytest.raises(RuntimeError):
            viz.corner(samples, parameters=["mu"])

    def test_numpy_input_labeled_indices(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(500, 3))
        out = viz.corner(samples, parameters=[(0, "x"), (2, "z")])
        figs = collect_figures(out)
        assert len(figs) == 3
        labels = [f.xaxis.axis_label for f in figs]
        assert "x" in labels and "z" in labels

    def test_too_many_parameters(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(100, 7))
        with pytest.raises(RuntimeError):
            viz.corner(samples)

    def test_parameters_and_omit_clash(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.corner(samples_dict, parameters=["mu"], omit="sigma")

    def test_missing_parameter(self, samples_dict):
        with pytest.raises(RuntimeError):
            viz.corner(samples_dict, parameters=["not_a_param"])

    def test_datashade_deprecated(self, samples_dict):
        with pytest.raises(DeprecationWarning):
            viz.corner(samples_dict, datashade=True)


# ---------------------------------------------------------------------------
# contour and contour lines
# ---------------------------------------------------------------------------


def make_gaussian_grid(n=50):
    x = np.linspace(-3.0, 3.0, n)
    y = np.linspace(-3.0, 3.0, n)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2)
    return x, y, X, Y, Z


class TestContour:
    def test_1d_xy(self):
        x, y, _, _, Z = make_gaussian_grid()
        p = viz.contour(x, y, Z)
        assert isinstance(p, bokeh.plotting.figure)
        assert len(glyphs(p, bokeh.models.MultiLine)) == 1

    def test_2d_xy(self):
        _, _, X, Y, Z = make_gaussian_grid()
        p = viz.contour(X, Y, Z)
        assert isinstance(p, bokeh.plotting.figure)

    def test_custom_levels(self):
        x, y, _, _, Z = make_gaussian_grid()
        p = viz.contour(x, y, Z, levels=[0.5, 0.9])
        assert isinstance(p, bokeh.plotting.figure)

    def test_existing_figure(self):
        x, y, _, _, Z = make_gaussian_grid()
        p0 = bokeh.plotting.figure()
        p = viz.contour(x, y, Z, p=p0)
        assert p is p0

    def test_dimension_mismatch(self):
        x, y, X, Y, Z = make_gaussian_grid()
        with pytest.raises(RuntimeError):
            viz.contour(x, Y, Z)

    def test_shape_mismatch(self):
        x, y, _, _, Z = make_gaussian_grid()
        with pytest.raises(RuntimeError):
            viz.contour(x, y, Z[:-1, :])

    def test_overlaid_with_p_clash(self):
        x, y, _, _, Z = make_gaussian_grid()
        with pytest.raises(RuntimeError):
            viz.contour(x, y, Z, overlaid=True, p=bokeh.plotting.figure())

    def test_fill_not_implemented(self):
        x, y, _, _, Z = make_gaussian_grid()
        with pytest.raises(NotImplementedError):
            viz.contour(x, y, Z, fill=True)


class TestContourLinesFromSamples:
    rng = np.random.default_rng(9)
    x = rng.normal(size=2000)
    y = 0.5 * x + rng.normal(size=2000)

    def test_basic(self):
        xs, ys = viz.contour_lines_from_samples(self.x, self.y)
        assert len(xs) == len(ys) and len(xs) > 0
        for xc, yc in zip(xs, ys):
            assert len(xc) == len(yc)
            assert np.all((xc >= self.x.min()) & (xc <= self.x.max()))

    def test_scalar_level(self):
        xs, ys = viz.contour_lines_from_samples(self.x, self.y, levels=0.95)
        assert len(xs) > 0

    def test_no_smoothing(self):
        xs, ys = viz.contour_lines_from_samples(self.x, self.y, smooth=None)
        assert len(xs) > 0

    def test_extend_domain(self):
        xs, ys = viz.contour_lines_from_samples(self.x, self.y, extend_domain=True)
        assert len(xs) > 0

    @pytest.mark.parametrize("levels", [0.0, -0.1, 1.5])
    def test_invalid_levels(self, levels):
        with pytest.raises(ValueError):
            viz.contour_lines_from_samples(self.x, self.y, levels=levels)

    @pytest.mark.parametrize("bins", [0, -5, 2.5])
    def test_invalid_bins(self, bins):
        with pytest.raises(ValueError):
            viz.contour_lines_from_samples(self.x, self.y, bins=bins)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------


class TestQToColor:
    colors = ["c0", "c1", "c2", "c3"]

    def test_scalar(self):
        assert viz.q_to_color(0.5, self.colors, low=0.0, high=1.0) == "c2"

    def test_scalar_out_of_range(self):
        assert (
            viz.q_to_color(-1.0, self.colors, low=0.0, high=1.0, low_color="lo") == "lo"
        )
        assert (
            viz.q_to_color(2.0, self.colors, low=0.0, high=1.0, high_color="hi") == "hi"
        )

    def test_scalar_nan(self):
        assert (
            viz.q_to_color(np.nan, self.colors, low=0.0, high=1.0, nan_color="na")
            == "na"
        )

    def test_scalar_requires_low_high(self):
        with pytest.raises(ValueError):
            viz.q_to_color(0.5, self.colors)

    def test_array(self):
        x = np.array([np.nan, -1.0, 0.0, 0.5, 1.0, 2.0])
        result = viz.q_to_color(
            x,
            self.colors,
            low=0.0,
            high=1.0,
            low_color="lo",
            high_color="hi",
            nan_color="na",
        )
        assert result == ["na", "lo", "c0", "c2", "c3", "hi"]

    def test_array_infers_range(self):
        result = viz.q_to_color(np.array([0.0, 1.0]), self.colors)
        assert result == ["c0", "c3"]


class TestMplCmapToColorMapper:
    def test_viridis(self):
        mapper = viz.mpl_cmap_to_color_mapper("viridis")
        assert isinstance(mapper, bokeh.models.LinearColorMapper)
        assert len(mapper.palette) == 256
        assert all(c.startswith("#") for c in mapper.palette)


# ---------------------------------------------------------------------------
# Parameter parsing and samples conversion helpers
# ---------------------------------------------------------------------------


class TestParseOmit:
    def test_default_screens_ppc_and_log_lik(self):
        omit = viz._parse_omit(None, include_ppc=False, include_log_lik=False)
        assert not bebi103.stan._screen_param("y_ppc[1]", omit)
        assert not bebi103.stan._screen_param("log_lik[1]", omit)
        assert bebi103.stan._screen_param("mu", omit)

    def test_include_ppc(self):
        omit = viz._parse_omit(None, include_ppc=True, include_log_lik=False)
        assert bebi103.stan._screen_param("y_ppc[1]", omit)
        assert not bebi103.stan._screen_param("log_lik[1]", omit)

    def test_include_log_lik(self):
        omit = viz._parse_omit(None, include_ppc=False, include_log_lik=True)
        assert not bebi103.stan._screen_param("y_ppc[1]", omit)
        assert bebi103.stan._screen_param("log_lik[1]", omit)

    def test_string_and_regex_entries(self):
        # Patterns match against the base (non-indexed) variable name.
        omit = viz._parse_omit(
            ["sigma", re.compile("alpha")],
            include_ppc=True,
            include_log_lik=True,
        )
        assert not bebi103.stan._screen_param("sigma", omit)
        assert not bebi103.stan._screen_param("alpha[2]", omit)
        assert bebi103.stan._screen_param("mu", omit)

    def test_input_not_mutated(self):
        omit_in = ["sigma"]
        viz._parse_omit(omit_in, include_ppc=False, include_log_lik=False)
        assert omit_in == ["sigma"]


class TestParseParameters:
    def test_strings(self):
        parameters, labels = viz._parse_parameters(["mu", "sigma"])
        assert parameters == ["mu", "sigma"]
        assert labels == ["mu", "sigma"]

    def test_tuples(self):
        parameters, labels = viz._parse_parameters([("mu", "µ"), "sigma"])
        assert parameters == ["mu", "sigma"]
        assert labels == ["µ", "sigma"]

    def test_bad_tuple_length(self):
        with pytest.raises(RuntimeError):
            viz._parse_parameters([("mu", "µ", "extra")])

    def test_bad_type(self):
        with pytest.raises(RuntimeError):
            viz._parse_parameters([42])


class TestSamplesToDf:
    def test_dict_input(self, samples_dict):
        omit = viz._parse_omit(None, False, False)
        parameters, df = viz._samples_to_df(samples_dict, None, omit)
        assert parameters == ["mu", "sigma", "alpha[1]", "alpha[2]"]
        assert set(["chain__", "draw__", "divergent__"]) <= set(df.columns)
        assert len(df) == N_CHAINS * N_DRAWS
        assert df["chain__"].min() == 1 and df["draw__"].min() == 1

    def test_polars_input(self, samples_dict):
        samples = bebi103.stan.convert_samples(samples_dict, "polars")
        omit = viz._parse_omit(None, False, False)
        _, df = viz._samples_to_df(samples, None, omit)
        assert isinstance(df, pd.DataFrame)

    def test_plain_dataframe_gets_chain_and_draw(self):
        plain = pd.DataFrame({"mu": np.arange(10.0), "sigma": np.arange(10.0)})
        omit = viz._parse_omit(None, False, False)
        _, df = viz._samples_to_df(plain, None, omit)
        assert np.all(df["chain__"] == 1)
        assert np.allclose(df["draw__"], np.arange(1, 11))
        assert np.all(df["divergent__"] == 0.0)
        # Input must not be mutated.
        assert "chain__" not in plain.columns

    def test_missing_parameter(self, samples_dict):
        omit = viz._parse_omit(None, False, False)
        with pytest.raises(RuntimeError):
            viz._samples_to_df(samples_dict, ["not_a_param"], omit)

    def test_explicit_parameters_with_labels(self, samples_dict):
        omit = viz._parse_omit(None, False, False)
        parameters, df = viz._samples_to_df(samples_dict, [("mu", "µ")], omit)
        assert parameters == [("mu", "µ")]
        assert "mu" in df.columns


class TestDataRange:
    def test_margin(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 10.0]})
        x_range, y_range = viz._data_range(df, "x", "y", margin=0.1)
        assert np.allclose(x_range, [-0.1, 1.1])
        assert np.allclose(y_range, [-1.0, 11.0])
