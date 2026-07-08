"""Test suite for bebi103.hmc.

The inferential-calibration machinery is intentionally not covered yet
(inferential_calibration, _multiprocessing_spawn_context,
_perform_inferential_calibration, _get_prior_sds,
_tidy_single_inferential_calibration_result,
_tidy_inferential_calibration_output).
"""

import logging
import re
import types

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray

import bebi103.hmc as st


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

REQUIRED_HMC = (
    "divergent__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "energy__",
)


def _make_hmc_diagnostics(n_chains, n_draws, rng, clean=True):
    """Build a set of HMC diagnostic arrays."""
    diagnostics = {
        "lp__": rng.normal(size=(n_chains, n_draws)),
        "accept_stat__": np.full((n_chains, n_draws), 0.9),
        "stepsize__": np.full((n_chains, n_draws), 0.1),
        "treedepth__": np.full((n_chains, n_draws), 3.0),
        "n_leapfrog__": np.full((n_chains, n_draws), 7.0),
        "energy__": rng.normal(size=(n_chains, n_draws)),
        "divergent__": np.zeros((n_chains, n_draws)),
    }
    if not clean:
        # Divergences in the first chain, saturated treedepth in the second.
        diagnostics["divergent__"][0, :] = 1.0
        diagnostics["treedepth__"][1, :] = 10.0
    return diagnostics


def make_samples_dict(n_chains=4, n_draws=400, seed=0, clean=True):
    """A validated samples dict with expectands and all HMC diagnostics."""
    rng = np.random.default_rng(seed)
    if clean:
        expectands = {
            "mu": rng.normal(size=(n_chains, n_draws)),
            "sigma": np.abs(rng.normal(size=(n_chains, n_draws))) + 1.0,
        }
    else:
        # Random walks: high split Rhat and tiny effective sample sizes.
        expectands = {
            "mu": np.cumsum(rng.normal(size=(n_chains, n_draws)), axis=1),
            "sigma": np.abs(rng.normal(size=(n_chains, n_draws))) + 1.0,
        }
    diagnostics = _make_hmc_diagnostics(n_chains, n_draws, rng, clean=clean)
    return {**expectands, **diagnostics}


@pytest.fixture
def samples_dict():
    return make_samples_dict()


@pytest.fixture
def bad_samples_dict():
    return make_samples_dict(clean=False, seed=1)


def make_arviz_tree(n_chains=4, n_draws=200, seed=1, warmup=False):
    """Build an ArviZ-style xarray.DataTree with posterior + sample_stats."""
    rng = np.random.default_rng(seed)
    coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}
    post = xarray.Dataset(
        {
            "mu": (("chain", "draw"), rng.normal(size=(n_chains, n_draws))),
            "beta": (
                ("chain", "draw", "beta_dim"),
                rng.normal(size=(n_chains, n_draws, 2)),
            ),
        },
        coords=coords,
    )
    stats = xarray.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros((n_chains, n_draws))),
            "lp": (("chain", "draw"), rng.normal(size=(n_chains, n_draws))),
        },
        coords=coords,
    )
    tree = {"posterior": post, "sample_stats": stats}
    if warmup:
        tree["warmup_posterior"] = xarray.Dataset(
            {
                "mu": (("chain", "draw"), rng.normal(size=(n_chains, 5))),
                "beta": (
                    ("chain", "draw", "beta_dim"),
                    rng.normal(size=(n_chains, 5, 2)),
                ),
            },
            coords={"chain": np.arange(n_chains), "draw": np.arange(5)},
        )
    return xarray.DataTree.from_dict(tree)


@pytest.fixture(scope="session")
def cmdstan_fit():
    """Compile and sample a trivial model once for integration tests."""
    cmdstanpy = pytest.importorskip("cmdstanpy")
    import os
    import tempfile

    code = """
data { int<lower=0> N; vector[N] y; }
parameters { real mu; real<lower=0> sigma; vector[2] beta; }
model {
  mu ~ normal(0, 5);
  sigma ~ normal(0, 5);
  beta ~ normal(0, 1);
  y ~ normal(mu, sigma);
}
"""
    try:
        tmpdir = tempfile.mkdtemp()
        stan_file = os.path.join(tmpdir, "model.stan")
        with open(stan_file, "w") as f:
            f.write(code)
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        rng = np.random.default_rng(0)
        fit = model.sample(
            data={"N": 20, "y": rng.normal(size=20).tolist()},
            chains=2,
            iter_warmup=200,
            iter_sampling=200,
            seed=1,
            show_progress=False,
            show_console=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CmdStan sampling unavailable: {exc}")
    return fit


# ---------------------------------------------------------------------------
# Deprecated / environment functions
# ---------------------------------------------------------------------------

class TestEnvironmentFunctions:
    def test_stanmodel_deprecated(self):
        with pytest.raises(RuntimeError):
            st.StanModel()

    def test_install_cmdstan_colab_outside_colab(self):
        with pytest.raises(RuntimeError):
            st.install_cmdstan_colab()

    def test_include_path(self):
        path = st.include_path()
        assert isinstance(path, str)
        assert path.endswith("stan_include")

    def test_cmdstan_version_returns_str(self):
        assert isinstance(st.cmdstan_version(), str)

    def test_clean_cmdstan_refuses_root(self):
        with pytest.raises(RuntimeError):
            st.clean_cmdstan(path="/")

    def test_clean_cmdstan_removes_artifacts(self, tmp_path):
        (tmp_path / "model.stan").write_text("")
        for ext in (".hpp", ".o", ".d"):
            (tmp_path / f"model{ext}").write_text("")
        exe = tmp_path / "model"
        exe.write_text("")
        exe.chmod(0o755)

        st.clean_cmdstan(path=str(tmp_path))

        for ext in (".hpp", ".o", ".d"):
            assert not (tmp_path / f"model{ext}").exists()
        assert not exe.exists()
        assert (tmp_path / "model.stan").exists()


# ---------------------------------------------------------------------------
# df_to_datadict_hier
# ---------------------------------------------------------------------------

class TestDfToDatadictHier:
    @staticmethod
    def _hier_df():
        return pd.DataFrame(
            {
                "day": ["mon", "mon", "mon", "mon", "tue", "tue"],
                "batch": [1, 1, 2, 2, 1, 1],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

    def test_basic_structure(self):
        df = self._hier_df()
        data, new_df = st.df_to_datadict_hier(
            df, level_cols=["day", "batch"], data_cols=["x"]
        )
        assert data["N"] == 6
        assert data["J_1"] == 2  # two days
        assert data["J_2"] == 3  # (mon,1),(mon,2),(tue,1)
        # index_1 maps each level-2 group to its level-1 parent.
        np.testing.assert_array_equal(data["index_1"], np.array([1, 1, 2]))
        # index_2 maps each data point to its level-2 group.
        assert len(data["index_2"]) == 6
        assert set(data["index_2"]) == {1, 2, 3}
        assert len(data["x"]) == 6
        # New stan-index columns added.
        assert "day_stan" in new_df.columns
        assert "batch_stan" in new_df.columns

    def test_requires_all_args(self):
        df = self._hier_df()
        with pytest.raises(RuntimeError):
            st.df_to_datadict_hier(df, level_cols=["day"])
        with pytest.raises(RuntimeError):
            st.df_to_datadict_hier(df, data_cols=["x"])

    def test_sort_cols_must_be_list(self):
        df = self._hier_df()
        with pytest.raises(RuntimeError):
            st.df_to_datadict_hier(
                df, level_cols=["day"], data_cols=["x"], sort_cols="x"
            )

    def test_cowardly(self):
        df = self._hier_df()
        _, new_df = st.df_to_datadict_hier(
            df, level_cols=["day", "batch"], data_cols=["x"]
        )
        with pytest.raises(RuntimeError):
            st.df_to_datadict_hier(
                new_df, level_cols=["day", "batch"], data_cols=["x"], cowardly=True
            )

    def test_polars_in_polars_out(self):
        df = pl.from_pandas(self._hier_df())
        data, new_df = st.df_to_datadict_hier(
            df, level_cols=["day", "batch"], data_cols=["x"]
        )
        assert isinstance(new_df, pl.DataFrame)
        assert data["N"] == 6

    def test_invalid_column_name(self):
        df = self._hier_df().rename(columns={"x": "x y"})
        with pytest.raises(RuntimeError):
            st.df_to_datadict_hier(
                df, level_cols=["day", "batch"], data_cols=["x y"]
            )


# ---------------------------------------------------------------------------
# convert_samples and the dataframe/dict helpers
# ---------------------------------------------------------------------------

class TestConvertSamples:
    def test_dict_to_dict_identity(self, samples_dict):
        assert st.convert_samples(samples_dict, "dict") is samples_dict

    def test_dict_to_pandas(self, samples_dict):
        df = st.convert_samples(samples_dict, "pandas")
        assert isinstance(df, pd.DataFrame)
        assert "chain__" in df.columns and "draw__" in df.columns

    def test_dict_to_polars(self, samples_dict):
        df = st.convert_samples(samples_dict, "polars")
        assert isinstance(df, pl.DataFrame)

    def test_dataframe_roundtrip(self, samples_dict):
        df = st.convert_samples(samples_dict, "polars")
        back = st.convert_samples(df, "dict")
        assert set(back) == set(samples_dict)
        for key in samples_dict:
            np.testing.assert_allclose(back[key], samples_dict[key])

    def test_arviz_tree_to_dict(self):
        tree = make_arviz_tree()
        d = st.convert_samples(tree, "dict")
        assert "mu" in d
        assert "beta[1]" in d and "beta[2]" in d
        assert "divergent__" in d

    def test_arviz_to_arviz_identity(self):
        tree = make_arviz_tree()
        assert st.convert_samples(tree, "arviz") is tree

    def test_bad_out_format(self, samples_dict):
        with pytest.raises(RuntimeError):
            st.convert_samples(samples_dict, "not_a_format")

    def test_inc_warmup_only_for_cmdstanpy(self, samples_dict):
        with pytest.raises(ValueError):
            st.convert_samples(samples_dict, "dict", inc_warmup=True)

    def test_arviz_out_requires_cmdstanpy(self, samples_dict):
        with pytest.raises(ValueError):
            st.convert_samples(samples_dict, "arviz")

    def test_unsupported_input_type(self):
        with pytest.raises(ValueError):
            st.convert_samples(42, "dict")


class TestDictDataFrameHelpers:
    def test_dict_to_dataframe_shape(self, samples_dict):
        df = st._dict_to_dataframe(samples_dict, dataframe_package="pandas")
        n_chains, n_draws = samples_dict["mu"].shape
        assert len(df) == n_chains * n_draws
        assert list(df["chain__"].unique()) == list(range(1, n_chains + 1))
        assert set(df["draw__"].unique()) == set(range(1, n_draws + 1))

    def test_dict_to_dataframe_bad_package(self, samples_dict):
        with pytest.raises(ValueError):
            st._dict_to_dataframe(samples_dict, dataframe_package="numpy")

    def test_dataframe_to_dict_roundtrip(self, samples_dict):
        for package in ("polars", "pandas"):
            df = st._dict_to_dataframe(samples_dict, dataframe_package=package)
            back = st._dataframe_to_dict(df)
            for key in samples_dict:
                np.testing.assert_allclose(back[key], samples_dict[key])

    def test_dataframe_to_dict_shuffled(self, samples_dict):
        # Rows out of order should be sorted back into a chain-by-draw grid.
        df = st._dict_to_dataframe(samples_dict, dataframe_package="pandas")
        shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        back = st._dataframe_to_dict(shuffled)
        for key in samples_dict:
            np.testing.assert_allclose(back[key], samples_dict[key])

    def test_dataframe_to_dict_requires_dataframe(self):
        with pytest.raises(ValueError):
            st._dataframe_to_dict({"chain__": [1]})

    def test_dataframe_to_dict_requires_chain_draw(self):
        for df in (pd.DataFrame({"mu": [1.0, 2.0]}), pl.DataFrame({"mu": [1.0, 2.0]})):
            with pytest.raises(ValueError):
                st._dataframe_to_dict(df)

    def test_dataframe_to_dict_incomplete_grid(self, samples_dict):
        df = st._dict_to_dataframe(samples_dict, dataframe_package="pandas")
        with pytest.raises(ValueError):
            st._dataframe_to_dict(df.iloc[:-1])


# ---------------------------------------------------------------------------
# _dict_from_arviz
# ---------------------------------------------------------------------------

class TestDictFromArviz:
    def test_flattens_groups(self):
        tree = make_arviz_tree(n_chains=3, n_draws=50)
        d = st._dict_from_arviz(tree)
        assert d["mu"].shape == (3, 50)
        assert d["beta[1]"].shape == (3, 50)
        assert d["beta[2]"].shape == (3, 50)
        # sample_stats renamed to CmdStanPy diagnostic names.
        assert "divergent__" in d
        assert "lp__" in d
        assert all(v.dtype == np.float64 for v in d.values())

    def test_warmup_prepended(self):
        tree = make_arviz_tree(n_chains=2, n_draws=20, warmup=True)
        d_no = st._dict_from_arviz(tree, inc_warmup=False)
        d_yes = st._dict_from_arviz(tree, inc_warmup=True)
        assert d_no["mu"].shape == (2, 20)
        assert d_yes["mu"].shape == (2, 25)

    def test_warmup_warns_when_absent(self):
        tree = make_arviz_tree(warmup=False)
        with pytest.warns(UserWarning):
            st._dict_from_arviz(tree, inc_warmup=True)


# ---------------------------------------------------------------------------
# extract_variable
# ---------------------------------------------------------------------------

class TestExtractVariable:
    def test_scalar(self, samples_dict):
        out = st.extract_variable(samples_dict, "mu")
        assert out.shape == samples_dict["mu"].shape
        assert out is samples_dict["mu"]

    def test_vector(self):
        nc, nd = 2, 10
        d = {
            "beta[1]": np.ones((nc, nd)),
            "beta[2]": 2 * np.ones((nc, nd)),
            "beta[3]": 3 * np.ones((nc, nd)),
        }
        out = st.extract_variable(d, "beta")
        assert out.shape == (nc, nd, 3)
        np.testing.assert_array_equal(out[:, :, 0], 1.0)
        np.testing.assert_array_equal(out[:, :, 2], 3.0)

    def test_matrix(self):
        nc, nd = 2, 5
        d = {
            f"A[{i},{j}]": (10 * i + j) * np.ones((nc, nd))
            for i in (1, 2)
            for j in (1, 2)
        }
        out = st.extract_variable(d, "A")
        assert out.shape == (nc, nd, 2, 2)
        # out[:, :, 1, 0] is one-indexed A[2,1] = 10*2 + 1 = 21.
        np.testing.assert_array_equal(out[:, :, 1, 0], 21.0)

    def test_missing_variable(self, samples_dict):
        with pytest.raises(ValueError):
            st.extract_variable(samples_dict, "does_not_exist")

    def test_incomplete_variable(self):
        d = {
            "beta[1]": np.ones((2, 3)),
            "beta[3]": np.ones((2, 3)),  # beta[2] missing
        }
        with pytest.raises(ValueError):
            st.extract_variable(d, "beta")

    def test_inconsistent_indices(self):
        d = {
            "beta[1]": np.ones((2, 3)),
            "beta[1,2]": np.ones((2, 3)),
        }
        with pytest.raises(ValueError):
            st.extract_variable(d, "beta")


# ---------------------------------------------------------------------------
# _split_dict / _validate_dict / _validate_dict_hmc_diagnostics
# ---------------------------------------------------------------------------

class TestSplitAndValidate:
    def test_split_dict(self, samples_dict):
        expectands, diagnostics = st._split_dict(samples_dict)
        assert set(expectands) == {"mu", "sigma"}
        assert set(diagnostics) == set(REQUIRED_HMC) | {"lp__"}

    def test_split_dict_non_string_key(self):
        with pytest.raises(ValueError):
            st._split_dict({1: np.ones((2, 2))})

    def test_validate_dict_ok(self, samples_dict):
        assert st._validate_dict(samples_dict) is None

    def test_validate_not_a_dict(self):
        with pytest.raises(ValueError):
            st._validate_dict([1, 2, 3])

    def test_validate_reserved_key(self):
        with pytest.raises(ValueError):
            st._validate_dict({"chain__": np.ones((2, 2))})

    def test_validate_not_array(self):
        with pytest.raises(ValueError):
            st._validate_dict({"mu": [[1.0, 2.0]]})

    def test_validate_not_2d(self):
        with pytest.raises(ValueError):
            st._validate_dict({"mu": np.ones(4)})

    def test_validate_not_float64(self):
        with pytest.raises(ValueError):
            st._validate_dict({"mu": np.ones((2, 2), dtype=np.int64)})

    def test_validate_mismatched_shapes(self):
        with pytest.raises(ValueError):
            st._validate_dict(
                {"mu": np.ones((2, 3)), "sigma": np.ones((2, 4))}
            )

    def test_validate_requires_hmc(self):
        with pytest.raises(ValueError):
            st._validate_dict(
                {"mu": np.ones((2, 3))},
                include_all_required_hmc_diagnostics=True,
            )

    def test_validate_hmc_diagnostics_missing(self):
        with pytest.raises(ValueError):
            st._validate_dict_hmc_diagnostics({"divergent__": np.ones((2, 2))})

    def test_validate_hmc_diagnostics_ok(self, samples_dict):
        assert st._validate_dict_hmc_diagnostics(samples_dict) is None


# ---------------------------------------------------------------------------
# Report-rendering helpers
# ---------------------------------------------------------------------------

class TestRenderHelpers:
    def _report(self, verbosity, no_warning=False):
        return dict(
            verbosity=verbosity,
            no_warning=no_warning,
            all_good_message="ALLGOOD",
            summary=["SUM1", "SUM2"],
            detail=["DET1"],
            descriptions=["DESC1"],
        )

    def test_verbosity_0_silent(self, capsys):
        st._render_diagnostic_report(**self._report(0))
        assert capsys.readouterr().out == ""

    def test_verbosity_1_summary_only(self, capsys):
        st._render_diagnostic_report(**self._report(1))
        out = capsys.readouterr().out
        assert "SUM1" in out and "SUM2" in out
        assert "DESC1" not in out and "DET1" not in out

    def test_verbosity_2_summary_and_descriptions(self, capsys):
        st._render_diagnostic_report(**self._report(2))
        out = capsys.readouterr().out
        assert "SUM1" in out and "DESC1" in out
        assert "DET1" not in out

    def test_verbosity_3_detail_and_descriptions(self, capsys):
        st._render_diagnostic_report(**self._report(3))
        out = capsys.readouterr().out
        assert "DET1" in out and "DESC1" in out
        assert "SUM1" not in out

    def test_no_warning_prints_all_good(self, capsys):
        st._render_diagnostic_report(**self._report(2, no_warning=True))
        out = capsys.readouterr().out
        assert "ALLGOOD" in out

    def test_failed_expectands_phrase_singular(self):
        phrase = st._failed_expectands_phrase(["mu"], "an Rhat warning", "Rhat warnings")
        assert phrase == "The expectand mu triggered an Rhat warning."

    def test_failed_expectands_phrase_plural(self):
        phrase = st._failed_expectands_phrase(
            ["mu", "nu"], "an Rhat warning", "Rhat warnings"
        )
        assert phrase == "The expectands mu, nu triggered Rhat warnings."


# ---------------------------------------------------------------------------
# Low-level diagnostic computations
# ---------------------------------------------------------------------------

class TestComputeHelpers:
    def test_split_chain_even(self):
        assert st._split_chain(list(range(10))) == [list(range(5)), list(range(5, 10))]

    def test_split_chain_odd(self):
        first, second = st._split_chain(list(range(11)))
        assert len(first) == 5 and len(second) == 6

    def test_split_rhat_iid_near_one(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(size=(4, 500))
        assert 0.95 < st._compute_split_rhat(vals) < 1.1

    def test_split_rhat_constant_is_nan(self):
        assert np.isnan(st._compute_split_rhat(np.ones((4, 100))))

    def test_split_rhat_bad_is_large(self):
        rng = np.random.default_rng(0)
        vals = np.cumsum(rng.normal(size=(4, 500)), axis=1)
        assert st._compute_split_rhat(vals) > 1.1

    def test_tau_hat_iid_finite(self):
        rng = np.random.default_rng(0)
        tau = st._compute_tau_hat(rng.normal(size=2000))
        assert np.isfinite(tau)
        assert 0.5 < tau < 2.0

    def test_tau_hat_constant_is_inf(self):
        assert np.isinf(st._compute_tau_hat(np.ones(100)))

    def test_xi_hat_constant(self):
        assert st._compute_xi_hat(np.ones(100)) == -2

    def test_xi_hat_negative_returns_nan(self, capsys):
        out = st._compute_xi_hat(np.array([-1.0, 0.0, 1.0, 2.0]))
        assert np.isnan(out)

    def test_xi_hat_positive_is_float(self):
        rng = np.random.default_rng(0)
        vals = np.abs(rng.normal(size=500)) + 0.01
        assert np.isfinite(st._compute_xi_hat(vals))

    def test_tail_xi_hats_returns_pair(self):
        rng = np.random.default_rng(0)
        left, right = st._compute_tail_xi_hats(rng.normal(size=1000))
        assert np.isfinite(left) and np.isfinite(right)

    def test_tail_xi_hats_short_is_nan(self):
        left, right = st._compute_tail_xi_hats(np.arange(50.0))
        assert np.isnan(left) and np.isnan(right)

    def test_compute_ess_hat(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(size=(3, 1000))
        ess = st._compute_ess_hat(vals)
        assert ess.shape == (3,)
        assert np.all(ess > 0)

    def test_compute_ess_hat_constant_is_zero(self):
        ess = st._compute_ess_hat(np.ones((2, 100)))
        np.testing.assert_array_equal(ess, 0.0)

    def test_variance_check(self):
        assert st._compute_variance_check(np.array([1.0, 2.0, 3.0]))
        assert not st._compute_variance_check(np.ones(10))

    def test_variance_check_epsilon(self):
        vals = np.array([0.0, 1e-6])
        assert st._compute_variance_check(vals, epsilon=1e-15)
        assert not st._compute_variance_check(vals, epsilon=1.0)

    def test_efmi(self):
        rng = np.random.default_rng(0)
        energy = rng.normal(size=100)
        expected = np.sum(np.diff(energy) ** 2) / (len(energy) - 1) / np.var(energy)
        assert st._efmi(energy) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Public per-expectand diagnostic wrappers
# ---------------------------------------------------------------------------

class TestExpectandWrappers:
    def test_tau_hat_dict(self, samples_dict):
        out = st.tau_hat(samples_dict)
        assert set(out) == {"mu", "sigma"}
        assert out["mu"].shape == (4,)

    def test_tail_xi_hat_dict(self, samples_dict):
        out = st.tail_xi_hat(samples_dict)
        assert out["mu"].shape == (4, 2)

    def test_ess_hat_dict(self, samples_dict):
        out = st.ess_hat(samples_dict)
        assert out["mu"].shape == (4,)
        assert np.all(out["mu"] > 0)

    def test_split_rhat_dict(self, samples_dict):
        out = st.split_rhat(samples_dict)
        assert set(out) == {"mu", "sigma"}
        assert np.isscalar(out["mu"]) or np.ndim(out["mu"]) == 0

    def test_check_variance_dict(self, samples_dict):
        out = st.check_variance(samples_dict)
        assert out["mu"].shape == (4,)
        assert out["mu"].dtype == bool
        assert np.all(out["mu"])

    def test_omit(self, samples_dict):
        assert "sigma" not in st.tau_hat(samples_dict, omit="sigma")
        assert "mu" not in st.ess_hat(samples_dict, omit="mu")

    def test_omit_glob(self):
        d = make_samples_dict()
        d["mu_pred"] = d["mu"].copy()
        out = st.split_rhat(d, omit="*_pred")
        assert "mu_pred" not in out
        assert "mu" in out


# ---------------------------------------------------------------------------
# check_hmc_diagnostics
# ---------------------------------------------------------------------------

class TestCheckHmcDiagnostics:
    def test_clean_all_success(self, samples_dict):
        out = st.check_hmc_diagnostics(samples_dict, verbosity=0)
        assert out["divergences_success"]
        assert out["treedepth_success"]
        assert out["efmi_success"]
        assert out["average_accept_proxy_success"]
        assert out["divergences"].shape == (4,)

    def test_divergences_flagged(self, bad_samples_dict):
        out = st.check_hmc_diagnostics(bad_samples_dict, verbosity=0)
        assert not out["divergences_success"]
        assert not out["treedepth_success"]

    def test_verbosity_0_silent(self, bad_samples_dict, capsys):
        st.check_hmc_diagnostics(bad_samples_dict, verbosity=0)
        assert capsys.readouterr().out == ""

    def test_verbosity_1_summary(self, bad_samples_dict, capsys):
        st.check_hmc_diagnostics(bad_samples_dict, verbosity=1)
        out = capsys.readouterr().out
        assert "divergences" in out

    def test_clean_reports_all_good(self, samples_dict, capsys):
        st.check_hmc_diagnostics(samples_dict, verbosity=2)
        assert "consistent" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# check_expectand_diagnostics
# ---------------------------------------------------------------------------

class TestCheckExpectandDiagnostics:
    def test_output_keys(self, samples_dict):
        out = st.check_expectand_diagnostics(samples_dict, verbosity=0)
        for key in ("xi_hat", "variance", "rhat", "inc_tau_hat", "ess_hat"):
            assert key in out
            assert f"{key}_success" in out or key == "variance"
        assert "variance_success" in out

    def test_clean_all_success(self, samples_dict):
        out = st.check_expectand_diagnostics(samples_dict, verbosity=0)
        assert out["xi_hat_success"]
        assert out["variance_success"]
        assert out["rhat_success"]
        assert out["inc_tau_hat_success"]
        assert out["ess_hat_success"]

    def test_bad_flags_rhat_and_ess(self, bad_samples_dict):
        out = st.check_expectand_diagnostics(bad_samples_dict, verbosity=0)
        assert not out["rhat_success"]
        assert not out["ess_hat_success"]

    def test_zero_variance_flagged(self, samples_dict):
        samples_dict["const"] = np.zeros_like(samples_dict["mu"])
        out = st.check_expectand_diagnostics(samples_dict, verbosity=0)
        assert not out["variance_success"]

    def test_verbosity_levels(self, bad_samples_dict, capsys):
        st.check_expectand_diagnostics(bad_samples_dict, verbosity=0)
        assert capsys.readouterr().out == ""
        st.check_expectand_diagnostics(bad_samples_dict, verbosity=1)
        assert "Parameter" in capsys.readouterr().out

    def test_omit_array_entry(self):
        d = make_samples_dict()
        d["theta[1]"] = d["mu"].copy()
        d["theta[2]"] = d["sigma"].copy()
        out = st.check_expectand_diagnostics(
            d, omit_array_entry="theta[1]", verbosity=0
        )
        assert "theta[1]" not in out["xi_hat"]
        assert "theta[2]" in out["xi_hat"]


# ---------------------------------------------------------------------------
# check_all_diagnostics
# ---------------------------------------------------------------------------

class TestCheckAllDiagnostics:
    def test_clean_zero_code(self, samples_dict):
        assert st.check_all_diagnostics(samples_dict, verbosity=0) == 0

    def test_return_diagnostics_merged(self, samples_dict):
        code, diag = st.check_all_diagnostics(
            samples_dict, verbosity=0, return_diagnostics=True
        )
        assert code == 0
        assert "divergences" in diag
        assert "xi_hat" in diag
        assert "rhat_success" in diag

    def test_bad_sets_bits(self, bad_samples_dict):
        code = st.check_all_diagnostics(bad_samples_dict, verbosity=0)
        assert code & (1 << 0)  # divergence
        assert code & (1 << 6)  # rhat
        assert code & (1 << 8)  # ess

    def test_decodes_consistently(self, bad_samples_dict):
        code = st.check_all_diagnostics(bad_samples_dict, verbosity=0)
        flags = st.decode_warning_code(code, quiet=True, return_dict=True)
        assert flags["divergence"] is False
        assert flags["rhat"] is False


# ---------------------------------------------------------------------------
# decode_warning_code / parse_warning_code
# ---------------------------------------------------------------------------

class TestWarningCode:
    def test_all_bits(self, capsys):
        flags = st.decode_warning_code(0b111111111, quiet=True, return_dict=True)
        assert all(v is False for v in flags.values())

    def test_no_warnings(self, capsys):
        st.decode_warning_code(0)
        assert "No diagnostic warnings" in capsys.readouterr().out

    def test_specific_bits(self):
        flags = st.decode_warning_code(1 << 6, quiet=True, return_dict=True)
        assert flags["rhat"] is False
        assert flags["divergence"] is True

    def test_prints(self, capsys):
        st.decode_warning_code(1 << 0)
        assert "divergence warning" in capsys.readouterr().out

    def test_quiet_without_return_raises(self):
        with pytest.raises(RuntimeError):
            st.decode_warning_code(0, quiet=True)

    def test_parse_warning_code_deprecated(self):
        with pytest.raises(DeprecationWarning):
            st.parse_warning_code(0)


# ---------------------------------------------------------------------------
# Name / omit helpers
# ---------------------------------------------------------------------------

class TestNameHelpers:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("alpha", "alpha"),
            ("alpha[1]", "alpha"),
            ("alpha[1,2]", "alpha"),
            ("alpha[a]", "alpha[a]"),
            ("alpha[]", "alpha[]"),
        ],
    )
    def test_get_var_name(self, name, expected):
        assert st._get_var_name(name) == expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("alpha", "alpha"),
            ("alpha[1]", "alpha"),
            ("alpha[1,2]", "alpha"),
            ("alpha[a]", "alpha[a]"),
        ],
    )
    def test_base_name(self, name, expected):
        assert st._base_name(name) == expected

    def test_parse_omit_string_wrapped(self):
        omit, entry = st._parse_omit("mu")
        assert omit == ("mu",)
        assert entry == ()

    def test_parse_omit_pattern_wrapped(self):
        pat = re.compile("mu.*")
        omit, _ = st._parse_omit(pat)
        assert omit == (pat,)

    def test_parse_omit_validates_array_entry(self):
        with pytest.raises(ValueError):
            st._parse_omit(omit_array_entry="mu")  # no brackets

    def test_parse_omit_accepts_valid_array_entry(self):
        _, entry = st._parse_omit(omit_array_entry=["beta[1,2]", "y[3]"])
        assert entry == ("beta[1,2]", "y[3]")

    def test_screen_param_glob(self):
        assert st._screen_param("theta", "*_pred") is True
        assert st._screen_param("y_pred[1]", "*_pred") is False

    def test_screen_param_base_name(self):
        assert st._screen_param("beta[1]", "beta") is False
        assert st._screen_param("beta[1]", "gamma") is True

    def test_screen_param_regex(self):
        assert st._screen_param("y_pred[1]", re.compile("y_pred")) is False

    def test_screen_param_array_entry(self):
        assert st._screen_param("beta[1,2]", omit_array_entry="beta[1,2]") is False
        assert st._screen_param("beta[1,3]", omit_array_entry="beta[1,2]") is True


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

class TestMiscUtilities:
    def test_get_output_prefix_format(self):
        prefix = st._get_output_prefix(7)
        assert prefix.startswith("0" * 8 + "7")
        assert prefix.count("-") == 2

    def test_remove_sampling_files(self, tmp_path):
        d = tmp_path / "runset_dir"
        d.mkdir()
        (d / "a.csv").write_text("x")
        runset = types.SimpleNamespace(_output_dir=str(d))
        st._remove_sampling_files(runset)
        assert not d.exists()

    def test_remove_sampling_files_missing_ok(self, tmp_path):
        runset = types.SimpleNamespace(_output_dir=str(tmp_path / "nope"))
        st._remove_sampling_files(runset)  # should not raise

    def test_disable_logging(self):
        outside = logging.root.manager.disable
        with st.disable_logging(logging.WARNING):
            assert logging.root.manager.disable == logging.WARNING
        assert logging.root.manager.disable == outside


# ---------------------------------------------------------------------------
# Integration tests with a live CmdStan fit
# ---------------------------------------------------------------------------

class TestCmdStanIntegration:
    def test_dict_from_cmdstanpy(self, cmdstan_fit):
        d = st._dict_from_cmdstanpy(cmdstan_fit)
        assert "mu" in d
        assert "beta[1]" in d and "beta[2]" in d
        assert "divergent__" in d
        assert d["mu"].shape == (2, 200)

    def test_convert_cmdstanpy_to_dict(self, cmdstan_fit):
        d = st.convert_samples(cmdstan_fit, "dict")
        assert "mu" in d and "sigma" in d

    def test_convert_cmdstanpy_to_arviz(self, cmdstan_fit):
        tree = st.convert_samples(cmdstan_fit, "arviz")
        assert isinstance(tree, xarray.DataTree)
        assert "posterior" in tree.children

    def test_convert_cmdstanpy_to_dataframe(self, cmdstan_fit):
        df = st.convert_samples(cmdstan_fit, "polars")
        assert isinstance(df, pl.DataFrame)
        assert "chain__" in df.columns

    def test_dict_from_arviz_matches_cmdstanpy(self, cmdstan_fit):
        import arviz as az

        tree = az.from_cmdstanpy(cmdstan_fit)
        d = st._dict_from_arviz(tree)
        assert "mu" in d
        assert "beta[1]" in d and "beta[2]" in d
        assert "divergent__" in d

    def test_extract_variable_scalar_and_vector(self, cmdstan_fit):
        mu = st.extract_variable(cmdstan_fit, "mu")
        assert mu.shape == (2, 200)
        beta = st.extract_variable(cmdstan_fit, "beta")
        assert beta.shape == (2, 200, 2)

    def test_check_all_diagnostics_runs(self, cmdstan_fit):
        code, diag = st.check_all_diagnostics(
            cmdstan_fit, verbosity=0, return_diagnostics=True
        )
        assert isinstance(code, int)
        assert "mu" in diag["rhat"]
