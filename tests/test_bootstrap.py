"""Test suite for bebi103.bootstrap."""

import numpy as np
import pytest

import bebi103.bootstrap as bs


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A fixed data set used across many tests.
DATA_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
DATA_2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])


@pytest.fixture
def normal_data():
    """A moderately sized normal sample with a known-ish mean/std."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0, scale=2.0, size=200)


# Module-level functions so they are picklable for the parallel MLE tests
# (multiprocessing on spawn platforms requires importable callables).
def _mle_fun(data):
    """MLE of (mean, std) for a normal distribution."""
    return np.array([np.mean(data), np.std(data)])


def _gen_fun(params, size, rng):
    """Draw a fresh normal data set parametrized by `params`."""
    return rng.normal(params[0], params[1], size=size)


# ---------------------------------------------------------------------------
# seed_rng
# ---------------------------------------------------------------------------


def test_seed_rng_runs():
    # Should not raise.
    bs.seed_rng(12345)


def test_seed_rng_reproducible():
    bs.seed_rng(7)
    reps1 = bs.draw_bs_reps(DATA_1, np.mean, size=100)

    bs.seed_rng(7)
    reps2 = bs.draw_bs_reps(DATA_1, np.mean, size=100)

    assert np.array_equal(reps1, reps2)


# ---------------------------------------------------------------------------
# draw_bs_reps — dispatched (mean/median/std) and generic paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [np.mean, np.median, np.std])
def test_draw_bs_reps_shape_dispatched(func):
    reps = bs.draw_bs_reps(DATA_1, func, size=57)
    assert reps.shape == (57,)
    assert np.all(np.isfinite(reps))


def test_draw_bs_reps_generic_func_shape():
    # np.var is not one of the special-cased functions, exercising the
    # generic Numba'd path.
    reps = bs.draw_bs_reps(DATA_1, np.var, size=33)
    assert reps.shape == (33,)
    assert np.all(reps >= 0.0)


def test_draw_bs_reps_default_size():
    reps = bs.draw_bs_reps(DATA_1, np.mean)
    assert reps.shape == (1,)


def test_draw_bs_reps_mean_unbiased():
    # The mean of many bootstrap replicates of the mean should be close
    # to the sample mean.
    bs.seed_rng(0)
    reps = bs.draw_bs_reps(DATA_1, np.mean, size=20000)
    sem = DATA_1.std() / np.sqrt(len(DATA_1))
    assert abs(reps.mean() - DATA_1.mean()) < 5 * sem


def test_draw_bs_reps_reps_bounded():
    # Every bootstrap mean must lie within the data range.
    reps = bs.draw_bs_reps(DATA_1, np.mean, size=500)
    assert reps.min() >= DATA_1.min()
    assert reps.max() <= DATA_1.max()


def test_draw_bs_reps_with_args():
    # A custom function taking an extra positional argument, exercising
    # the `args` branch.
    def weighted_mean(x, w):
        return w * np.mean(x)

    reps = bs.draw_bs_reps(DATA_1, weighted_mean, size=50, args=(2.0,))
    assert reps.shape == (50,)
    # 2 * mean of a resample is bounded by 2 * data range.
    assert reps.min() >= 2.0 * DATA_1.min()
    assert reps.max() <= 2.0 * DATA_1.max()


def test_draw_bs_reps_drops_nan():
    data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    reps = bs.draw_bs_reps(data, np.mean, size=200)
    # NaNs are dropped, so replicate means stay within the finite range.
    assert np.all(np.isfinite(reps))
    assert reps.min() >= 1.0
    assert reps.max() <= 4.0


# ---------------------------------------------------------------------------
# draw_bs_reps_pairs
# ---------------------------------------------------------------------------


def test_draw_bs_reps_pairs_shape():
    reps = bs.draw_bs_reps_pairs(DATA_1, 2.0 * DATA_1, bs.pearson_r, size=40)
    assert reps.shape == (40,)


def test_draw_bs_reps_pairs_perfect_correlation():
    # Pairs are perfectly correlated, so every pairs-bootstrap Pearson r
    # (on a resample with >= 2 distinct points) should be ~1.
    y = 3.0 * DATA_1 + 1.0
    bs.seed_rng(1)
    reps = bs.draw_bs_reps_pairs(DATA_1, y, bs.pearson_r, size=200)
    finite = reps[np.isfinite(reps)]
    assert np.allclose(finite, 1.0)


# ---------------------------------------------------------------------------
# draw_perm_reps — dispatched and generic paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func", [bs.diff_of_means, bs.studentized_diff_of_means]
)
def test_draw_perm_reps_shape_dispatched(func):
    reps = bs.draw_perm_reps(DATA_1, DATA_2, func, size=45)
    assert reps.shape == (45,)
    assert np.all(np.isfinite(reps))


def test_draw_perm_reps_generic_func_shape():
    def mean_diff(x, y):
        return np.mean(x) - np.mean(y)

    reps = bs.draw_perm_reps(DATA_1, DATA_2, mean_diff, size=30)
    assert reps.shape == (30,)


def test_draw_perm_reps_mean_centered_at_zero():
    # Under the null (label shuffling), the difference of means should be
    # centered near zero.
    bs.seed_rng(3)
    reps = bs.draw_perm_reps(DATA_1, DATA_2, bs.diff_of_means, size=20000)
    assert abs(reps.mean()) < 0.2


def test_draw_perm_reps_studentized_requires_two_entries():
    with pytest.raises(RuntimeError, match="at least two entries"):
        bs.draw_perm_reps(
            np.array([1.0]), DATA_2, bs.studentized_diff_of_means, size=10
        )


# ---------------------------------------------------------------------------
# Statistic functions
# ---------------------------------------------------------------------------


def test_diff_of_means():
    assert np.isclose(
        bs.diff_of_means(DATA_1, DATA_2), DATA_1.mean() - DATA_2.mean()
    )


def test_studentized_diff_of_means_matches_reference():
    d1, d2 = DATA_1, DATA_2
    denom = np.sqrt(
        np.var(d1) / (len(d1) - 1) + np.var(d2) / (len(d2) - 1)
    )
    expected = (d1.mean() - d2.mean()) / denom
    assert np.isclose(bs.studentized_diff_of_means(d1, d2), expected)


def test_studentized_diff_of_means_constant_is_nan():
    a = np.array([3.0, 3.0, 3.0])
    b = np.array([5.0, 5.0, 5.0])
    assert np.isnan(bs.studentized_diff_of_means(a, b))


def test_studentized_diff_of_means_requires_min_len():
    with pytest.raises(RuntimeError):
        bs.studentized_diff_of_means(np.array([1.0]), DATA_2)


def test_pearson_r_perfect_positive():
    assert np.isclose(bs.pearson_r(DATA_1, 2.0 * DATA_1 + 1.0), 1.0)


def test_pearson_r_perfect_negative():
    assert np.isclose(bs.pearson_r(DATA_1, -DATA_1), -1.0)


def test_pearson_r_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = rng.normal(size=50) + 0.5 * x
    assert np.isclose(bs.pearson_r(x, y), np.corrcoef(x, y)[0, 1])


def test_pearson_r_drops_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, np.nan, 10.0])
    # Surviving pairs are (1,2), (2,4), (5,10) — perfectly correlated.
    assert np.isclose(bs.pearson_r(x, y), 1.0)


# ---------------------------------------------------------------------------
# draw_bs_reps_mle (parametric bootstrap)
# ---------------------------------------------------------------------------


def test_draw_bs_reps_mle_shape(normal_data):
    reps = bs.draw_bs_reps_mle(
        _mle_fun, _gen_fun, normal_data, size=25, rng=np.random.default_rng(0)
    )
    assert reps.shape == (25, 2)


def test_draw_bs_reps_mle_recovers_params(normal_data):
    reps = bs.draw_bs_reps_mle(
        _mle_fun, _gen_fun, normal_data, size=500, rng=np.random.default_rng(1)
    )
    # Bootstrap MLEs should be centered near the point estimate.
    point = _mle_fun(normal_data)
    assert np.allclose(reps.mean(axis=0), point, atol=0.3)


def test_draw_bs_reps_mle_reproducible(normal_data):
    reps1 = bs.draw_bs_reps_mle(
        _mle_fun, _gen_fun, normal_data, size=20, rng=np.random.default_rng(99)
    )
    reps2 = bs.draw_bs_reps_mle(
        _mle_fun, _gen_fun, normal_data, size=20, rng=np.random.default_rng(99)
    )
    assert np.array_equal(reps1, reps2)


def test_draw_bs_reps_mle_progress_bar(normal_data):
    # Simply make sure the progress-bar code path runs without error.
    reps = bs.draw_bs_reps_mle(
        _mle_fun,
        _gen_fun,
        normal_data,
        size=5,
        progress_bar=True,
        rng=np.random.default_rng(0),
    )
    assert reps.shape == (5, 2)


def test_draw_bs_reps_mle_parallel_shape(normal_data):
    reps = bs.draw_bs_reps_mle(
        _mle_fun, _gen_fun, normal_data, size=12, n_jobs=2
    )
    assert reps.shape == (12, 2)


def test_draw_bs_reps_mle_parallel_uneven_split(normal_data):
    # size not divisible by n_jobs must still yield exactly `size` reps.
    reps = bs.draw_bs_reps_mle(
        _mle_fun, _gen_fun, normal_data, size=7, n_jobs=3
    )
    assert reps.shape == (7, 2)


def test_draw_bs_reps_mle_parallel_with_rng_raises(normal_data):
    with pytest.raises(RuntimeError, match="parallel"):
        bs.draw_bs_reps_mle(
            _mle_fun,
            _gen_fun,
            normal_data,
            size=10,
            n_jobs=2,
            rng=np.random.default_rng(0),
        )
