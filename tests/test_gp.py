"""Test suite for bebi103.gp.

Correctness of the derivative kernels and covariance matrices is checked
against finite differences of the base kernels, which serve as ground
truth. Two known bugs are documented with ``xfail`` markers:

* The general-nu Matern mixed second derivative
  (``_d1_d2_matern_kernel``) is computed with an operator-precedence
  error, so ``cov_d1_d2_matern`` / ``d1_d2_matern_kernel`` are wrong for
  nu not in {1.5, 2.5}.
* The single-derivative "sym" covariance matrices force symmetry even
  though the true derivative cross-covariance is antisymmetric, so
  ``cov_d1_*`` called with ``x2=None`` (or identical arguments) returns a
  wrong lower triangle.
"""

import numpy as np
import pytest

import bebi103.gp as gp


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

ALPHA, RHO = 1.3, 0.7
H = 1e-6  # finite-difference step


def fd_d1(k, x1, x2):
    """Finite-difference derivative w.r.t. the first argument."""
    return (k(x1 + H, x2) - k(x1 - H, x2)) / (2 * H)


def fd_d1_d2(k, x1, x2):
    """Finite-difference mixed second derivative."""
    return (
        k(x1 + H, x2 + H)
        - k(x1 + H, x2 - H)
        - k(x1 - H, x2 + H)
        + k(x1 - H, x2 - H)
    ) / (4 * H * H)


@pytest.fixture
def points_1d():
    return np.array([0.0, 0.4, 1.1, 2.3, 3.0])


# ---------------------------------------------------------------------------
# _vector_to_array
# ---------------------------------------------------------------------------


def test_vector_to_array_1d():
    x = np.array([1.0, 2.0, 3.0])
    out = gp._vector_to_array(x)
    assert out.shape == (3, 1)


def test_vector_to_array_2d_passthrough():
    x = np.arange(6.0).reshape(3, 2)
    out = gp._vector_to_array(x)
    assert out.shape == (3, 2)
    assert np.array_equal(out, x)


def test_vector_to_array_accepts_list():
    out = gp._vector_to_array([1.0, 2.0])
    assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# append_sort_index
# ---------------------------------------------------------------------------


def test_append_sort_index_1d():
    Xstar = np.array([0.0, 1.0, 2.0, 3.0])
    X = np.array([1.0, 2.5])
    X_out, inds = gp.append_sort_index(Xstar, X)
    # Result is sorted and unique.
    assert np.array_equal(X_out, np.unique(X_out))
    assert np.all(np.diff(X_out) > 0)
    # inds recover X.
    assert np.allclose(X_out[inds], X)


def test_append_sort_index_index_origin():
    Xstar = np.array([0.0, 1.0, 2.0])
    X = np.array([1.0])
    _, inds0 = gp.append_sort_index(Xstar, X, index_origin=0)
    _, inds1 = gp.append_sort_index(Xstar, X, index_origin=1)
    assert np.array_equal(inds1, inds0 + 1)


def test_append_sort_index_2d():
    Xstar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    X_out, inds = gp.append_sort_index(Xstar, X)
    assert np.allclose(X_out[inds], X)


def test_append_sort_index_2d_shape_mismatch():
    Xstar = np.array([[0.0, 0.0], [1.0, 1.0]])
    X = np.array([[1.0, 1.0, 2.0]])
    with pytest.raises(RuntimeError, match="Shape mismatch"):
        gp.append_sort_index(Xstar, X)


# ---------------------------------------------------------------------------
# Base kernel values
# ---------------------------------------------------------------------------


def test_se_kernel_value():
    x1, x2 = 0.4, 1.1
    expected = ALPHA**2 * np.exp(-((x1 - x2) ** 2) / 2 / RHO**2)
    assert np.isclose(gp.se_kernel(x1, x2, ALPHA, RHO), expected)


def test_se_kernel_at_same_point_is_variance():
    assert np.isclose(gp.se_kernel(2.0, 2.0, ALPHA, RHO), ALPHA**2)


def test_exp_quad_kernel_is_se_alias():
    assert np.isclose(
        gp.exp_quad_kernel(0.3, 1.2, ALPHA, RHO),
        gp.se_kernel(0.3, 1.2, ALPHA, RHO),
    )


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.5])
def test_matern_kernel_at_same_point_is_variance(nu):
    assert np.isclose(gp.matern_kernel(1.0, 1.0, ALPHA, RHO, nu), ALPHA**2)


@pytest.mark.parametrize(
    "nu,special",
    [
        (0.5, gp._matern_1_kernel),
        (1.5, gp._matern_32_kernel),
        (2.5, gp._matern_52_kernel),
    ],
)
def test_matern_general_matches_special_case(nu, special):
    x1, x2 = np.array([0.4]), np.array([1.1])
    gen = gp._matern_kernel(x1, x2, ALPHA, RHO, nu)
    spec = special(x1, x2, ALPHA, RHO)
    assert np.isclose(gen, spec)


def test_periodic_kernel_variance_and_periodicity():
    T = 2.0
    assert np.isclose(gp.periodic_kernel(0.5, 0.5, ALPHA, RHO, T), ALPHA**2)
    # k(x, x + T) == k(x, x)
    assert np.isclose(
        gp.periodic_kernel(0.5, 0.5 + T, ALPHA, RHO, T),
        gp.periodic_kernel(0.5, 0.5, ALPHA, RHO, T),
    )


def test_linear_kernel_value():
    assert np.isclose(gp.linear_kernel(2.0, 3.0, 1.0, 2.0), 1.0**2 + 2.0**2 * 6)


def test_linear_kernel_scalar_and_vector_agree():
    v = gp.linear_kernel(np.array([2.0, 0.0]), np.array([3.0, 0.0]), 1.0, 2.0)
    assert np.isclose(v, gp.linear_kernel(2.0, 3.0, 1.0, 2.0))


def test_polynomial_kernel_value():
    assert np.isclose(gp.polynomial_kernel(2.0, 3.0, 1.0, 1.0, d=2), (1 + 6) ** 2)


# ---------------------------------------------------------------------------
# Derivative kernels vs finite differences
# ---------------------------------------------------------------------------

X1, X2 = 0.4, 1.1


def test_d1_se_kernel_matches_fd():
    k = lambda a, b: gp.se_kernel(a, b, ALPHA, RHO)
    assert np.isclose(gp.d1_se_kernel(X1, X2, ALPHA, RHO), fd_d1(k, X1, X2))


def test_d2_se_kernel_is_d1_with_swapped_args():
    assert np.isclose(
        gp.d2_se_kernel(X1, X2, ALPHA, RHO),
        gp.d1_se_kernel(X2, X1, ALPHA, RHO),
    )


def test_d1_d2_se_kernel_matches_fd():
    k = lambda a, b: gp.se_kernel(a, b, ALPHA, RHO)
    assert np.isclose(
        gp.d1_d2_se_kernel(X1, X2, ALPHA, RHO), fd_d1_d2(k, X1, X2), atol=1e-5
    )


def test_d1_exp_quad_kernel_is_se_alias():
    assert np.isclose(
        gp.d1_exp_quad_kernel(X1, X2, ALPHA, RHO),
        gp.d1_se_kernel(X1, X2, ALPHA, RHO),
    )


def test_derivative_kernels_require_scalars():
    with pytest.raises(NotImplementedError):
        gp.d1_se_kernel(np.array([1.0, 2.0]), np.array([1.0, 2.0]), ALPHA, RHO)


@pytest.mark.parametrize("nu", [1.5, 2.5])
def test_d1_matern_kernel_matches_fd(nu):
    k = lambda a, b: gp.matern_kernel(a, b, ALPHA, RHO, nu)
    assert np.isclose(
        gp.d1_matern_kernel(X1, X2, ALPHA, RHO, nu), fd_d1(k, X1, X2), atol=1e-5
    )


# The scalar `d1_d2_matern_kernel` routes through the general
# `_d1_d2_matern_kernel` for all nu, so it exercises the (now fixed) formula.
@pytest.mark.parametrize("nu", [1.5, 2.0, 2.5])
def test_d1_d2_matern_kernel_matches_fd(nu):
    k = lambda a, b: gp.matern_kernel(a, b, ALPHA, RHO, nu)
    assert np.isclose(
        gp.d1_d2_matern_kernel(X1, X2, ALPHA, RHO, nu),
        fd_d1_d2(k, X1, X2),
        atol=1e-4,
    )


@pytest.mark.parametrize("nu", [1.5, 2.0, 2.5, 3.0])
def test_d1_d2_matern_kernel_equal_points_finite(nu):
    # The x1 == x2 branch must be finite (not nan) for all nu > 1, including
    # integer nu where the old gamma(nu - 2) form evaluated to 0 * inf.
    val = gp.d1_d2_matern_kernel(0.5, 0.5, ALPHA, RHO, nu)
    assert np.isfinite(val)
    assert np.isclose(val, ALPHA**2 * nu / (nu - 1) / RHO**2)


# ---------------------------------------------------------------------------
# Covariance matrices: shape, symmetry, PSD, aliases, dispatch
# ---------------------------------------------------------------------------


def test_cov_exp_quad_symmetric_and_diagonal(points_1d):
    K = gp.cov_exp_quad(points_1d, alpha=ALPHA, rho=RHO)
    n = len(points_1d)
    assert K.shape == (n, n)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), ALPHA**2)


def test_cov_exp_quad_psd(points_1d):
    K = gp.cov_exp_quad(points_1d, alpha=ALPHA, rho=RHO)
    eigs = np.linalg.eigvalsh(K)
    assert eigs.min() > -1e-10


def test_cov_exp_quad_cross_shape(points_1d):
    Xstar = np.array([0.1, 0.2, 0.9])
    K = gp.cov_exp_quad(points_1d, Xstar, alpha=ALPHA, rho=RHO)
    assert K.shape == (len(points_1d), len(Xstar))


def test_cov_se_is_exp_quad_alias(points_1d):
    assert np.allclose(
        gp.cov_se(points_1d, alpha=ALPHA, rho=RHO),
        gp.cov_exp_quad(points_1d, alpha=ALPHA, rho=RHO),
    )


def test_cov_exp_quad_matches_cov_from_kernel(points_1d):
    Xstar = np.array([0.1, 0.9, 2.0])
    K1 = gp.cov_exp_quad(points_1d, Xstar, alpha=ALPHA, rho=RHO)
    K2 = gp.cov_from_kernel(points_1d, Xstar, gp.se_kernel, alpha=ALPHA, rho=RHO)
    assert np.allclose(K1, K2)


def test_cov_exp_quad_2d_input():
    X = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    K = gp.cov_exp_quad(X, alpha=ALPHA, rho=RHO)
    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 3.5])
def test_cov_matern_symmetric_and_diagonal(points_1d, nu):
    K = gp.cov_matern(points_1d, alpha=ALPHA, rho=RHO, nu=nu)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), ALPHA**2)


def test_cov_matern_dispatch_matches_general(points_1d):
    # The nu=2.5 fast path must agree with the generic kernel evaluation.
    K_fast = gp.cov_matern(points_1d, alpha=ALPHA, rho=RHO, nu=2.5)
    K_gen = gp.cov_from_kernel(
        points_1d, points_1d, gp.matern_kernel, alpha=ALPHA, rho=RHO, nu=2.5
    )
    assert np.allclose(K_fast, K_gen)


def test_cov_periodic_symmetric_and_diagonal(points_1d):
    K = gp.cov_periodic(points_1d, alpha=ALPHA, rho=RHO, T=2.0)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), ALPHA**2)


def test_cov_d1_se_is_exp_quad_alias(points_1d):
    Xstar = np.array([0.1, 0.9])
    assert np.allclose(
        gp.cov_d1_se(points_1d, Xstar, alpha=ALPHA, rho=RHO),
        gp.cov_d1_exp_quad(points_1d, Xstar, alpha=ALPHA, rho=RHO),
    )


def test_cov_d1_exp_quad_cross_matches_kernel(points_1d):
    Xstar = np.array([0.1, 0.9, 2.0])
    K = gp.cov_d1_exp_quad(points_1d, Xstar, alpha=ALPHA, rho=RHO)
    expected = np.array(
        [
            [gp.d1_se_kernel(a, b, ALPHA, RHO) for b in Xstar]
            for a in points_1d
        ]
    )
    assert np.allclose(K, expected)


def test_cov_d1_exp_quad_sym_matches_kernel(points_1d):
    # The single-argument path builds K[i,j] = d1_se_kernel(x_i, x_j), which is
    # antisymmetric with a zero diagonal.
    K = gp.cov_d1_exp_quad(points_1d, alpha=ALPHA, rho=RHO)
    expected = np.array(
        [
            [gp.d1_se_kernel(a, b, ALPHA, RHO) for b in points_1d]
            for a in points_1d
        ]
    )
    assert np.allclose(K, expected)
    assert np.allclose(K, -K.T)
    assert np.allclose(np.diag(K), 0.0)


def test_cov_d1_d2_exp_quad_symmetric(points_1d):
    K = gp.cov_d1_d2_exp_quad(points_1d, alpha=ALPHA, rho=RHO)
    assert K.shape == (len(points_1d), len(points_1d))
    assert np.allclose(K, K.T)


def test_cov_d1_matern_nu_requires_gt_one():
    with pytest.raises(RuntimeError, match="differentiability"):
        gp.cov_d1_matern(np.array([0.0, 1.0]), alpha=ALPHA, rho=RHO, nu=0.5)


@pytest.mark.parametrize("nu", [1.5, 2.5, 2.2])
def test_cov_d1_matern_sym_is_antisymmetric(points_1d, nu):
    # Single-argument (sym) path: the derivative cross-covariance is
    # antisymmetric with a zero diagonal.
    K = gp.cov_d1_matern(points_1d, alpha=ALPHA, rho=RHO, nu=nu)
    expected = np.array(
        [
            [gp.d1_matern_kernel(a, b, ALPHA, RHO, nu) for b in points_1d]
            for a in points_1d
        ]
    )
    assert np.allclose(K, expected)
    assert np.allclose(K, -K.T)


def _fd_d1_d2_matrix(x, nu):
    """Finite-difference mixed-second-derivative covariance matrix."""
    k = lambda a, b: gp.matern_kernel(a, b, ALPHA, RHO, nu)
    return np.array([[fd_d1_d2(k, a, b) for b in x] for a in x])


@pytest.mark.parametrize("nu", [1.5, 2.5])
def test_cov_d1_d2_matern_special_nu_matches_fd(nu):
    # nu in {1.5, 2.5} dispatch to the correct specialized closed forms.
    x = np.array([0.0, 0.4, 1.1, 2.3])
    K = gp.cov_d1_d2_matern(x, alpha=ALPHA, rho=RHO, nu=nu)
    assert np.allclose(K, _fd_d1_d2_matrix(x, nu), atol=1e-3)


def test_cov_d1_d2_matern_general_nu_matches_fd():
    # A non-half-integer nu exercises the general _d1_d2_matern_kernel path.
    x = np.array([0.0, 0.4, 1.1, 2.3])
    K = gp.cov_d1_d2_matern(x, alpha=ALPHA, rho=RHO, nu=2.2)
    assert np.allclose(K, _fd_d1_d2_matrix(x, 2.2), atol=1e-3)


# ---------------------------------------------------------------------------
# Covariance-matrix argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func", [gp.cov_exp_quad, gp.cov_matern, gp.cov_periodic]
)
@pytest.mark.parametrize("bad", [{"alpha": 0.0}, {"alpha": -1.0}, {"rho": 0.0}])
def test_cov_positive_params(func, bad):
    with pytest.raises(RuntimeError):
        func(np.array([0.0, 1.0]), **bad)


def test_cov_matern_nu_positive():
    with pytest.raises(RuntimeError, match="nu"):
        gp.cov_matern(np.array([0.0, 1.0]), nu=0.0)


def test_cov_periodic_T_positive():
    with pytest.raises(RuntimeError, match="`T`"):
        gp.cov_periodic(np.array([0.0, 1.0]), T=0.0)


# ---------------------------------------------------------------------------
# Triangular solvers
# ---------------------------------------------------------------------------


def test_lower_tri_solve():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    L = np.tril(A) + 5 * np.eye(5)  # well-conditioned lower triangular
    b = rng.standard_normal(5)
    x = gp._lower_tri_solve(L, b)
    assert np.allclose(L @ x, b)


def test_upper_tri_solve():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    U = np.triu(A) + 5 * np.eye(5)
    b = rng.standard_normal(5)
    x = gp._upper_tri_solve(U, b)
    assert np.allclose(U @ x, b)


def test_tri_solve_via_cholesky():
    # Solve a symmetric positive-definite system with the two solvers.
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 4))
    A = M @ M.T + np.eye(4)
    b = rng.standard_normal(4)
    L = np.linalg.cholesky(A)
    z = gp._lower_tri_solve(L, b)
    x = gp._upper_tri_solve(L.T, z)
    assert np.allclose(A @ x, b)


# ---------------------------------------------------------------------------
# posterior_mean_cov
# ---------------------------------------------------------------------------


@pytest.fixture
def gp_data():
    rng = np.random.default_rng(42)
    X = np.linspace(0, 3, 8)
    y = np.sin(X) + 0.01 * rng.standard_normal(8)
    return X, y


def test_posterior_mean_cov_shapes(gp_data):
    X, y = gp_data
    Xstar = np.linspace(0, 3, 11)
    m, Sigma = gp.posterior_mean_cov(X, y, Xstar, 0.1, alpha=1.0, rho=1.0)
    assert m.shape == (11,)
    assert Sigma.shape == (11, 11)
    assert np.allclose(Sigma, Sigma.T)


def test_posterior_mean_cov_interpolates(gp_data):
    X, y = gp_data
    # Near-noiseless GP evaluated at the data points recovers the data.
    m, _ = gp.posterior_mean_cov(X, y, X, 1e-4, alpha=1.0, rho=1.0)
    assert np.allclose(m, y, atol=1e-2)


@pytest.mark.parametrize(
    "kernel,hyper",
    [
        (gp.se_kernel, {"alpha": 1.0, "rho": 1.0}),
        (gp.matern_kernel, {"alpha": 1.0, "rho": 1.0, "nu": 2.5}),
        (gp.periodic_kernel, {"alpha": 1.0, "rho": 1.0, "T": 3.0}),
    ],
)
def test_posterior_mean_cov_kernels_run(gp_data, kernel, hyper):
    X, y = gp_data
    Xstar = np.linspace(0, 3, 6)
    m, Sigma = gp.posterior_mean_cov(X, y, Xstar, 0.1, kernel=kernel, **hyper)
    assert m.shape == (6,)
    assert np.all(np.isfinite(m))
    assert np.all(np.isfinite(Sigma))


def test_posterior_mean_cov_custom_kernel(gp_data):
    # A kernel not special-cased exercises the cov_from_kernel path.
    X, y = gp_data
    Xstar = np.linspace(0, 3, 5)
    m, Sigma = gp.posterior_mean_cov(
        X, y, Xstar, 0.1, kernel=gp.exp_quad_kernel, alpha=1.0, rho=1.0
    )
    # exp_quad_kernel is the SE kernel, but reaches the generic branch.
    m_se, _ = gp.posterior_mean_cov(
        X, y, Xstar, 0.1, kernel=gp.se_kernel, alpha=1.0, rho=1.0
    )
    assert np.allclose(m, m_se)


def test_posterior_heteroscedastic_sigma(gp_data):
    X, y = gp_data
    sigma = np.linspace(0.05, 0.2, len(X))
    m, Sigma = gp.posterior_mean_cov(X, y, X, sigma, alpha=1.0, rho=1.0)
    assert m.shape == X.shape
    assert np.all(np.isfinite(Sigma))


# ---------------------------------------------------------------------------
# posterior_mean_cov_deriv
# ---------------------------------------------------------------------------


def test_posterior_mean_cov_deriv_shapes(gp_data):
    X, y = gp_data
    Xstar = np.linspace(0.2, 2.8, 6)
    g, Sigma_g = gp.posterior_mean_cov_deriv(X, y, Xstar, 0.1, alpha=1.0, rho=1.0)
    assert g.shape == (6,)
    assert Sigma_g.shape == (6, 6)


def test_posterior_mean_cov_deriv_matches_fd_of_mean(gp_data):
    # The posterior mean of the derivative equals the derivative of the
    # posterior mean function.
    X, y = gp_data
    Xstar = np.array([0.7, 1.5, 2.2])
    h = 1e-4
    g, _ = gp.posterior_mean_cov_deriv(X, y, Xstar, 0.05, alpha=1.0, rho=1.0)
    mp, _ = gp.posterior_mean_cov(X, y, Xstar + h, 0.05, alpha=1.0, rho=1.0)
    mm, _ = gp.posterior_mean_cov(X, y, Xstar - h, 0.05, alpha=1.0, rho=1.0)
    assert np.allclose(g, (mp - mm) / (2 * h), atol=1e-4)


def test_posterior_mean_cov_deriv_matern_runs(gp_data):
    X, y = gp_data
    Xstar = np.linspace(0.2, 2.8, 5)
    g, Sigma_g = gp.posterior_mean_cov_deriv(
        X, y, Xstar, 0.1, kernel=gp.matern_kernel, alpha=1.0, rho=1.0, nu=2.5
    )
    assert g.shape == (5,)
    assert np.all(np.isfinite(g))


def test_posterior_mean_cov_deriv_requires_1d(gp_data):
    X, y = gp_data
    Xstar2d = np.column_stack([np.linspace(0, 3, 4), np.linspace(0, 1, 4)])
    with pytest.raises(NotImplementedError, match="1D"):
        gp.posterior_mean_cov_deriv(X, y, Xstar2d, 0.1, alpha=1.0, rho=1.0)


def test_posterior_mean_cov_deriv_rejects_bad_kernel(gp_data):
    X, y = gp_data
    Xstar = np.linspace(0, 3, 4)
    with pytest.raises(NotImplementedError):
        gp.posterior_mean_cov_deriv(
            X, y, Xstar, 0.1, kernel=gp.periodic_kernel, alpha=1.0, rho=1.0
        )


def test_posterior_mean_cov_deriv_accepts_lists(gp_data):
    # List inputs for X and Xstar must be accepted (not only ndarrays).
    X, y = gp_data
    Xstar = [0.7, 1.5, 2.2]
    g, Sigma_g = gp.posterior_mean_cov_deriv(
        list(X), y, Xstar, 0.1, alpha=1.0, rho=1.0
    )
    assert g.shape == (3,)
    assert Sigma_g.shape == (3, 3)
