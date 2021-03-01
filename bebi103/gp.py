import numpy as np
import numba
import scipy.special

_float_eps = 1.0e-14


def cov_exp_quad(X1, X2=None, alpha=1.0, rho=1.0):
    """Return covariance matrix for squared exponential kernel.

    Parameters
    ----------
    X1 : 1D, shape (n,) or 2D array, shape (n, d)
        Array of n points to compute kernel. If a 1D array, assume the
        points are one-dimensional. If a 2D array, assume the points are
        d-dimensional.
    X2 : 1D, shape (m, ) or 2D array, shape (m, d) or None
        Array of m points to compute kernel. If a 1D array, assume the
        points are one-dimensional. If a 2D array, assume the points are
        d-dimensional. If None, assume m = 0 in output.
    alpha : float
        Marginalized standard deviation of the SE kernel.
    rho : float
        Length scale of the SE kernel.

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")

    X1 = _vector_to_array(X1)
    X2 = _vector_to_array(X2) if X2 is not None else None

    if X2 is None or (X1.shape == X2.shape and np.allclose(X1, X2)):
        return _se_covariance_matrix_sym(X1, alpha, rho)

    return _se_covariance_matrix(X1, X2, alpha, rho)


def cov_d1_exp_quad(x1, x2=None, alpha=1.0, rho=1.0):
    """Return covariance matrix for squared exponential kernel
    differentiated by the first variable.

    Parameters
    ----------
    x1 : array shape (n,)
        Array of n points to compute kernel.
    x2 : array shape (m,) or None
        Array of m points to compute kernel. If None, assume m = 0 in
        output.
    alpha : float
        Marginalized standard deviation of the SE kernel.
    rho : float
        Length scale of the SE kernel.

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")

    if x2 is None or (len(x1) == len(x2) and np.allclose(x1, x2)):
        return _d1_se_covariance_matrix_sym(x1, alpha, rho)

    return _d1_se_covariance_matrix(x1, x2, alpha, rho)


def cov_d1_d2_exp_quad(x, alpha=1.0, rho=1.0):
    """Return covariance matrix for squared exponential kernel
    differentiated once by the first variable and once by the second.

    Parameters
    ----------
    x : array shape (n,)
        Array of n points to compute kernel.
    alpha : float
        Marginalized standard deviation of the SE kernel.
    rho : float
        Length scale of the SE kernel.

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")

    return _d1_d2_se_covariance_matrix(x, alpha, rho)


def cov_matern(X1, X2=None, alpha=1.0, rho=1.0, nu=2.5):
    """Return covariance matrix for a Matérn kernel.

    Parameters
    ----------
    X1 : 1D, shape (n,) or 2D array, shape (n, d)
        Array of n points to compute kernel. If a 1D array, assume the
        points are one-dimensional. If a 2D array, assume the points are
        d-dimensional.
    X2 : 1D, shape (m, ) or 2D array, shape (m, d) or None
        Array of m points to compute kernel. If a 1D array, assume the
        points are one-dimensional. If a 2D array, assume the points are
        d-dimensional. If None, assume m = 0 in output.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.
    nu : float
        Smoothness parameter of the kernel.

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")
    if nu <= 0:
        raise RuntimeError("`nu` must be positive")

    X1 = _vector_to_array(X1)
    X2 = _vector_to_array(X2) if X2 is not None else None

    if X2 is None or (X1.shape == X2.shape and np.allclose(X1, X2)):
        if nu == 0.5:
            return _matern_1_covariance_matrix_sym(X1, alpha, rho)
        if nu == 1.5:
            return _matern_3_covariance_matrix_sym(X1, alpha, rho)
        if nu == 2.5:
            return _matern_5_covariance_matrix_sym(X1, alpha, rho)
        return _matern_covariance_matrix_sym(X1, alpha, rho, nu)

    if nu == 0.5:
        return _matern_1_covariance_matrix(X1, X2, alpha, rho)
    if nu == 1.5:
        return _matern_3_covariance_matrix(X1, X2, alpha, rho)
    if nu == 2.5:
        return _matern_5_covariance_matrix(X1, X2, alpha, rho)
    return _matern_covariance_matrix(X1, X2, alpha, rho, nu)


def cov_periodic(X1, X2=None, alpha=1.0, rho=1.0, T=1.0):
    """Return covariance matrix for a perdioic kernel.

    Parameters
    ----------
    X1 : 1D, shape (n,) or 2D array, shape (n, d)
        Array of n points to compute kernel. If a 1D array, assume the
        points are one-dimensional. If a 2D array, assume the points are
        d-dimensional.
    X2 : 1D, shape (m, ) or 2D array, shape (m, d) or None
        Array of m points to compute kernel. If a 1D array, assume the
        points are one-dimensional. If a 2D array, assume the points are
        d-dimensional. If None, assume m = 0 in output.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.
    T : float
        Period of the kernel.

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")
    if T <= 0:
        raise RuntimeError("`T` must be positive.")

    X1 = _vector_to_array(X1)
    X2 = _vector_to_array(X2) if X2 is not None else None

    if X2 is None or (X1.shape == X2.shape and np.allclose(X1, X2)):
        return _periodic_covariance_matrix_sym(X1, alpha, rho, T)

    return _periodic_covariance_matrix(X1, X2, alpha, rho, T)


def _vector_to_array(x):
    """Make sure input to kernel calculator two-dimensional."""
    x = np.array(x)

    if len(x.shape) == 1:
        return x.reshape((len(x), 1))

    return x


def cov_from_kernel(X1, X2, kernel, **kernel_params):
    """Return covariance matrix for specified kernel."""
    X1 = _vector_to_array(X1)
    X2 = _vector_to_array(X2) if X2 is not None else None

    if X2 is None or (X1.shape == X2.shape and np.allclose(X1, X2)):
        n = X1.shape[0]
        K = np.empty((n, n))

        for i in range(n):
            for j in range(i, n):
                K[i, j] = kernel(X1[i, :], X1[j, :], **kernel_params)
                K[j, i] = K[i, j]
    else:
        n, m = X1.shape[0], X2.shape[0]
        K = np.empty((n, m))

        for i in range(n):
            for j in range(m):
                K[i, j] = kernel(X1[i, :], X2[j, :], **kernel_params)

    return K


@numba.njit
def _se_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for squared exponential kernel."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _se_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


def se_kernel(x1, x2, alpha, rho):
    """Squared exponential kernel.

    Parameters
    ----------
    x1 : float or array
        Point in the space of covariates.
    x2 : float or array
        Point in the space of covariates.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.

    Returns
    -------
    output : float
        Value of returned by kernel evaluated at x1, x2.

    """
    # Make sure input is vectorial
    try:
        len_x1 = len(x1)
    except:
        x1 = np.array([x1])

    try:
        len_x2 = len(x2)
    except:
        x2 = np.array([x2])

    return _se_kernel(x1, x2, alpha, rho)


def d1_se_kernel(x1, x2, alpha, rho):
    """Derivative of first variable of squared exponential kernel.

    Parameters
    ----------
    x1 : float
        Point in the space of covariates.
    x2 : float
        Point in the space of covariates.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.

    Returns
    -------
    output : float
        Value of returned by kernel evaluated at x1, x2.

    """
    # Make sure input is float
    if not np.isscalar(x1) or not np.isscalar(x2):
        raise NotImplementedError(
            "Derivatives of kernels only allowed for scalar variables."
        )

    return _d1_se_kernel(x1, x2, alpha, rho)


def d2_se_kernel(x1, x2, alpha, rho):
    """Derivative of second variable of squared exponential kernel.

    Parameters
    ----------
    x1 : float
        Point in the space of covariates.
    x2 : float
        Point in the space of covariates.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.

    Returns
    -------
    output : float
        Value of returned by kernel evaluated at x1, x2.

    """
    # Make sure input is float
    if not np.isscalar(x1) or not np.isscalar(x2):
        raise NotImplementedError(
            "Derivatives of kernels only allowed for scalar variables."
        )

    return _d2_se_kernel(x1, x2, alpha, rho)


def d1_d2_se_kernel(x1, x2, alpha, rho):
    """Mixed second derivative of squared exponential kernel.

    Parameters
    ----------
    x1 : float
        Point in the space of covariates.
    x2 : float
        Point in the space of covariates.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.

    Returns
    -------
    output : float
        Value of returned by kernel evaluated at x1, x2.

    """
    # Make sure input is float
    if not np.isscalar(x1) or not np.isscalar(x2):
        raise NotImplementedError(
            "Derivatives of kernels only allowed for scalar variables."
        )

    return _d1_d2_se_kernel(x1, x2, alpha, rho)


def matern_kernel(x1, x2, alpha, rho, nu):
    """Matern kernel.

    Parameters
    ----------
    x1 : float or array
        Point in the space of covariates.
    x2 : float or array
        Point in the space of covariates.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.

    Returns
    -------
    output : float
        Value of returned by kernel evaluated at x1, x2.

    """
    # Make sure input is vectorial
    try:
        len_x1 = len(x1)
    except:
        x1 = np.array([x1])

    try:
        len_x2 = len(x2)
    except:
        x2 = np.array([x2])

    if nu == 0.5:
        return _matern_1_kernel(x1, x2, alpha, rho)
    if nu == 1.5:
        return _matern_3_kernel(x1, x2, alpha, rho)
    if nu == 2.5:
        return _matern_5_kernel(x1, x2, alpha, rho)
    return _matern_kernel(x1, x2, alpha, rho, nu)


def periodic_kernel(x1, x2, alpha, rho):
    """Periodic kernel.

    Parameters
    ----------
    x1 : float or array
        Point in the space of covariates.
    x2 : float or array
        Point in the space of covariates.
    alpha : float
        Marginalized standard deviation of the kernel.
    rho : float
        Length scale of the kernel.

    Returns
    -------
    output : float
        Value of returned by kernel evaluated at x1, x2.

    """
    # Make sure input is vectorial
    try:
        len_x1 = len(x1)
    except:
        x1 = np.array([x1])

    try:
        len_x2 = len(x2)
    except:
        x2 = np.array([x2])

    return _periodic_kernel(x1, x2, alpha, rho)


@numba.njit
def _se_kernel(x1, x2, alpha, rho):
    """Squared exponential kernel."""
    x_diff = x1 - x2
    return alpha ** 2 * np.exp(-np.dot(x_diff, x_diff) / 2.0 / rho ** 2)


@numba.njit
def _d1_se_kernel(x1, x2, alpha, rho):
    """Derivative of first variable of squared exponential kernel."""
    x_diff = x1 - x2
    rho2 = rho ** 2

    return -alpha ** 2 * x_diff * np.exp(-x_diff ** 2 / 2.0 / rho2) / rho2


@numba.njit
def _d2_se_kernel(x1, x2, alpha, rho):
    """Derivative of first variable of squared exponential kernel."""
    return _d1_se_kernel(x2, x1, alpha, rho)


@numba.njit
def _d1_d2_se_kernel(x1, x2, alpha, rho):
    """Derivative of first variable of squared exponential kernel."""
    x_diff2 = (x1 - x2)**2
    rho2 = rho**2

    return (alpha / rho)**2 * np.exp(-x_diff2 / 2.0 / rho2) * (1 - x_diff2 / rho2)


def _matern_kernel(x1, x2, alpha, rho, nu):
    """Matern kernel."""
    x_diff = x1 - x2
    beta = np.sqrt(2 * nu * np.dot(x_diff, x_diff)) / rho

    # Special case where x1 = x2
    if beta == 0:
        return alpha ** 2

    ret_val = alpha ** 2 * 2 ** (1 - nu) * beta ** nu / scipy.special.gamma(nu)
    ret_val *= scipy.special.kv(nu, beta)
    return ret_val


@numba.njit
def _matern_1_kernel(x1, x2, alpha, rho):
    """Matern kernel with nu = 1/2."""
    x_diff = x1 - x2
    return alpha ** 2 * np.exp(-np.sqrt(np.dot(x_diff, x_diff)) / rho)


@numba.njit
def _matern_3_kernel(x1, x2, alpha, rho):
    """Matern kernel with nu = 3/2."""
    x_diff = x1 - x2
    x_diff2 = np.dot(x_diff, x_diff)

    exp_arg = np.sqrt(3.0 * x_diff2) / rho

    prefact = 1.0 + exp_arg

    return alpha ** 2 * prefact * np.exp(-exp_arg)


@numba.njit
def _matern_5_kernel(x1, x2, alpha, rho):
    """Matern kernel with nu = 5/2."""
    x_diff = x1 - x2
    x_diff2 = np.dot(x_diff, x_diff)

    exp_arg = np.sqrt(5.0 * x_diff2) / rho

    prefact = 1.0 + exp_arg + 5.0 * x_diff2 / 3.0 / rho ** 2

    return alpha ** 2 * prefact * np.exp(-exp_arg)


@numba.njit
def _periodic_kernel(x1, x2, alpha, rho, T):
    """Perdioic kernel."""
    x_diff = x1 - x2
    exp_arg = 2.0 / rho ** 2 * sin(np.pi / T * np.sqrt(np.dot(x_diff, x_diff)))

    return alpha ** 2 * np.exp(-exp_arg)


@numba.njit
def _se_covariance_matrix(X1, X2, alpha, rho):
    """Return covariance matrix for squared exponential kernel."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _se_kernel(X1[i, :], X2[j, :], alpha, rho)

    return K


@numba.njit
def _se_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for squared exponential kernel."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _se_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _d1_se_covariance_matrix(x1, x2, alpha, rho):
    """Return covariance matrix for  derivative of first variable of
    squared exponential kernel."""
    n, m = len(x1), len(x2)
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _d1_se_kernel(x1[i], x2[j], alpha, rho)

    return K


@numba.njit
def _d1_se_covariance_matrix_sym(x, alpha, rho):
    """Return covariance matrix for  derivative of first variable of
    squared exponential kernel."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_se_kernel(x[i], x[j], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _d1_d2_se_covariance_matrix(x, alpha, rho):
    """Return covariance matrix for mixed second derivative of
    squared exponential kernel."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_d2_se_kernel(x[i], x[j], alpha, rho)
            K[j, i] = K[i, j]

    return K


def _matern_covariance_matrix(X1, X2, alpha, rho, nu):
    """Return covariance matrix for Matérn kernel."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _matern_kernel(X1[i, :], X2[j, :], alpha, rho, nu)

    return K


def _matern_covariance_matrix_sym(X, alpha, rho, nu):
    """Return covariance matrix for Matérn kernel."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _matern_kernel(X[i, :], X[j, :], alpha, rho, nu)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _matern_1_covariance_matrix(X1, X2, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 1/2."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _matern_1_kernel(X1[i, :], X2[j, :], alpha, rho)

    return K


@numba.njit
def _matern_1_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 1/2."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _matern_1_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _matern_3_covariance_matrix(X1, X2, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 3/2."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _matern_3_kernel(X1[i, :], X2[j, :], alpha, rho)

    return K


@numba.njit
def _matern_3_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 3/2."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _matern_3_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _matern_5_covariance_matrix(X1, X2, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 5/2."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _matern_5_kernel(X1[i, :], X2[j, :], alpha, rho)

    return K


@numba.njit
def _matern_5_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for Matérn kernel 5/2."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _matern_5_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _periodic_covariance_matrix(X1, X2, alpha, rho, T):
    """Return covariance matrix for squared exponential kernel."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _periodic_kernel(X1[i, :], X2[j, :], alpha, rho, T)

    return K


@numba.njit
def _periodic_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for squared exponential kernel."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _periodic_kernel(X[i, :], X[j, :], alpha, rho, T)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _lower_tri_solve(L, b):
    """
    Solves the lower triangular system Lx = b.
    Uses column-based forward substitution, outlined in algorithm
    3.1.3 of Golub and van Loan.
    Parameters
    ----------
    L : ndarray
        Square lower triangulatar matrix (including diagonal)
    b : ndarray, shape L.shape[0]
        Right hand side of Lx = b equation being solved.
    Returns
    -------
    x : ndarray
        Solution to Lx = b.
    """
    n = L.shape[0]

    # Solve Lx = b.
    x = np.copy(b)
    for j in range(n - 1):
        if abs(L[j, j]) > _float_eps:
            x[j] /= L[j, j]
            for i in range(j + 1, n):
                x[i] -= x[j] * L[i, j]
        else:
            x[j] = 0.0

    if n > 0:
        if abs(L[n - 1, n - 1]) > _float_eps:
            x[n - 1] /= L[n - 1, n - 1]
        else:
            x[n - 1] = 0.0

    return x


@numba.njit
def _upper_tri_solve(U, b):
    """
    Solves the lower triangular system Ux = b.
    Uses column-based forward substitution, outlined in algorithm
    3.1.4 of Golub and van Loan.
    Parameters
    ----------
    U: ndarray
        Square upper triangulatar matrix (including diagonal)
    b : ndarray, shape L.shape[0]
        Right hand side of Ux = b equation being solved.
    Returns
    -------
    x : ndarray
        Solution to Ux = b.
    """
    n = U.shape[0]

    # Solve Ux = b by back substitution.
    x = np.copy(b)
    for j in range(n - 1, 0, -1):
        if abs(U[j, j]) > _float_eps:
            x[j] /= U[j, j]
            for i in range(0, j):
                x[i] -= x[j] * U[i, j]
        else:
            x[j] = 0.0

    if n > 0:
        if abs(U[0, 0]) > _float_eps:
            x[0] /= U[0, 0]
        else:
            x[0] = 0.0

    return x


def posterior_mean_cov(
    X,
    y,
    Xstar,
    sigma,
    kernel=se_kernel,
    include_deriv=False,
    delta=1e-8,
    **kernel_hyperparams
):
    """
    Compute the posterior mean vector and covariance matrix for a
    posterior Gaussian process derived from a Normal likelihood and
    Gaussian process prior.

    Parameters
    ----------
    X : 1D, shape (n,) or 2D array, shape (n, d)
        Array of n data points for which observations were made. If a 1D
        array, assume the points are one-dimensional. If a 2D array,
        assume the points are d-dimensional.
    y : array, shape (n,)
        Measured data points.
    Xstar : 1D, shape (nstar,) or 2D array, shape (nstar, d)
        Array of nstar data points for posterior predictions are to be
        made. If a 1D array, assume the points are one-dimensional. If a
        2D array, assume the points are d-dimensional.
    sigma : float or array, shape (n,)
        Standard deviation for Normal likelihood. If a float, assumed to
        be homoscedastic for all points.
    kernel : function, default se_kernel
        Kernel defining the Gaussian process. Must have call signature
        kernel(x1, x2, **kernel_hyperparams).
    include_deriv : bool, default False
        If True, include first derivatives in mean vectors and
        covariances. If True, `X` and `Xstar` must both be 1D because
        multivariate gradients are not implemented.
    delta : float, default 1e-8
        Small number, used to add to the diagonal of covariance matrices
        to ensure numerical positive definiteness.
    **kernel_hyperparams : kwargs
        All additional kwargs are sent as kwargs to the `kernel`
        function.

    Returns
    -------
    m : array, shape (nstar,)
        The mean function of the Gaussian process posterior evaluated at
        the points given by `Xstar`.
    Sigma : array, shape (nstar, nstar) or (2*nstar, 2*nstar)
        Covariance matrix of the Gaussian process posterior evaluated at
        the points given by `Xstar`.
    g : array, shape (nstar,)
        The derivative function of the Gaussian process posterior
        evaluated at the points given by `Xstar`. Only returned if
        `include_deriv` is True.
    Sigma_g : array, shape (nstar, nstar)
        The covariance matrix for the derivative of the Gaussian process
        posterior evaluated at the points given by `Xstar`. Only
        returned if `include_deriv` is True.

    Notes
    -----
    .. If include_deriv is True, X1 and Xstar must be 1D and a SE kernel
    must be used.
    """
    if include_deriv == 1:
        if len(X.shape) > 1 or len(Xstar.shape) > 1:
            raise NotImplementedError(
                "If `deriv` is True, then `X` and `Xstar` must be 1D."
            )
        if kernel != se_kernel:
            raise NotImplementedError(
                "If `deriv` is True, then `kernel` must be the default `se_kernel`."
            )

        alpha = kernel_hyperparams.get("alpha", 1.0)
        rho = kernel_hyperparams.get("rho", 1.0)

        return _posterior_mean_cov_deriv(X, y, Xstar, sigma, alpha, rho, delta)

    X = _vector_to_array(X)
    Xstar = _vector_to_array(Xstar)

    if np.isscalar(sigma):
        sigma2 = np.ones(len(X)) * sigma ** 2
    else:
        sigma2 = sigma ** 2

    # Build covariance matrices
    if kernel == se_kernel:
        Ky = cov_exp_quad(X, **kernel_hyperparams) + np.diag(sigma2)
        Kstar = cov_exp_quad(X, Xstar, **kernel_hyperparams)
        Kstarstar = cov_exp_quad(Xstar, Xstar, **kernel_hyperparams)
    elif kernel == matern_kernel:
        Ky = cov_matern(X, **kernel_hyperparams) + np.diag(sigma2)
        Kstar = cov_matern(X, Xstar, **kernel_hyperparams)
        Kstarstar = cov_matern(Xstar, Xstar, **kernel_hyperparams)
    elif kernel == periodic_kernel:
        Ky = cov_periodic(X, **kernel_hyperparams) + np.diag(sigma2)
        Kstar = cov_periodic(X, Xstar, **kernel_hyperparams)
        Kstarstar = cov_periodic(Xstar, Xstar, **kernel_hyperparams)
    else:
        Ky = cov_from_kernel(X, Xstar, kernel, **kernel_params)
        Kstar = cov_from_kernel(X, Xstar, **kernel_hyperparams)
        Kstarstar = cov_from_kernel(Xstar, Xstar, **kernel_hyperparams)

    return _solve_mean_cov(y, Ky, Kstar, Kstarstar, delta)


def _posterior_mean_cov_deriv(x, y, xstar, sigma, alpha, rho, delta):
    if np.isscalar(sigma):
        sigma2 = np.ones(len(x)) * sigma ** 2
    else:
        sigma2 = sigma ** 2

    Ky = cov_exp_quad(x, alpha=alpha, rho=rho) + np.diag(sigma2)
    Kstar = cov_exp_quad(x, xstar, alpha=alpha, rho=rho)
    Kstarstar = cov_exp_quad(xstar, xstar, alpha=alpha, rho=rho)

    d1_Kstar = cov_d1_exp_quad(x, xstar, alpha=alpha, rho=rho)
    d1_d2_Kstarstar = cov_d1_d2_exp_quad(xstar, alpha=alpha, rho=rho)

    # Solve for mstar and Sigmastar
    mstar, Sigmastar = _solve_mean_cov(y, Ky, Kstar, Kstarstar, delta)

    # Solve for gstar and Sigma_g_star
    neg_gstar, Sigma_g_star = _solve_mean_cov(y, Ky, d1_Kstar, d1_d2_Kstarstar, delta)

    return mstar, Sigmastar, -neg_gstar, Sigma_g_star


@numba.njit
def _solve_mean_cov(y, Ky, Kstar, Kstarstar, delta):
    # Solve for m_star
    L = np.linalg.cholesky(Ky)
    z = _lower_tri_solve(L, y)
    xi = _upper_tri_solve(L.transpose(), z)
    mstar = np.dot(Kstar.transpose(), xi)

    # Solve for Sigmastar
    Xi = np.empty_like(Kstar)
    for j in range(Xi.shape[1]):
        z = _lower_tri_solve(L, Kstar[:, j])
        Xi[:, j] = _upper_tri_solve(L.transpose(), z)
    Sigmastar = (
        Kstarstar
        - np.dot(Kstar.transpose(), Xi)
        + np.diag(np.ones(len(Kstarstar)) * delta)
    )

    return mstar, Sigmastar

