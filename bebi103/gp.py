import math

import numpy as np
import numba
import scipy.special

_float_eps = 1.0e-14


def append_sort_index(Xstar, X, index_origin=0):
    """Add the entries of X to Xstar, sort the result, and find indices
    in the results where the entries in X appear. Typically, Xstar is
    an array of x-values for which a GP is to be computed and X is an
    array of x-values for which there are data.

    Parameters
    ----------
    Xstar : array, shape (n,) or (n, d)
        Array whose unique, sorted entries are returned.
    X : array, shape (m,) or (m, d)
        Array to be added to Xstar and whose entries have their indices
        returned.
    index_origin : int, default 0
        Origin of indexing. Choose `index_origin=1` for use with Stan.

    Returns
    -------
    X_out : array, 1D if X is shape (n,), 2D if X is shape (n, d)
        Sorted, unique entries of concatenation of Xstar and X.
    inds : array, shape (n,)
        Array of indices such that X_out[inds] = X.
    """
    Xstar = np.array(Xstar)
    X = np.array(X)

    if len(X.shape) == 1:
        X_out = np.unique(np.concatenate((Xstar, X)))
        inds = np.searchsorted(X_out, X)
    else:
        if X.shape[1] != Xstar.shape[1]:
            raise RuntimeError("Shape mismatch in `Xstar` and `X`.")
        X_out = np.unique(np.concatenate((Xstar, X), axis=0), axis=0)
        inds = _searchsorted_vector(X_out, X)

    return X_out, inds + index_origin


@numba.njit
def _searchsorted_vector(X_unique, X):
    """Search a 2-D array that is sorted along the row axis."""
    X_unique_inds = np.empty(len(X), dtype=np.int64)

    for i, x in enumerate(X):
        j = 0
        while j < len(X_unique) and not (x == X_unique[j]).all():
            j += 1
        if j == len(X_unique):
            raise RuntimeError("Not finding.")
        X_unique_inds[i] = j

    return X_unique_inds


def cov_exp_quad(X1, X2=None, alpha=1.0, rho=1.0):
    """
    Return covariance matrix for exponentiated quadratic (a.k.a. squared
    exponential) kernel.

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
    """Return covariance matrix for exponentiated quadratic (a.k.a. 
    squared exponential) kernel differentiated by the first variable.

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
    """
    Return covariance matrix for exponentiated quadratic (a.k.a. 
    squared exponential) kernel differentiated once by the first 
    variable and once by the second.

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


def cov_se(X1, X2=None, alpha=1.0, rho=1.0):
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
    return cov_exp_quad(X1, X2, alpha, rho)


def cov_d1_se(x1, x2=None, alpha=1.0, rho=1.0):
    """
    Return covariance matrix for squared exponential kernel
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
    return cov_d1_exp_quad(x1, x2, alpha, rho)


def cov_d1_d2_se(x, alpha=1.0, rho=1.0):
    """
    Return covariance matrix for squared exponential kernel
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
    return cov_d1_d2_exp_quad(x, alpha, rho)


def cov_matern(X1, X2=None, alpha=1.0, rho=1.0, nu=2.5):
    """
    Return covariance matrix for a Matérn kernel.

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
        if np.isclose(nu, 0.5):
            return _matern_1_covariance_matrix_sym(X1, alpha, rho)
        if np.isclose(nu, 1.5):
            return _matern_32_covariance_matrix_sym(X1, alpha, rho)
        if np.isclose(nu, 2.5):
            return _matern_52_covariance_matrix_sym(X1, alpha, rho)
        return _matern_covariance_matrix_sym(X1, alpha, rho, nu)

    if np.isclose(nu, 0.5):
        return _matern_1_covariance_matrix(X1, X2, alpha, rho)
    if np.isclose(nu, 1.5):
        return _matern_32_covariance_matrix(X1, X2, alpha, rho)
    if np.isclose(nu, 2.5):
        return _matern_52_covariance_matrix(X1, X2, alpha, rho)
    return _matern_covariance_matrix(X1, X2, alpha, rho, nu)


def cov_d1_matern(x1, x2=None, alpha=1.0, rho=1.0, nu=2.5):
    """
    Return covariance matrix for Matérn kernel
    differentiated by the first variable.

    Parameters
    ----------
    x1 : array shape (n,)
        Array of n points to compute kernel.
    x2 : array shape (m,) or None
        Array of m points to compute kernel. If None, assume m = 0 in
        output.
    alpha : float
        Marginalized standard deviation of the Matérn kernel.
    rho : float
        Length scale of the Matérn kernel.
    nu : float
        Roughness parameter of the Matérn kernel

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")
    if nu <= 1:
        raise RuntimeError("`nu` must be greater than one for differentiability.")

    if x2 is None or (len(x1) == len(x2) and np.allclose(x1, x2)):
        if nu == 1.5:
            return _d1_matern_32_covariance_matrix_sym(x1, alpha, rho)
        elif nu == 2.5:
            return _d1_matern_52_covariance_matrix_sym(x1, alpha, rho)

        return _d1_matern_covariance_matrix_sym(x1, alpha, rho, nu)

    if nu == 1.5:
        return _d1_matern_32_covariance_matrix(x1, x2, alpha, rho)
    elif nu == 2.5:
        return _d1_matern_52_covariance_matrix(x1, x2, alpha, rho)

    return _d1_matern_covariance_matrix(x1, x2, alpha, rho, nu)


def cov_d1_d2_matern(x, alpha=1.0, rho=1.0, nu=2.5):
    """Return covariance matrix for Matérn kernel
    differentiated once by the first variable and once by the second.

    Parameters
    ----------
    x : array shape (n,)
        Array of n points to compute kernel.
    alpha : float
        Marginalized standard deviation of the Matérn kernel.
    rho : float
        Length scale of the Matérn kernel.
    nu : float
        Roughness parameter of the Matérn kernel

    Returns
    -------
    output : array, shape(n + m, n + m)
    """
    if alpha <= 0:
        raise RuntimeError("`alpha` must be positive.")
    if rho <= 0:
        raise RuntimeError("`rho` must be positive.")

    if nu == 1.5:
        return _d1_d2_matern_32_covariance_matrix(x, alpha, rho)
    elif nu == 2.5:
        return _d1_d2_matern_52_covariance_matrix(x, alpha, rho)

    return _d1_d2_matern_covariance_matrix(x, alpha, rho, nu)


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
    """Return covariance matrix for specified kernel.

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
    kernel : function
        Function of the form kernel(x1, x2, **kernel_params) that
        returns as scalar value that is the kernel evaluated at points
        x1, x2.

    Returns
    -------
    output : Numpy array of what (X1.shape[0], X2.shape[0])
        Covariance matrix computed from inputted kernel.
    """
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


def linear_kernel(x1, x2, sigma_b=0.0, sigma=1.0):
    """Linear kernel. k = sigma_b ** 2 + sigma ** 2 * np.dot(x1, x2)

    Parameters
    ----------
    x1 : float or array
        Point in the space of covariates.
    x2 : float or array
        Point in the space of covariates.
    sigma_b : float, default 0.0
        Offset of linear kernel.
    sigma : float or array, default 1.0
        scale of linear kernel. If an array, must be of the same length
        as x1 and x2.

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

    return sigma_b**2 + sigma**2 * np.dot(x1, x2)


def polynomial_kernel(x1, x2, sigma_b=1.0, sigma_p=1.0, d=1):
    """Polynomial kernel: (sigma_0^2 + sigma_p^2 x1 . x2)^d

    Parameters
    ----------
    x1 : float or array
        Point in the space of covariates.
    x2 : float or array
        Point in the space of covariates.
    sigma_b : float, default 1.0
        Offset of polynomial kernel.
    sigma_p : float, default 1.0
        Scale of polynomial kernel.
    d : int, default 1
        Degree of polynomial kernel

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

    return (sigma_b**2 + sigma_p**2 * np.dot(x1, x2)) ** d


def exp_quad_kernel(x1, x2, alpha, rho):
    """Exponentiated quadratic (a.k.a. squared exponential) kernel.

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

    Notes
    -----
    .. This is an alias for `se_kernel`.
    """
    return se_kernel(x1, x2, alpha, rho)


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


def d1_exp_quad_kernel(x1, x2, alpha, rho):
    """
    Derivative of first variable of the exponentiated quadratic 
    (a.k.a. squared exponential) kernel.

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

    Notes
    -----
    .. This is an alias for `d1_se_kernel`.
    """
    return d1_se_kernel(x1, x2, alpha, rho)


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


def d2_exp_quad_kernel(x1, x2, alpha, rho):
    """
    Derivative of the exponentiated quadratic (a.k.a. squared 
    exponential) kernel with respect to the second variable.

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

    Notes
    -----
    .. This is an alias for `d2_se_kernel`.
    """
    return d2_se_kernel(x1, x2, alpha, rho)


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


def d1_d2_exp_quad_kernel(x1, x2, alpha, rho):
    """
    Mixed second derivative of exponentiated quadratic (a.k.a. 
    squared exponential) kernel.

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

    Notes
    -----
    .. This is an alias for `d2_se_kernel`.

    """
    return d1_d2_se_kernel(x1, x2, alpha, rho)


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
    nu : float
        Roughness parameter of the kernel
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
        return _matern_32_kernel(x1, x2, alpha, rho)
    if nu == 2.5:
        return _matern_52_kernel(x1, x2, alpha, rho)
    return _matern_kernel(x1, x2, alpha, rho, nu)


def d1_matern_kernel(x1, x2, alpha, rho, nu):
    """
    Derivative of the Matern kernel with respect to the first variable.

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
    nu : float
        Roughness parameter.

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

    return _d1_matern_kernel(x1, x2, alpha, rho, nu)


def d2_matern_kernel(x1, x2, alpha, rho, nu):
    """
    Derivative of the Matern kernel with respect to the second parameter.

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
    nu : float
        Roughness parameter.

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

    return _d2_matern_kernel(x1, x2, alpha, rho, nu)


def d1_d2_matern_kernel(x1, x2, alpha, rho, nu):
    """Mixed second derivative of Matern kernel.

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
    nu : float
        Roughness parameter.

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

    return _d1_d2_matern_kernel(x1, x2, alpha, rho, nu)


def periodic_kernel(x1, x2, alpha, rho, T):
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
    T : float
        Period of kernel.

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

    return _periodic_kernel(x1, x2, alpha, rho, T)


@numba.njit
def _se_kernel(x1, x2, alpha, rho):
    """Squared exponential kernel."""
    x_diff = x1 - x2
    return alpha**2 * np.exp(-np.dot(x_diff, x_diff) / 2.0 / rho**2)


@numba.njit
def _d1_se_kernel(x1, x2, alpha, rho):
    """Derivative of squared exponential kernel with respect to first
    variable."""
    x_diff = x1 - x2
    rho2 = rho**2

    return -(alpha**2) * x_diff * np.exp(-(x_diff**2) / 2.0 / rho2) / rho2


@numba.njit
def _d2_se_kernel(x1, x2, alpha, rho):
    """Derivative of squared exponential kernel with respect to second
    variable."""
    return _d1_se_kernel(x2, x1, alpha, rho)


@numba.njit
def _d1_d2_se_kernel(x1, x2, alpha, rho):
    """Derivative of squared exponential kernel with respect to both
    variables."""
    x_diff2 = (x1 - x2) ** 2
    rho2 = rho**2

    return (alpha / rho) ** 2 * np.exp(-x_diff2 / 2.0 / rho2) * (1 - x_diff2 / rho2)


def _matern_kernel(x1, x2, alpha, rho, nu):
    """Matern kernel."""
    x_diff = x1 - x2
    beta = np.sqrt(2 * nu * np.dot(x_diff, x_diff)) / rho

    # Special case where x1 = x2
    if np.abs(beta) < 1e-8:
        return alpha**2

    ret_val = alpha**2 * 2 ** (1 - nu) * beta**nu / scipy.special.gamma(nu)
    ret_val *= scipy.special.kv(nu, beta)
    return ret_val


def _d1_matern_kernel(x1, x2, alpha, rho, nu):
    """Derivative of Matern kernel with respect to first variable."""
    x_diff = x1 - x2
    beta = np.sqrt(2 * nu * np.dot(x_diff, x_diff)) / rho

    # Special case where x1 = x2
    if np.abs(beta) < 1e-8:
        return 0.0

    ret_val = -alpha ** 2 * 2 ** (1 - nu) / scipy.special.gamma(nu)
    ret_val *= beta ** (nu + 1) / x_diff * scipy.special.kv(nu - 1, beta)

    return ret_val


def _d2_matern_kernel(x1, x2, alpha, rho, nu):
    """Derivative of Matern kernel with respect to second variable."""
    return _d1_matern_kernel(x2, x1, alpha, rho, nu)


def _d1_d2_matern_kernel(x1, x2, alpha, rho, nu):
    """Derivative of Matern kernel with respect to first and second
    variables."""
    x_diff = x1 - x2
    beta = np.sqrt(2 * nu * np.dot(x_diff, x_diff)) / rho

    # Special case where x1 = x2
    if np.abs(beta) < 1e-8:
        return (
            alpha**2 * (nu - 2) * nu * scipy.special.gamma(nu - 2) 
            / scipy.special.gamma(nu) / rho**2
        )

    return (
        alpha ** 2 * 2 ** (2 - nu) * nu / scipy.special.gamma(nu) / rho ** 2
        * beta ** (nu - 1)
        * scipy.special.kv(nu - 1, beta) - beta * scipy.special.kv(nu - 2, beta)
    )


@numba.njit
def _d1_matern_32_kernel(x1, x2, alpha, rho):
    """Derivative of 3/2 Matern kernel with respect to first variable."""
    x_diff = x1 - x2
    rho2 = rho**2
    exp_arg = np.sqrt(3 * x_diff**2 / rho2)

    return -3 * alpha**2 / rho2 * x_diff * np.exp(-exp_arg)


@numba.njit
def _d2_matern_32_kernel(x1, x2, alpha, rho):
    """Derivative of 3/2 Matern kernel with respect to second variable."""
    return _d1_matern_32_kernel(x2, x1, alpha, rho)


@numba.njit
def _d1_d2_matern_32_kernel(x1, x2, alpha, rho):
    """Derivative of 3/2 Matern kernel with respect to first and second
    variables."""
    x_diff = x1 - x2
    rho2 = rho**2
    exp_arg = np.sqrt(3 * x_diff**2 / rho2)

    return 3 * alpha**2 / rho2 * (1 - exp_arg) * np.exp(-exp_arg)


@numba.njit
def _d1_matern_52_kernel(x1, x2, alpha, rho):
    """Derivative of 5/2 Matern kernel with respect to first variable."""
    x_diff = x1 - x2
    rho2 = rho**2
    exp_arg = np.sqrt(5 * x_diff**2 / rho2)

    return -5 * alpha**2 / 3 / rho2 * x_diff * (1 + exp_arg) * np.exp(-exp_arg)


@numba.njit
def _d2_matern_52_kernel(x1, x2, alpha, rho):
    """Derivative of 5/2 Matern kernel with respect to second variable."""
    return _d1_matern_52_kernel(x2, x1, alpha, rho)


@numba.njit
def _d1_d2_matern_52_kernel(x1, x2, alpha, rho):
    """Derivative of 5/2 Matern kernel with respect to first and second
    variables."""
    x_diff = x1 - x2
    x_diff2 = x_diff**2
    rho2 = rho**2
    exp_arg = np.sqrt(5 * x_diff2 / rho2)

    ret_val = 5 * alpha**2 / 3 / rho2**2 * np.exp(-exp_arg)
    ret_val *= (1 + exp_arg) * rho2 - 5 * x_diff2

    return ret_val


@numba.njit
def _matern_1_kernel(x1, x2, alpha, rho):
    """Matern kernel with nu = 1/2."""
    x_diff = x1 - x2
    return alpha**2 * np.exp(-np.sqrt(np.dot(x_diff, x_diff)) / rho)


@numba.njit
def _matern_32_kernel(x1, x2, alpha, rho):
    """Matern kernel with nu = 3/2."""
    x_diff = x1 - x2
    x_diff2 = np.dot(x_diff, x_diff)

    exp_arg = np.sqrt(3.0 * x_diff2) / rho

    prefact = 1.0 + exp_arg

    return alpha**2 * prefact * np.exp(-exp_arg)


@numba.njit
def _matern_52_kernel(x1, x2, alpha, rho):
    """Matern kernel with nu = 5/2."""
    x_diff = x1 - x2
    x_diff2 = np.dot(x_diff, x_diff)

    exp_arg = np.sqrt(5.0 * x_diff2) / rho

    prefact = 1.0 + exp_arg + 5.0 * x_diff2 / 3.0 / rho**2

    return alpha**2 * prefact * np.exp(-exp_arg)


@numba.njit
def _periodic_kernel(x1, x2, alpha, rho, T):
    """Perdioic kernel."""
    x_diff = x1 - x2
    exp_arg = 2.0 / rho**2 * np.sin(np.pi / T * np.sqrt(np.dot(x_diff, x_diff))) ** 2

    return alpha**2 * np.exp(-exp_arg)


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
    squared exponential kernel evaluated at (x, x), thereby giving
    a symmetric covariance matrix."""
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
def _matern_32_covariance_matrix(X1, X2, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 3/2."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _matern_32_kernel(X1[i, :], X2[j, :], alpha, rho)

    return K


@numba.njit
def _matern_32_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 3/2."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _matern_32_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _matern_52_covariance_matrix(X1, X2, alpha, rho):
    """Return covariance matrix for Matérn kernel with nu = 5/2."""
    n, m = X1.shape[0], X2.shape[0]
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _matern_52_kernel(X1[i, :], X2[j, :], alpha, rho)

    return K


@numba.njit
def _matern_52_covariance_matrix_sym(X, alpha, rho):
    """Return covariance matrix for Matérn kernel 5/2."""
    n = X.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _matern_52_kernel(X[i, :], X[j, :], alpha, rho)
            K[j, i] = K[i, j]

    return K


def _d1_matern_covariance_matrix(x1, x2, alpha, rho, nu):
    """Return covariance matrix for  derivative of first variable of
    Matérn kernel."""
    n, m = len(x1), len(x2)
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _d1_matern_kernel(x1[i], x2[j], alpha, rho, nu)

    return K


def _d1_matern_covariance_matrix_sym(x, alpha, rho, nu):
    """Return covariance matrix for  derivative of first variable of
    Matérn kernel evaluated at (x, x), thereby giving
    a symmetric covariance matrix."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_matern_kernel(x[i], x[j], alpha, rho, nu)
            K[j, i] = K[i, j]

    return K


def _d1_d2_matern_covariance_matrix(x, alpha, rho, nu):
    """Return covariance matrix for mixed second derivative of
    Matérn kernel."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_d2_matern_kernel(x[i], x[j], alpha, rho, nu)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _d1_matern_32_covariance_matrix(x1, x2, alpha, rho):
    """Return covariance matrix for  derivative of first variable of
    Matérn 3/2 kernel."""
    n, m = len(x1), len(x2)
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _d1_matern_32_kernel(x1[i], x2[j], alpha, rho)

    return K


@numba.njit
def _d1_matern_32_covariance_matrix_sym(x, alpha, rho):
    """Return covariance matrix for  derivative of first variable of
    Matérn 3/2 kernel evaluated at (x, x), thereby giving
    a symmetric covariance matrix."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_matern_32_kernel(x[i], x[j], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _d1_d2_matern_32_covariance_matrix(x, alpha, rho):
    """Return covariance matrix for mixed second derivative of
    Matérn 3/2 kernel."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_d2_matern_32_kernel(x[i], x[j], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _d1_matern_52_covariance_matrix(x1, x2, alpha, rho):
    """Return covariance matrix for  derivative of first variable of
    Matérn 5/2 kernel."""
    n, m = len(x1), len(x2)
    K = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = _d1_matern_52_kernel(x1[i], x2[j], alpha, rho)

    return K


@numba.njit
def _d1_matern_52_covariance_matrix_sym(x, alpha, rho):
    """Return covariance matrix for  derivative of first variable of
    Matérn 5/2 kernel evaluated at (x, x), thereby giving
    a symmetric covariance matrix."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_matern_52_kernel(x[i], x[j], alpha, rho)
            K[j, i] = K[i, j]

    return K


@numba.njit
def _d1_d2_matern_52_covariance_matrix(x, alpha, rho):
    """Return covariance matrix for mixed second derivative of
    Matérn 5/2 kernel."""
    n = len(x)
    K = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            K[i, j] = _d1_d2_matern_52_kernel(x[i], x[j], alpha, rho)
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
def _periodic_covariance_matrix_sym(X, alpha, rho, T):
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
    X, y, Xstar, sigma, kernel=se_kernel, delta=1e-8, **kernel_hyperparams
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
    """
    X = _vector_to_array(X)
    Xstar = _vector_to_array(Xstar)

    if np.isscalar(sigma):
        sigma2 = np.ones(len(X)) * sigma**2
    else:
        sigma2 = sigma**2

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
        Ky = cov_from_kernel(X, X, kernel, **kernel_hyperparams) + np.diag(sigma2)
        Kstar = cov_from_kernel(X, Xstar, kernel, **kernel_hyperparams)
        Kstarstar = cov_from_kernel(Xstar, Xstar, kernel, **kernel_hyperparams)

    return _solve_mean_cov(y, Ky, Kstar, Kstarstar, delta)


def posterior_mean_cov_deriv(
    X, y, Xstar, sigma, kernel=se_kernel, delta=1e-8, **kernel_hyperparams
):
    """
    Compute the posterior mean vector and covariance matrix for the
    derivative of a posterior Gaussian process derived from a Normal
    likelihood and Gaussian process prior.

    Parameters
    ----------
    X : 1D, shape (n,)
        Array of n data points for which observations were made. If a
        1D array, assume the points are one-dimensional. Note that
        these are NOT observations of derivatives; derivatives will be
        calculated.
    y : array, shape (n,)
        Measured data points.
    Xstar : 1D, shape (nstar,) or 2D array, shape (nstar, d)
        Array of nstar data points for posterior predictions are to be
        made. If a 1D array, assume the points are one-dimensional. If
        a 2D array, assume the points are d-dimensional.
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
    g : array, shape (nstar,)
        The derivative function of the Gaussian process posterior
        evaluated at the points given by `Xstar`.
    Sigma_g : array, shape (nstar, nstar)
        The covariance matrix for the derivative of the Gaussian process
        posterior evaluated at the points given by `Xstar`.

    Notes
    -----
    .. X1 and Xstar must be 1D and a SE kernel must be used.
    Multidimensional x-values and other kernels will hopefully be
    implemented later.
    """
    if len(X.shape) > 1 or len(Xstar.shape) > 1:
        raise NotImplementedError("`X` and `Xstar` must be 1D.")
    if kernel not in (exp_quad_kernel, se_kernel, matern_kernel):
        raise NotImplementedError(
            "`kernel` must be `exp_quad_kernel`, `se_kernel` or `matern_kernel`."
        )

    alpha = kernel_hyperparams.get("alpha", 1.0)
    rho = kernel_hyperparams.get("rho", 1.0)
    nu = kernel_hyperparams.get("nu", 2.5)

    if np.isscalar(sigma):
        sigma2 = np.ones(len(X)) * sigma**2
    else:
        sigma2 = sigma**2

    # Build covariance matrices
    if kernel == se_kernel or kernel == exp_quad_kernel:
        Ky = cov_exp_quad(X, alpha=alpha, rho=rho) + np.diag(sigma2)
        d1_Kstar = cov_d1_exp_quad(X, Xstar, alpha=alpha, rho=rho)
        d1_d2_Kstarstar = cov_d1_d2_exp_quad(Xstar, alpha=alpha, rho=rho)
    elif kernel == matern_kernel:
        Ky = cov_matern(X, alpha=alpha, rho=rho, nu=nu) + np.diag(sigma2)
        d1_Kstar = cov_d1_matern(X, Xstar, alpha=alpha, rho=rho, nu=nu)
        d1_d2_Kstarstar = cov_d1_d2_matern(Xstar, alpha=alpha, rho=rho, nu=nu)

    # Solve for gstar and Sigma_g_star
    neg_gstar, Sigma_g_star = _solve_mean_cov(y, Ky, d1_Kstar, d1_d2_Kstarstar, delta)

    return -neg_gstar, Sigma_g_star


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
