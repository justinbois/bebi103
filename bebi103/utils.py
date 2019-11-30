import multiprocessing
import warnings

import tqdm

import numpy as np
import numba


@numba.njit
def ecdf(x, data):
    """
    Compute the values of the formal ECDF generated from `data` at
    points `x`. I.e., if F is the ECDF, return F(x).

    Parameters
    ----------
    x : array_like
        Positions at which the formal ECDF is to be evaluated.
    data : array_like
        Data set to use to generate the ECDF.

    Returns
    -------
    output : float or ndarray
        Value of the ECDF at `x`.
    """
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]


def draw_bs_reps(data, func, size=1, args=()):
    """
    Generate bootstrap replicates out of `data` using `func`.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data.
    func : function
        Function, with call signature `func(data, *args)` to compute
        replicate statistic from resampled `data`.
    size : int, default 1
        Number of bootstrap replicates to generate.
    args : tuple, default ()
        Arguments to be passed to `func`.

    Returns
    -------
    output : ndarray
        Bootstrap replicates computed from `data` using `func`.

    Notes
    -----
    .. nan values are ignored.
    """
    data = _convert_data(data)

    if args == ():
        if func == np.mean:
            return _draw_bs_reps_mean(data, size=size)
        elif func == np.median:
            return _draw_bs_reps_median(data, size=size)
        elif func == np.std:
            return _draw_bs_reps_std(data, size=size)

    # Make Numba'd function
    f = _make_one_arg_numba_func(func)

    @numba.jit
    def _draw_bs_reps(data):
        # Set up output array
        bs_reps = np.empty(size)

        # Draw replicates
        n = len(data)
        for i in range(size):
            bs_reps[i] = f(np.random.choice(data, size=n), args)

        return bs_reps

    return _draw_bs_reps(data)


@numba.jit(nopython=True)
def _draw_bs_reps_mean(data, size=1):
    """
    Generate bootstrap replicates of the mean out of `data`.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data.
    size : int, default 1
        Number of bootstrap replicates to generate.

    Returns
    -------
    output : float
        Bootstrap replicates of the mean computed from `data`.
    """
    # Set up output array
    bs_reps = np.empty(size)

    # Draw replicates
    n = len(data)
    for i in range(size):
        bs_reps[i] = np.mean(np.random.choice(data, size=n))

    return bs_reps


@numba.jit(nopython=True)
def _draw_bs_reps_median(data, size=1):
    """
    Generate bootstrap replicates of the median out of `data`.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data.
    size : int, default 1
        Number of bootstrap replicates to generate.

    Returns
    -------
    output : float
        Bootstrap replicates of the median computed from `data`.
    """
    # Set up output array
    bs_reps = np.empty(size)

    # Draw replicates
    n = len(data)
    for i in range(size):
        bs_reps[i] = np.median(np.random.choice(data, size=n))

    return bs_reps


@numba.jit(nopython=True)
def _draw_bs_reps_std(data, ddof=0, size=1):
    """
    Generate bootstrap replicates of the median out of `data`.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data.
    ddof : int
        Delta degrees of freedom. Divisor in standard deviation
        calculation is `len(data) - ddof`.
    size : int, default 1
        Number of bootstrap replicates to generate.

    Returns
    -------
    output : float
        Bootstrap replicates of the median computed from `data`.
    """
    # Set up output array
    bs_reps = np.empty(size)

    # Draw replicates
    n = len(data)
    for i in range(size):
        bs_reps[i] = np.std(np.random.choice(data, size=n))

    if ddof > 0:
        return bs_reps * np.sqrt(n / (n - ddof))

    return bs_reps


def draw_bs_pairs(x, y, func, size=1, args=()):
    """
    Perform pairs bootstrap for single statistic.

    Parameters
    ----------
    x : array_like
        x-values of data.
    y : array_like
        y-values of data.
    func : function
        Function, with call signature `func(x, y, *args)` to compute
        replicate statistic from pairs bootstrap sample. It must return
        a single, scalar value.
    size : int, default 1
        Number of pairs bootstrap replicates to draw.
    args : tuple, default ()
        Arguments to be passed to `func`.

    Returns
    -------
    output : ndarray
        Bootstrap replicates.
    """
    x, y = _convert_two_data(x, y, min_len=1)

    # Make Numba'd function
    f = _make_two_arg_numba_func(func)

    n = len(x)

    @numba.jit
    def _draw_bs_pairs(x, y):
        # Set up array of indices to sample from
        inds = np.arange(n)

        # Initialize replicates
        bs_replicates = np.empty(size)

        # Generate replicates
        for i in range(size):
            bs_inds = np.random.choice(inds, n)
            bs_x, bs_y = x[bs_inds], y[bs_inds]
            bs_replicates[i] = f(bs_x, bs_y, args)

        return bs_replicates

    return _draw_bs_pairs(x, y)


def permutation_sample(data_1, data_2):
    """
    Generate a permutation sample from two data sets. Specifically,
    concatenate `data_1` and `data_2`, scramble the order of the
    concatenated array, and then return the first len(data_1) entries
    and the last len(data_2) entries.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    out_1 : ndarray, same shape as `data_1`
        Permutation sample corresponding to `data_1`.
    out_2 : ndarray, same shape as `data_2`
        Permutation sample corresponding to `data_2`.
    """
    data_1 = _convert_data(data_1)
    data_2 = _convert_data(data_2)

    return _permutation_sample(data_1, data_2)


@numba.jit(nopython=True)
def _permutation_sample(data_1, data_2):
    """
    Generate a permutation sample from two data sets. Specifically,
    concatenate `data_1` and `data_2`, scramble the order of the
    concatenated array, and then return the first len(data_1) entries
    and the last len(data_2) entries.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    out_1 : ndarray, same shape as `data_1`
        Permutation sample corresponding to `data_1`.
    out_2 : ndarray, same shape as `data_2`
        Permutation sample corresponding to `data_2`.
    """
    x = np.concatenate((data_1, data_2))
    np.random.shuffle(x)
    return x[: len(data_1)], x[len(data_1) :]


def draw_perm_reps(data_1, data_2, func, size=1, args=()):
    """
    Generate permutation replicates of `func` from `data_1` and
    `data_2`

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.
    func : function
        Function, with call signature `func(x, y, *args)` to compute
        replicate statistic from permutation sample. It must return
        a single, scalar value.
    size : int, default 1
        Number of pairs bootstrap replicates to draw.
    args : tuple, default ()
        Arguments to be passed to `func`.

    Returns
    -------
    output : ndarray
        Permutation replicates.
    """
    # Convert to Numpy arrays
    data_1 = _convert_data(data_1)
    data_2 = _convert_data(data_2)

    if args == ():
        if func == diff_of_means:
            return _draw_perm_reps_diff_of_means(data_1, data_2, size=size)
        elif func == studentized_diff_of_means:
            if len(data_1) == 1 or len(data_2) == 1:
                raise RuntimeError("Data sets must have at least two entries")
            return _draw_perm_reps_studentized_diff_of_means(data_1, data_2, size=size)

    # Make a Numba'd function for drawing reps.
    f = _make_two_arg_numba_func(func)

    @numba.jit
    def _draw_perm_reps(data_1, data_2):
        n1 = len(data_1)
        x = np.concatenate((data_1, data_2))

        perm_reps = np.empty(size)
        for i in range(size):
            np.random.shuffle(x)
            perm_reps[i] = f(x[:n1], x[n1:], args)

        return perm_reps

    return _draw_perm_reps(data_1, data_2)


@numba.jit(nopython=True)
def _draw_perm_reps_diff_of_means(data_1, data_2, size=1):
    """
    Generate permutation replicates of difference of means from
    `data_1` and `data_2`

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.
    size : int, default 1
        Number of pairs bootstrap replicates to draw.

    Returns
    -------
    output : ndarray
        Permutation replicates.
    """
    n1 = len(data_1)
    x = np.concatenate((data_1, data_2))

    perm_reps = np.empty(size)
    for i in range(size):
        np.random.shuffle(x)
        perm_reps[i] = _diff_of_means(x[:n1], x[n1:])

    return perm_reps


@numba.jit(nopython=True)
def _draw_perm_reps_studentized_diff_of_means(data_1, data_2, size=1):
    """
    Generate permutation replicates of Studentized difference
    of means from  `data_1` and `data_2`

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.
    size : int, default 1
        Number of pairs bootstrap replicates to draw.

    Returns
    -------
    output : ndarray
        Permutation replicates.
    """
    n1 = len(data_1)
    x = np.concatenate((data_1, data_2))

    perm_reps = np.empty(size)
    for i in range(size):
        np.random.shuffle(x)
        perm_reps[i] = _studentized_diff_of_means(x[:n1], x[n1:])

    return perm_reps


def diff_of_means(data_1, data_2):
    """
    Difference in means of two arrays.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    output : float
        np.mean(data_1) - np.mean(data_2)
    """
    data_1 = _convert_data(data_1)
    data_2 = _convert_data(data_2)

    return _diff_of_means(data_1, data_2)


@numba.jit(nopython=True)
def _diff_of_means(data_1, data_2):
    """
    Difference in means of two arrays.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    output : float
        np.mean(data_1) - np.mean(data_2)
    """
    return np.mean(data_1) - np.mean(data_2)


def studentized_diff_of_means(data_1, data_2):
    """
    Studentized difference in means of two arrays.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    output : float
        Studentized difference of means.

    Notes
    -----
    .. If the variance of both `data_1` and `data_2` is zero, returns
       np.nan.
    """
    data_1 = _convert_data(data_1, min_len=2)
    data_2 = _convert_data(data_2, min_len=2)

    return _studentized_diff_of_means(data_1, data_2)


@numba.jit(nopython=True)
def _studentized_diff_of_means(data_1, data_2):
    """
    Studentized difference in means of two arrays.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    output : float
        Studentized difference of means.

    Notes
    -----
    .. If the variance of both `data_1` and `data_2` is zero, returns
       np.nan.
    """
    if _allequal(data_1) and _allequal(data_2):
        return np.nan

    denom = np.sqrt(
        np.var(data_1) / (len(data_1) - 1) + np.var(data_2) / (len(data_2) - 1)
    )

    return (np.mean(data_1) - np.mean(data_2)) / denom


def pearson_r(data_1, data_2):
    """
    Compute the Pearson correlation coefficient between two samples.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    output : float
        The Pearson correlation coefficient between `data_1`
        and `data_2`.

    Notes
    -----
    .. Only entries where both `data_1` and `data_2` are not NaN are
       used.
    .. If the variance of `data_1` or `data_2` is zero, return NaN.
    """
    x, y = _convert_two_data(data_1, data_2, inf_ok=False, min_len=2)
    return _pearson_r(x, y)


@numba.jit(nopython=True)
def _pearson_r(x, y):
    """
    Compute the Pearson correlation coefficient between two samples.

    Parameters
    ----------
    data_1 : array_like
        One-dimensional array of data.
    data_2 : array_like
        One-dimensional array of data.

    Returns
    -------
    output : float
        The Pearson correlation coefficient between `data_1`
        and `data_2`.

    Notes
    -----
    .. Only entries where both `data_1` and `data_2` are not NaN are
       used.
    .. If the variance of `data_1` or `data_2` is zero, return NaN.
    """
    if _allequal(x) or _allequal(y):
        return np.nan

    return (np.mean(x * y) - np.mean(x) * np.mean(y)) / np.std(x) / np.std(y)


def _draw_bs_reps_mle(
    mle_fun,
    gen_fun,
    data,
    mle_args=(),
    gen_args=(),
    size=1,
    progress_bar=False,
    rg=None,
):
    """Draw parametric bootstrap replicates of maximum likelihood
    estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *mle_args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(params, *gen_args, size, rg)`. Note
        that `size` in as an argument in this function relates to the
        number of data you will generate, which is always equal to
        len(data). This is not the same as the `size` argument of
        `_draw_bs_reps_mle()`, which is the number of bootstrap
        replicates you wish to draw.
    data : Numpy array, possibly multidimensional
        Array of measurements. The first index should index repeat of
        experiment. E.g., if the data consist of n (x, y) pairs, `data`
        should have shape (n, 2).
    mle_args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    gen_args : tuple, default ()
        Arguments to be passed to `gen_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.
    rg : numpy.random.Generator instance, default None
        RNG to be used in bootstrapping. If None, the default
        Numpy RNG is used with a fresh seed based on the clock.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if rg is None:
        rg = np.random.default_rng()

    params = mle_fun(data, *mle_args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [
            mle_fun(gen_fun(params, *gen_args, size=len(data), rg=rg), *mle_args)
            for _ in iterator
        ]
    )


def draw_bs_reps_mle(
    mle_fun,
    gen_fun,
    data,
    mle_args=(),
    gen_args=(),
    size=1,
    n_jobs=1,
    progress_bar=False,
    rg=None,
):
    """Draw bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *mle_args) that computes
        a MLE for the parameters.
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(params, *gen_args, size, rg)`. Note
        that `size` in as an argument in this function relates to the
        number of data you will generate, which is always equal to
        len(data). This is not the same as the `size` argument of
        `draw_bs_reps_mle()`, which is the number of bootstrap
        replicates you wish to draw.
    data : Numpy array, possibly multidimensional
        Array of measurements. The first index should index repeat of
        experiment. E.g., if the data consist of n (x, y) pairs, `data`
        should have shape (n, 2).
    mle_args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    gen_args : tuple, default ()
        Arguments to be passed to `gen_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    n_jobs : int, default 1
        Number of cores to use in drawing bootstrap replicates.
    progress_bar : bool, default False
        Whether or not to display progress bar.
    rg : numpy.random.Generator instance, default None
        RNG to be used in bootstrapping. If None, the default
        Numpy RNG is used with a fresh seed based on the clock.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    # Just call the original function if n_jobs is 1 (no parallelization)
    if n_jobs == 1:
        return _draw_bs_reps_mle(
            mle_fun,
            gen_fun,
            data,
            mle_args=mle_args,
            gen_args=gen_args,
            size=size,
            progress_bar=progress_bar,
            rg=rg,
        )

    if rg is not None:
        raise RuntimeError(
            "You are attempting to draw replicates in parallel with a specified random number generator (`rg` is not `None`). Each of the sets of replicates drawn in parallel will be the same since the random number generator is not reseeded for each thread. When running in parallel, you  must have `rg=None`."
        )

    # Set up sizes of bootstrap replicates for each core, making sure we
    # get all of them, even if sizes % n_jobs != 0
    sizes = [size // n_jobs for _ in range(n_jobs)]
    sizes[-1] += size - sum(sizes)

    # Build arguments
    arg_iterable = [
        (mle_fun, gen_fun, data, mle_args, gen_args, s, progress_bar, None)
        for s in sizes
    ]

    with multiprocessing.Pool(n_jobs) as pool:
        result = pool.starmap(_draw_bs_reps_mle, arg_iterable)

    return np.concatenate(result)


def _convert_data(data, inf_ok=False, min_len=1):
    """
    Convert inputted 1D data set into NumPy array of floats.
    All nan's are dropped.

    Parameters
    ----------
    data : int, float, or array_like
        Input data, to be converted.
    inf_ok : bool, default False
        If True, np.inf values are allowed in the arrays.
    min_len : int, default 1
        Minimum length of array.

    Returns
    -------
    output : ndarray
        `data` as a one-dimensional NumPy array, dtype float.
    """
    # If it's scalar, convert to array
    if np.isscalar(data):
        data = np.array([data], dtype=np.float)

    # Convert data to NumPy array
    data = np.array(data, dtype=np.float)

    # Make sure it is 1D
    if len(data.shape) != 1:
        raise RuntimeError("Input must be a 1D array or Pandas series.")

    # Remove NaNs
    data = data[~np.isnan(data)]

    # Check for infinite entries
    if not inf_ok and np.isinf(data).any():
        raise RuntimeError("All entries must be finite.")

    # Check to minimal length
    if len(data) < min_len:
        raise RuntimeError(
            "Array must have at least {0:d} non-NaN entries.".format(min_len)
        )

    return data


def _convert_two_data(x, y, inf_ok=False, min_len=1):
    """
    Converted two inputted 1D data sets into Numpy arrays of floats.
    Indices where one of the two arrays is nan are dropped.

    Parameters
    ----------
    x : array_like
        Input data, to be converted. `x` and `y` must have the same length.
    y : array_like
        Input data, to be converted. `x` and `y` must have the same length.
    inf_ok : bool, default False
        If True, np.inf values are allowed in the arrays.
    min_len : int, default 1
        Minimum length of array.

    Returns
    -------
    x_out : ndarray
        `x` as a one-dimensional NumPy array, dtype float.
    y_out : ndarray
        `y` as a one-dimensional NumPy array, dtype float.
    """
    # Make sure they are array-like
    if np.isscalar(x) or np.isscalar(y):
        raise RuntimeError("Arrays must be 1D arrays of the same length.")

    # Convert to Numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Check for infinite entries
    if not inf_ok and (np.isinf(x).any() or np.isinf(y).any()):
        raise RuntimeError("All entries in arrays must be finite.")

    # Make sure they are 1D arrays
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise RuntimeError("Input must be a 1D array or Pandas series.")

    # Must be the same length
    if len(x) != len(y):
        raise RuntimeError("Arrays must be 1D arrays of the same length.")

    # Clean out nans
    inds = ~np.logical_or(np.isnan(x), np.isnan(y))
    x = x[inds]
    y = y[inds]

    # Check to minimal length
    if len(x) < min_len:
        raise RuntimeError(
            "Arrays must have at least {0:d} mutual non-NaN entries.".format(min_len)
        )

    return x, y


@numba.jit(nopython=True)
def _allequal(x, rtol=1e-7, atol=1e-14):
    """
    Determine if all entries in an array are equal.

    Parameters
    ----------
    x : ndarray
        Array to test.

    Returns
    -------
    output : bool
        True is all entries in the array are equal, False otherwise.
    """
    if len(x) == 1:
        return True

    for a in x[1:]:
        if np.abs(a - x[0]) > (atol + rtol * np.abs(a)):
            return False
    return True


@numba.jit(nopython=True)
def _allclose(x, y, rtol=1e-7, atol=1e-14):
    """
    Determine if all entries in two arrays are close to each other.

    Parameters
    ----------
    x : ndarray
        First array to compare.
    y : ndarray
        Second array to compare.

    Returns
    -------
    output : bool
        True is each entry in the respective arrays is equal.
        False otherwise.
    """
    for a, b in zip(x, y):
        if np.abs(a - b) > (atol + rtol * np.abs(b)):
            return False
    return True


def _make_one_arg_numba_func(func):
    """
    Make a Numba'd version of a function that takes one positional
    argument.

    Parameters
    ----------
    func : function
        Function with call signature `func(x, *args)`.

    Returns
    -------
    output : Numba'd function
        A Numba'd version of the functon

    Notes
    -----
    .. If the function is Numba'able in nopython mode, it will compile
       in that way. Otherwise, falls back to object mode.
    """

    @numba.jit
    def f(x, args=()):
        return func(x, *args)

    return f


def _make_two_arg_numba_func(func):
    """
    Make a Numba'd version of a function that takes two positional
    arguments.

    Parameters
    ----------
    func : function
        Function with call signature `func(x, y, *args)`.

    Returns
    -------
    output : Numba'd function
        A Numba'd version of the functon

    Notes
    -----
    .. If the function is Numba'able in nopython mode, it will compile
       in that way. Otherwise, falls back to object mode.
    """

    @numba.jit
    def f(x, y, args=()):
        return func(x, y, *args)

    return f


def _make_rng_numba_func(func):
    """
    Make a Numba'd version of a function to draw random numbers.

    Parameters
    ----------
    func : function
        Function with call signature `func(*args, size=1)`.

    Returns
    -------
    output : Numba'd function
        A Numba'd version of the functon

    Notes
    -----
    .. If the function is Numba'able in nopython mode, it will compile
       in that way. Otherwise, falls back to object mode.
    """

    @numba.jit
    def f(args, size=1):
        all_args = args + (size,)
        return func(*all_args)

    return f


@numba.jit(nopython=True)
def _seed_numba(seed):
    """
    Seed the random number generator for Numba'd functions.

    Parameters
    ----------
    seed : int
        Seed of the RNG.

    Returns
    -------
    None
    """
    np.random.seed(seed)
