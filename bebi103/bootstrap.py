try:
    import multiprocess
except:
    import multiprocessing as multiprocess

import warnings

import tqdm

import numpy as np
import numba

from . import utils


def seed_rng(seed):
    """
    Seed random number generators for Numpy and Numba'd functions.

    Parameters
    ----------
    seed : long int
        Seed for RNG
    """
    try:

        @numba.jit(nopython=True)
        def _seed(seed):
            np.random.seed(seed)

        _seed(seed)
    except:
        warnings.warn(
            "Unable to seed Numba RNG. It is possible Numba is not"
            " properly installed. If that is the case, you all bootstrap calculations"
            " will use un-Numba'd random number generation that is properly seeded."
            " However, if Numba is installed and there is some other issue preventing"
            " proper seeding of the random number generator used in Numba'd code,"
            " you may get unexpected results."
        )

    np.random.seed(seed)


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
    nan values are ignored.
    """
    data = utils._convert_data(data)

    if args == ():
        if func == np.mean:
            return _draw_bs_reps_mean(data, size=size)
        elif func == np.median:
            return _draw_bs_reps_median(data, size=size)
        elif func == np.std:
            return _draw_bs_reps_std(data, size=size)

    # Make Numba'd function
    f, numba_success = utils._make_one_arg_numba_func(func, args)

    if numba_success:
        jit = numba.jit
    else:
        jit = utils._dummy_jit

    @jit(nopython=True)
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


def draw_bs_reps_pairs(x, y, func, size=1, args=()):
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
    x, y = utils._convert_two_data(x, y, min_len=1)

    # Make Numba'd function
    f, numba_success = utils._make_two_arg_numba_func(func, args)

    if numba_success:
        jit = numba.jit
    else:
        jit = utils._dummy_jit

    n = len(x)

    @jit(nopython=True)
    def _draw_bs_reps_pairs(x, y):
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

    return _draw_bs_reps_pairs(x, y)


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
    data_1 = utils._convert_data(data_1)
    data_2 = utils._convert_data(data_2)

    if args == ():
        if func == diff_of_means:
            return _draw_perm_reps_diff_of_means(data_1, data_2, size=size)
        elif func == studentized_diff_of_means:
            if len(data_1) == 1 or len(data_2) == 1:
                raise RuntimeError("Data sets must have at least two entries")
            return _draw_perm_reps_studentized_diff_of_means(data_1, data_2, size=size)

    # Make a Numba'd function for drawing reps.
    f, numba_success = utils._make_two_arg_numba_func(func, args)

    if numba_success:
        jit = numba.jit
    else:
        jit = utils._dummy_jit

    @jit(nopython=True)
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
    data_1 = utils._convert_data(data_1)
    data_2 = utils._convert_data(data_2)

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
    If the variance of both `data_1` and `data_2` is zero, returns
    np.nan.
    """
    data_1 = utils._convert_data(data_1, min_len=2)
    data_2 = utils._convert_data(data_2, min_len=2)

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
    If the variance of both `data_1` and `data_2` is zero, returns
    np.nan.
    """
    if utils._allequal(data_1) and utils._allequal(data_2):
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
    Only entries where both `data_1` and `data_2` are not NaN are
    used.

    If the variance of `data_1` or `data_2` is zero, return NaN.
    """
    x, y = utils._convert_two_data(data_1, data_2, inf_ok=False, min_len=2)
    return utils._pearson_r(x, y)


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
        Function with call signature `mle_fun(data, *mle_args)` that
        computes a MLE for the parameters
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
        Function with call signature `mle_fun(data, *mle_args)` that
        computes a MLE for the parameters.
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(params, *gen_args, size, rg)`. Note
        that `size` as an argument in this function relates to the
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
            "You are attempting to draw replicates in parallel with a specified random"
            " number generator (`rg` is not `None`). Each of the sets of replicates"
            " drawn in parallel will be the same since the random number generator is"
            " not reseeded for each thread. When running in parallel, you  must have"
            " `rg=None`."
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

    with multiprocess.Pool(n_jobs) as pool:
        result = pool.starmap(_draw_bs_reps_mle, arg_iterable)

    return np.concatenate(result)
