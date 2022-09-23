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
    data_1 = utils._convert_data(data_1)
    data_2 = utils._convert_data(data_2)

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
