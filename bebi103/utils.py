import os

import multiprocessing
import warnings

import tqdm

import numpy as np
import numba

def unpivot_csv(
    input_file: str,
    output_file: str,
    n_header_rows=None,
    header_names=None,
    value_name: str = "value",
    sep: str = ",",
    retain_row_index: bool = True,
    row_index_name: str = "original_row_index",
    comment_prefix=None,
    force_overwrite=False,
    include_comments=True,
    additional_comment=None,
) -> type(None):
    """Converts a CSV file in wide format, possibly with a hierarchical
    column, index to tall format via a unpivot operation.

    Parameters
    ----------
    input_file : str
        Path to CSV file containing data in a wide format.
    output_file : str
        Path to file to write contents in tall format.
    n_header_rows : int, default 1
        Number of header rows, ignored if `header_names` is given.
    header_names : string, list of strings, or None, default None
        Names of each level of the header. These become the names of
        columns in the tall representation of the data set. If None,
        column names are variable_0, variable_1, etc.
    value_name : str or None, default 'value'
        Name to assign to column containing values in tall
        representation.
    sep : str, default ','
        Separator. Cannot be a regex; must be a single character.
    retain_row_index : bool, default True
        If True, include a column containing the original row index
        in the original file. This is useful to avoid information loss,
        since all entries in a given row can be related. These are 
        essentially identifies variables.
    row_index_name : str, default 'original_row_index'
        Label in the row index if the row index is retained. Ignored if
        `retain_row_index` is False.
    comment_prefix : str or None, default None
        A single character defining which rows in the input file should
        be treated as comments.
    force_overwrite : bool, default False
        If True, if the output_file already exists, it is overwritten.
    include_comments : bool, default True
        If True, write comments from the input file in the output files.
    additional_comment : str or None, default None
        Additional comment at top of output file.

    Returns
    -------
    output : None
        Does not return anything. The output is written to output_file.
    """
    if input_file == output_file:
        raise RuntimeError("`input_file` and `output_file` are the same.")

    if not force_overwrite and os.path.isfile(output_file):
        raise RuntimeError(f"File {output_file} exists, not overwriting.")

    if header_names is None:
        if n_header_rows is None:
            n_header_rows = 1
        header_names = [f"variable_{i}" for i in range(n_header_rows)]
    elif np.isscalar(header_names):
        header_names = [header_names]

    if n_header_rows is None:
        n_header_rows = len(header_names)

    if len(header_names) != n_header_rows:
        raise RuntimeError("len(header_names) != n_header_rows")

    with open(input_file, "r") as f, open(output_file, "w") as fout:
        # Read and write comments
        line = f.readline()
        while line != "" and line[0] == comment_prefix:
            if include_comments:
                fout.write(line)
            line = f.readline()

        # Headers, a list of lists
        headers = [line.rstrip().split(sep)] + [
            f.readline().rstrip().split(sep) for _ in range(n_header_rows - 1)
        ]

        # Header to output
        if retain_row_index:
            fout.write(f"{row_index_name},")
        fout.write(sep.join(header_names + [value_name]) + "\n")

        # Now read data and add to output file
        row_index = 0
        line = f.readline()
        while line != "":
            data = line.split(sep)
            for i, data_str in enumerate(data):
                data_str = data_str.strip().rstrip()
                if retain_row_index:
                    fout.write(str(row_index) + sep)
                if data_str != "":
                    fout.write(
                        sep.join([head[i] for head in headers] + [data_str]) + "\n"
                    )
            line = f.readline()
            row_index += 1

    return None


def _dummy_jit(*args, **kwargs):
    """Dummy wrapper for jitting if numba not applicable."""

    def wrapper(f):
        return f

    def marker(*args, **kwargs):
        return marker

    if (
        len(args) > 0
        and (args[0] is marker or not callable(args[0]))
        or len(kwargs) > 0
    ):
        return wrapper
    elif len(args) == 0:
        return wrapper
    else:
        return args[0]


@numba.jit(nopython=True)
def _ecdf(x, data):
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
    Only entries where both `data_1` and `data_2` are not NaN are
    used.

    If the variance of `data_1` or `data_2` is zero, return NaN.
    """
    if _allequal(x) or _allequal(y):
        return np.nan

    return (np.mean(x * y) - np.mean(x) * np.mean(y)) / np.std(x) / np.std(y)


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
        data = np.array([data], dtype=float)

    # Convert data to NumPy array
    data = np.array(data, dtype=float)

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


def _make_one_arg_numba_func(func, func_args):
    """
    Make a Numba'd version of a function that takes one positional
    argument.

    Parameters
    ----------
    func : function
        Function with call signature `func(x, *args)`.
    func_args : tuple
        Tuple of args to use in testing a function call.

    Returns
    -------
    output : Numba'd function (or original function)
        A Numba'd version of the function. If that is not possible,
        returns the original function.
    numba_success : bool
        True if function was successfully jitted.

    """
    try:
        func_numba = numba.jit(func, nopython=True)

        @numba.jit(nopython=True)
        def f(x, args=()):
            return func_numba(x, *args)

        # Attempt function call
        _ = f(np.array([1.0, 2.0]), func_args)

        return f, True
    except:

        def f(x, args=()):
            return func(x, *args)

        return f, False


def _make_two_arg_numba_func(func, func_args):
    """
    Make a Numba'd version of a function that takes two positional
    arguments.

    Parameters
    ----------
    func : function
        Function with call signature `func(x, y, *args)`.
    func_args : tuple
        Tuple of args to use in testing a function call.

    Returns
    -------
    output : Numba'd function (or original function)
        A Numba'd version of the function. If that is not possible,
        returns the original function.
    numba_success : bool
        True if function was successfully jitted.
    """
    try:
        func_numba = numba.jit(func, nopython=True)

        @numba.jit(nopython=True)
        def f(x, args=()):
            return func_numba(x, *args)

        # Attempt function call
        _ = f(np.array([1.0, 2.0]), np.array([1.0, 2.0]), func_args)

        return f, True
    except:

        def f(x, y, args=()):
            return func(x, y, *args)

        return f, False


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
