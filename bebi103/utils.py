import os
import typing

import numpy as np
import numpy.typing as npt
import numba

def array_to_dict(
    input_array: npt.NDArray,
    base_name: str, 
    n_index_axes: int = 1,
    one_indexed: bool = True,
    dtype=None,
) -> dict[str, typing.Any]:
    """Given a Numpy array, convert to a dictionary with each key 
    being of the form 'base_name[1]', 'base_name[1,2]', etc. The last
    `n_index_axes` axes of `input_array` are the indices of the array
    itself and become the indices in the keys. The leading axes,
    typically those of the samples, are retained in the values. This is
    the inverse of `dict_to_array()`.

    Parameters
    ----------
    input_array : Numpy array
        Numpy array to convert to dictionary.
    base_name : str
        Name of array represented to be represented in the dictionary.
        The keys will be of the form `f'{base_name}[1]`.
    n_index_axes : int, default 1
        Number of trailing axes of `input_array` that index the array
        itself. These become the indices in the keys of the outputted
        dictionary, so the keys have `n_index_axes` indices each. The
        remaining leading axes, in typical applications in MCMC the
        chain ID and the draw, are retained in each value.
    one_indexed : bool, default True
        True if indexing of the array in the dictionary starts at 1,
        as is the case with Stan, and False if indexing starts at 0.
    dtype : int, float, or None
        If not None, each entry is outputted as the specified dtype.
    
    Returns
    -------
    output : dict
        Dictionary with entries like 'base_name[1]', 'base_name[2]',
        etc. Each value is a Numpy array of shape
        `input_array.shape[:-n_index_axes]`. If `n_index_axes` is the
        dimensionality of `input_array`, no axes are left over and
        each value is instead a scalar.

    Raises
    ------
    ValueError
        If `n_index_axes` is less than one or exceeds the
        dimensionality of `input_array`.

    Notes
    -----
    This is the inverse of `dict_to_array()`. For an array `a`, ::

        dict_to_array(
            array_to_dict(a, 'a', n_index_axes=k), 'a', squeeze=False
        )

    gives back `a`. With `squeeze=True`, `dict_to_array()` discards
    axes of length one, so it is not in general invertible.

    The values of the outputted dictionary are copies; they are not
    views into `input_array`.

    Examples
    --------
    >>> array_to_dict(np.array([1, 2, 3]), 'a')
    {'a[1]': 1, 'a[2]': 2, 'a[3]': 3}

    >>> array_to_dict(np.array([[1, 2], [3, 4], [5, 6]]), 'a', n_index_axes=2)
    {'a[1,1]': 1, 'a[1,2]': 2, 'a[2,1]': 3, 'a[2,2]': 4, 'a[3,1]': 5, 'a[3,2]': 6}

    >>> array_to_dict(np.array([[1, 2], [3, 4], [5, 6]]), 'a')
    {'a[1]': array([1, 3, 5]), 'a[2]': array([2, 4, 6])}

    >>> array_to_dict(np.array([1, 2, 3]), 'a', one_indexed=False)
    {'a[0]': 1, 'a[1]': 2, 'a[2]': 3}
    """
    input_array = np.asarray(input_array)

    if n_index_axes < 1:
        raise ValueError(
            f"`n_index_axes` must be at least one; got {n_index_axes}."
        )

    if n_index_axes > input_array.ndim:
        raise ValueError(
            f"`n_index_axes` is {n_index_axes}, but `input_array` has only "
            f"{input_array.ndim} dimensions."
        )

    offset = int(one_indexed)

    # Shape of the array itself; the leading axes belong to the samples.
    index_shape = input_array.shape[input_array.ndim - n_index_axes :]

    # With no leading axes left over, each value is a single number.
    scalar_values = n_index_axes == input_array.ndim

    output = {}
    for inds in np.ndindex(*index_shape):
        key = "{0}[{1}]".format(
            base_name, ",".join(str(i + offset) for i in inds)
        )

        # `astype()` returns a copy; otherwise copy explicitly so that
        # the output does not hold a view into `input_array`.
        val = input_array[(Ellipsis,) + inds]
        val = val.astype(dtype) if dtype is not None else val.copy()

        output[key] = val.item() if scalar_values else val

    return output


def dict_to_array(
    input_dict: dict, 
    base_name: str, 
    one_indexed: bool = True,
    dtype=None,
    squeeze: bool = False,
    squeeze_to_scalar: bool = True,
) -> npt.NDArray:
    """Given a dictionary `input_dict` with keys 'base_name[1]',
    'base_name[2]', etc., for a one-dimensional array, or 
    'base_name[1,1]', 'base_name[1,2]', etc., for a multidimensional
    array, where each value is a scalar, a 1D array, or a 2D array, 
    as would be returned in a dictionary from MCMC sampling, convert 
    to a NumPy array. If there are n indices in the keys, these 
    become the last n indices of the outputted array.

    Parameters
    ----------
    input_dict : dict
        Dictionary with entries like 'base_name[1]', 'base_name[2]',
        `base_name[1,1]`, etc. Whitespace around the indices is 
        ignored, so `base_name[1, 1]` is also accepted. Each entry is a
        1D array or 2D array, and all entries must have the same shape.
        As an example, an input dictionary could be::

            {'a[1]': np.array([[1, 2, 3, 4]]),
             'a[2]': np.array([[3, 5, 6, 7]]),
             'a[3]': np.array([[9, 10, 11, 12]])}
    base_name : str
        Name of array represented in the dictionary. In the above 
        example, `base_name` is 'a'.
    one_indexed : bool, default True
        True if indexing of the array in the dictionary starts at 1,
        as is the case with Stan, and False if indexing starts at 0.
    dtype : data type or None
        If not None, outputted array is returned as data type dtype.
    squeeze : bool, default False
        If True, dimensions of length 1 are removed from the arrays
        in the output dictionary using `np.squeeze()`.
    squeeze_to_scalar : bool, default True
        If all dimensions are of length 1, `np.squeeze()` gives a 
        0D array, not a scalar. If `squeeze_to_scalar` is True, then
        0D arrays are converted to scalars.
    
    Returns
    -------
    output : Numpy array
        Numpy array whose leading indices are those of the samples
        (the shape of the entries of `input_dict`) and whose last n
        indices are the n indices of the array in the keys of 
        `input_dict`.

    Raises
    ------
    ValueError
        If `input_dict` has no array in it with name `base_name`; if
        the keys for `base_name` do not all have the same number of
        indices; if the entries of `input_dict` for `base_name` do not
        all have the same shape; or if any element of the array is
        missing from `input_dict`.

    Notes
    -----
    Every element of the array must be present in `input_dict`; the
    indices in the keys must cover a complete array. An exception is
    raised otherwise.

    In typical applications in MCMC, the first index of the outputted
    Numpy array is the chain ID, the second is the draw, and the
    remaining indices are the indices of the array.

    This is the inverse of `array_to_dict()`. For an array `a`, ::

        dict_to_array(
            array_to_dict(a, 'a', n_index_axes=k), 'a', squeeze=False
        )

    gives back `a`. With `squeeze=True`, axes of length one are
    discarded, so the conversion is not in general invertible.

    Examples
    --------
    >>> dict_to_array(
            {'a[1]': np.array([[1, 2, 3, 4]]),
             'a[2]': np.array([[5, 6, 7, 8]]),
             'a[3]': np.array([[9, 10, 11, 12]])},
            'a',
            squeeze=True,
        )
    array([[ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11],
           [ 4,  8, 12]])

    >>> dict_to_array(
            {'a[1]': np.array([[1, 2, 3, 4]]),
                'a[2]': np.array([[5, 6, 7, 8]]),
                'a[3]': np.array([[9, 10, 11, 12]])},
            'a',
        )
    array([[[ 1,  5,  9],
            [ 2,  6, 10],
            [ 3,  7, 11],
            [ 4,  8, 12]]])

    >>> dict_to_array(
            {'a[1]': np.array([[1, 2, 3, 4],
                               [1.1, 2.2, 3.3, 4.4]]),
             'a[2]': np.array([[5, 6, 7, 8],
                               [5.5, 6.6, 7.7, 8.8]]),
             'a[3]': np.array([[9, 10, 11, 12],
                               [9.9, 10.0, 11.1, 12.2]])},
            'a',
        )
    array([[[ 1. ,  5. ,  9. ],
            [ 2. ,  6. , 10. ],
            [ 3. ,  7. , 11. ],
            [ 4. ,  8. , 12. ]],

           [[ 1.1,  5.5,  9.9],
            [ 2.2,  6.6, 10. ],
            [ 3.3,  7.7, 11.1],
            [ 4.4,  8.8, 12.2]]]) 

    >>> dict_to_array(
            {'a[1,1]': np.array([[1, 2]]),
             'a[1,2]': np.array([[3, 4]]),
             'a[2,1]': np.array([[5, 6]]),
             'a[2,2]': np.array([[7, 8]]),
             'a[3,1]': np.array([[9, 10]]),
             'a[3,2]': np.array([[11, 12]])},
            'a',
        )
    array([[[ 1,  3],
            [ 5,  7],
            [ 9, 11]],

           [[ 2,  4],
            [ 6,  8],
            [10, 12]]])
    """
    offset = int(one_indexed)

    # Pull out the entries of the dictionary for `base_name`, keyed by
    # their multi-index, shifted to be zero-indexed.
    entries = {}
    n_inds = None
    for key, val in input_dict.items():
        inds = _parse_indices(key, base_name)

        if inds is None:
            continue

        if n_inds is None:
            n_inds = len(inds)
        elif len(inds) != n_inds:
            raise ValueError(
                f"Dimensionality mismatch for array '{base_name}' in the "
                f"input dictionary; key '{key}' has {len(inds)} indices, "
                f"but other keys have {n_inds}."
            )

        entries[tuple(i - offset for i in inds)] = np.asarray(val)

    if not entries:
        raise ValueError(
            f"No array named '{base_name}' in the input dictionary."
        )

    # The samples for each element of the array must have matching shape
    # for the output array to be feasible.
    shapes = {val.shape for val in entries.values()}
    if len(shapes) > 1:
        raise ValueError(
            f"Entries for array '{base_name}' in the input dictionary do "
            f"not all have the same shape; found shapes "
            f"{', '.join(str(shape) for shape in sorted(shapes))}."
        )
    sample_shape = shapes.pop()

    if min(min(inds) for inds in entries) < 0:
        raise ValueError(
            f"Array '{base_name}' has an index less than {offset} in the "
            f"input dictionary, but `one_indexed` is {one_indexed}."
        )

    # Shape of the array itself, inferred from the largest index along
    # each of its dimensions.
    index_shape = tuple(
        max(inds[i] for inds in entries) + 1 for i in range(n_inds)
    )

    # Every element of the array must be present to fill the output.
    missing = [inds for inds in np.ndindex(*index_shape) if inds not in entries]
    if missing:
        missing_keys = ", ".join(
            "{0}[{1}]".format(base_name, ",".join(str(i + offset) for i in inds))
            for inds in missing[:5]
        )
        raise ValueError(
            f"Array '{base_name}' has shape {index_shape} in the input "
            f"dictionary, but is missing {len(missing)} of its "
            f"{len(missing) + len(entries)} elements, including: "
            f"{missing_keys}."
        )

    # Stack the elements in C order along the leading axis, then move
    # that axis to the end and unravel it into the array's own indices.
    result = np.array([entries[inds] for inds in np.ndindex(*index_shape)])
    result = np.moveaxis(result, 0, -1)
    result = result.reshape(sample_shape + index_shape)

    if dtype is not None:
        result = result.astype(dtype)

    if squeeze:
        result = result.squeeze().copy()
    if len(result.shape) == 0 and squeeze_to_scalar:
        result = result.item()
    
    return result


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
        Additional comment at top of output file. It is written above
        any comments carried over from the input file. Each line of the
        comment is prefixed with `comment_prefix` (or '#' if
        `comment_prefix` is None) unless it already begins with it.

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

    # Prefix used for comments written to the output file
    out_comment_prefix = "#" if comment_prefix is None else comment_prefix

    with open(input_file, "r") as f, open(output_file, "w") as fout:
        # Write additional comment at top of output file
        if additional_comment is not None:
            for comment_line in str(additional_comment).splitlines():
                if not comment_line.startswith(out_comment_prefix):
                    comment_line = out_comment_prefix + " " + comment_line
                fout.write(comment_line.rstrip() + "\n")

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


def _parse_indices(key, base_name):
    """Parse the indices out of a dictionary key such as 'a[1,3,2]'.

    Parameters
    ----------
    key : str
        Key from a dictionary of samples, e.g., 'a[1,3,2]'.
    base_name : str
        Name of the array of interest, e.g., 'a'.

    Returns
    -------
    output : tuple of ints or None
        The indices in the key, e.g., (1, 3, 2). If `key` is not an
        entry of the array `base_name`, return None.
    """
    if not key.startswith(base_name + "[") or not key.endswith("]"):
        return None

    inds = [ind.strip() for ind in key[len(base_name) + 1 : -1].split(",")]

    # Anything that is not a plain index means this is not an entry of
    # the array `base_name`.
    if any(not ind.isdigit() for ind in inds):
        return None

    return tuple(int(ind) for ind in inds)


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
        def f(x, y, args=()):
            return func_numba(x, y, *args)

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


def _check_kwarg_vs_dict(in_dict, kwarg, kwarg_value):
    """Check to see if an inputted kwarg is in a dictionary. If not in
    dictionary, add it. Otherwise, check to make sure values in 
    dictionary and inputted kwarg are consistent."""
    if kwarg in in_dict:
        if in_dict[kwarg] != kwarg_value:
            raise RuntimeError(f"Disagreement between {kwarg} and kwarg dictionary.")
    else:
        in_dict[kwarg] = kwarg_value    

    return in_dict