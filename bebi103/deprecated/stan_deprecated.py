import numpy as np
import pandas as pd


def to_dataframe(
    fit, pars=None, permuted=False, dtypes=None, inc_warmup=False, diagnostics=True
):
    """
    Convert the results of a fit to a DataFrame.

    Parameters
    ----------
    fit : StanFit4Model instance
        The output of fitting a Stan model.
    pars : str or list of strs
        The parameters to extract into the outputted data frame.
    permuted : bool, default False
        If True, the chains are combined and the samples are permuted.
    dypes : dict
        Each entry key, value pair is param_name, data dtype. If None,
        all data types are floats. Be careful: results are cast into
        whatever you specify.
    inc_warmup : bool, default False
        If True, include warmup samples.
    diagnostics : bool, default True
        If True, also include diagnostic information about the samples.
        Note that if no Monte Carlo sampling is done (e.g., if the
        Fixed_param algorithm is used).

    Returns
    -------
    output : DataFrame
        A Pandas DataFrame containing the samples. Each row consists of
        a single sample, including the parameters and diagnostics if
        `diagnostics` is True.

    Notes
    -----
    .. This functionality is present in PyStan >= 2.18. Due to
       compilation issues with higher versions of PyStan, we include
       this functionality here. The output of this function is meant to
       match that of fit.to_dataframe() of a StanFit4Model instance
       from PyStan 2.18 and above. Because of PyStan's GPL license,
       I have re-written this functionality here to enable permissive
       licensing of this module.
    """
    if permuted and inc_warmup:
        raise RuntimeError("If `permuted` is True, `inc_warmup` must be False.")

    if permuted and diagnostics:
        raise RuntimeError("Diagnostics are not available when `permuted` is True.")

    if dtypes is not None and not permuted and pars is None:
        raise RuntimeError(
            "`dtypes` cannot be specified when `permuted`"
            + " is False and `pars` is None."
        )

    try:
        return fit.to_dataframe(
            pars=pars,
            permuted=permuted,
            dtypes=dtypes,
            inc_warmup=inc_warmup,
            diagnostics=diagnostics,
        )
    except:
        pass

    # Diagnostics to pull out
    diags = [
        "diverging__",
        "energy__",
        "treedepth__",
        "accept_stat__",
        "stepsize__",
        "n_leapfrog__",
    ]

    # Build parameters if not supplied
    if pars is None:
        pars = tuple(fit.flatnames + ["lp__"])
    if type(pars) not in [list, tuple, pd.core.indexes.base.Index]:
        raise RuntimeError("`pars` must be list or tuple or pandas index.")
    if "lp__" not in pars:
        pars = tuple(list(pars) + ["lp__"])

    # Build dtypes if not supplied
    if dtypes is None:
        dtypes = {par: float for par in pars}
    else:
        dtype["lp__"] = float

    # Make sure dtypes supplied for every parameter
    for par in pars:
        if par not in dtypes:
            raise RuntimeError(f"'{par}' not in `dtypes`.")

    # Retrieve samples
    samples = fit.extract(
        pars=pars, permuted=permuted, dtypes=dtypes, inc_warmup=inc_warmup
    )
    n_chains = len(fit.stan_args)
    thin = fit.stan_args[0]["thin"]
    n_iters = fit.stan_args[0]["iter"] // thin
    n_warmup = fit.stan_args[0]["warmup"] // thin
    n_samples = n_iters - n_warmup

    # Dimensions of parameters
    dim_dict = {par: dim for par, dim in zip(fit.model_pars, fit.par_dims)}
    dim_dict["lp__"] = []

    if inc_warmup:
        n = (n_warmup + n_samples) * n_chains
        warmup = np.concatenate(
            [[1] * n_warmup + [0] * n_samples for _ in range(n_chains)]
        ).astype(int)
        chain = np.concatenate(
            [[i + 1] * (n_warmup + n_samples) for i in range(n_chains)]
        ).astype(int)
        chain_idx = np.concatenate(
            [np.arange(1, n_warmup + n_samples + 1) for _ in range(n_chains)]
        ).astype(int)
    else:
        n = n_samples * n_chains
        warmup = np.array([0] * n, dtype=int)
        chain = np.concatenate([[i + 1] * n_samples for i in range(n_chains)]).astype(
            int
        )
        chain_idx = np.concatenate(
            [np.arange(1, n_samples + 1) for _ in range(n_chains)]
        ).astype(int)

    if permuted:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data=dict(chain=chain, chain_idx=chain_idx, warmup=warmup))

    if diagnostics:
        sampler_params = fit.get_sampler_params(inc_warmup=inc_warmup)
        diag_vals = list(sampler_params[0].keys())

        # If they are standard, do same order as PyStan 2.18 to_dataframe
        if (
            len(diag_vals) == len(diags)
            and sum([val not in diags for val in diag_vals]) == 0
        ):
            diag_vals = diags
        for diag in diag_vals:
            df[diag] = np.concatenate(
                [sampler_params[i][diag] for i in range(n_chains)]
            )
            if diag in ["treedepth__", "n_leapfrog__"]:
                df[diag] = df[diag].astype(int)

    if isinstance(samples, np.ndarray):
        for k, par in enumerate(pars):
            try:
                indices = re.search("\[(\d+)(,\d+)*\]", par).group()
                base_name = re.split("\[(\d+)(,\d+)*\]", par, maxsplit=1)[0]
                col = base_name + re.sub(
                    "\d+", lambda x: str(int(x.group()) + 1), indices
                )
            except AttributeError:
                col = par

            df[col] = samples[:, :, k].flatten(order="F")
    else:
        for par in pars:
            if len(dim_dict[par]) == 0:
                df[par] = samples[par].flatten(order="F")
            else:
                for inds in itertools.product(*[range(dim) for dim in dim_dict[par]]):
                    col = (
                        par + "[" + ",".join([str(ind + 1) for ind in inds[::-1]]) + "]"
                    )

                    if permuted:
                        array_slice = tuple([slice(n), *inds[::-1]])
                    else:
                        array_slice = tuple([slice(n), slice(n), *inds[::-1]])

                    df[col] = samples[par][array_slice].flatten(order="F")

    return df


def to_arviz(fit, log_likelihood=None, logging_disable_level=logging.ERROR):
    """Convert a StanFit4Model to an ArviZ data set.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : ArviZ data object.

    Notes
    -----
    .. This function is only necessary because of problems ArviZ has
       with PyStan < 2.18. You should just directly use
       `arviz.from_pystan()` if you have PyStan >= 2.18. Note that this
       function does not allow other kwargs besides `log_likelihood`
       that are available in `arviz.from_pystan()`. The main purpose
       of this function is to get minimal information necessary for
       doing PSIS-LOO and WAIC calculations using ArviZ.
    """
    with _disable_logging(level=logging_disable_level):
        az_data = az.from_pystan(fit=fit, log_likelihood=log_likelihood)

    if pystan.__version__ < "2.18":
        if log_likelihood is not None:
            # Get the log likelihood
            log_lik = np.swapaxes(
                extract_par(
                    fit, log_likelihood, logging_disable_level=logging_disable_level
                ),
                0,
                1,
            )

            # dims for xarray
            dims = ["chain", "draw", "log_likelihood_dim_0"]

            # Properly do the log likelihood
            az_data.sample_stats["log_likelihood"] = (dims, log_lik)

        # Get the lp__
        with _disable_logging(level=logging_disable_level):
            lp = np.swapaxes(fit.extract(permuted=False)[:, :, -1], 0, 1)
        az_data.sample_stats["lp"] = (["chain", "draw"], lp)

    return az_data

def waic(
    fit, log_likelihood=None, pointwise=False, logging_disable_level=logging.ERROR
):
    """Compute the WAIC using ArviZ.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    pointwise : bool, default False
        If True, also return point-wise WAIC.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Pandas data frame
        Pandas DataFrame with columns:
          waic: widely available information criterion
          waic_se: standard error of waic
          p_waic: effective number parameters
          var_warn: 1 if posterior variance of the log predictive
            densities exceeds 0.4
          waic_i: and array of the pointwise predictive accuracy, only
            if `pointwise` True
    """
    if log_likelihood is None:
        raise RuntimeError("Must supply `log_likelihood`.")

    az_data = to_arviz(
        fit, log_likelihood=log_likelihood, logging_disable_level=logging_disable_level
    )

    return az.waic(az_data, pointwise=pointwise)


def loo(
    fit,
    log_likelihood=None,
    pointwise=False,
    reff=None,
    logging_disable_level=logging.ERROR,
):
    """Compute the PSIS-LOO.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    pointwise : bool, default False
        If True, also return point-wise predictive accuracy.
    reff : float, optional
        Relative MCMC efficiency, `effective_n / n` i.e. number of
        effective samples divided by the number of actual samples.
        Computed from trace by default.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Pandas data frame
        Pandas DataFrame with columns:
          loo: approximated Leave-one-out cross-validation
          loo_se: standard error of loo
          p_loo: effective number of parameters
          shape_warn: 1 if the estimated shape parameter of Pareto
            distribution is greater than 0.7 for one or more samples.
          loo_i: array of pointwise predictive accuracy, only if
            `pointwise` True
    """
    if log_likelihood is None:
        raise RuntimeError("Must supply `log_likelihood`.")

    az_data = to_arviz(
        fit, log_likelihood=log_likelihood, logging_disable_level=logging_disable_level
    )

    return az.loo(az_data, pointwise=pointwise, reff=reff)


def compare(
    fit_dict, log_likelihood=None, logging_disable_level=logging.ERROR, **kwargs
):
    """Compare models.

    Parameters
    ----------
    fit_dict : StanFit4Model instance
        A dictionary where each key it a name attached to a model and
        the value is a StanFit4Model instance that has the posterior
        samples.
    log_likelihood : str
        Name of the variable containing the log likelihood.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Pandas data frame
        Pandas DataFrame with columns:
          IC : Information Criteria (WAIC or LOO).
          pIC : Estimated effective number of parameters.
          dIC : Relative difference between each IC (WAIC or LOO) and
            the lowest IC (WAIC or LOO). It is always 0 for the
            top-ranked model.
          weight: Relative weight for each model.
            This can be loosely interpreted as the probability of each
            model (among the compared model) given the data. By default
            the uncertainty in the weights estimation is considered using
            Bayesian bootstrap.
          SE : Standard error of the IC estimate. If `method` is
            BB-pseudo-BMA, these values are estimated using Bayesian
            bootstrap.
          dSE : Standard error of the difference in IC between each
            model and the top-ranked model. It is always 0 for the
            top-ranked model.
          warning : A value of 1 indicates that the computation of the
            IC may not be reliable. This could be indication of WAIC/LOO
            starting to fail; see http://arxiv.org/abs/1507.04544.

    Notes
    -----
    .. All kwargs are passed into arviz.stats.compare(). These kwargs
       are given in the ArviZ documentation. Importantly, use
       `ic='waic'` or `ic='loo'` to respectively use WAIC or LOO as the
       information criterion. WAIC is the default. Use
       `method='stacking'` for stacking to compute weights (default) and
       `method='BB-pseudo-BMA'` to use pseudo-Bayesian model averaging
       with Akaike-type weights.

    """
    if log_likelihood is None:
        raise RuntimeError("Must supply `log_likelihood`.")

    arviz_dict = {}
    for key in fit_dict:
        arviz_dict[key] = to_arviz(
            fit_dict[key],
            log_likelihood=log_likelihood,
            logging_disable_level=logging_disable_level,
        )

    return az.compare(arviz_dict, **kwargs)


def pickle_dump_samples(fit=None, model=None, pkl_file=None):
    """Dump samples into a pickle file.

    Parameters
    ----------
    fit : StanFit4Model instance
        The output of fitting a Stan model.
    model : StanModel instance
        StanModel instance used to get `fit`.
    pkl_file : str
        Name of pickle file to create and dump result.

    Returns
    -------
    None
    """
    if None in [fit, model, pkl_file]:
        raise RuntimeError("`fit`, `model`, and `pkl_file` must all be specified.")

    if os.path.isfile(pkl_file):
        raise RuntimeError(f"File {pkl_file} already exists.")

    with open(pkl_file, "wb") as f:
        pickle.dump({"model": model, "fit": fit}, f, protocol=-1)


def pickle_load_samples(pkl_file):
    """Load samples out of a pickle file.

    Parameters
    ----------
    fit : StanFit4Model instance
        The output of fitting a Stan model.
    model : StanModel instance
        StanModel instance used to get `fit`.
    pkl_file : str
        Name of pickle file to create and dump result.

    Returns
    -------
    None
    """
    with open(pkl_file, "rb") as f:
        data_dict = pickle.load(f)

    return data_dict["fit"], data_dict["model"]


def extract_array(samples, name):
    """Extract an array values from a DataFrame containing samples.

    Parameters
    ----------
    samples : StanFit4Model instance or Pandas DataFrame
        Samples from a Stan calculation that contains an array for
        extraction.
    name : str
        The name of the array to extract. For example, if `name` is
        'my_array', and the array is one-dimensional, entries with
        variable names 'my_array[1]', 'my_array[2]', ... are extracted.

    Returns
    -------
    output : Pandas DataFrame
        A tidy DataFrame with the extracted array. For a 1-D array, the
        DataFrame has columns:
            ['chain', 'chain_idx', 'warmup', 'index_1', 'name']
        For a 2D array, there is an additional column named 'index_2'. A
        3D array has a column 'index_3' and so on.
    """
    df = _fit_to_df(samples, diagnostics=False)

    regex_name = name
    for char in "[\^$.|?*+(){}":
        regex_name = regex_name.replace(char, "\\" + char)

    # Extract columns that match the name
    sub_df = df.filter(regex=regex_name + "\[(\d+)(,\d+)*\]")

    if len(sub_df.columns) == 0:
        raise RuntimeError(
            "column '{}' is either absent or scalar-valued.".format(name)
        )

    n_entries = len(sub_df.columns)
    n = len(sub_df)

    df_out = pd.DataFrame(data={name: sub_df.values.flatten(order="F")})
    for col in ["chain", "chain_idx", "warmup"]:
        if col in df:
            df_out[col] = np.concatenate([df[col].values] * n_entries)

    indices = [
        re.search("\[(\d+)(,\d+)*\]", col).group()[1:-1].split(",")
        for col in sub_df.columns
    ]
    indices = np.vstack([np.array([[int(i) for i in ind]] * n) for ind in indices])
    ind_df = pd.DataFrame(
        columns=["index_{0:d}".format(i) for i in range(1, indices.shape[1] + 1)],
        data=indices,
    )

    return pd.concat([ind_df, df_out], axis=1)


def extract_par(
    fit, par, permuted=False, inc_warmup=False, logging_disable_level=logging.ERROR
):
    """Extract samples of a parameter out of a StanFit4Model.

    Parameters
    ----------
    fit : StanFit4Model instance
        Samples from a Stan calculation.
    par : str
        The name of the variable to extract. For example, if `name` is
        'my_array', and the array is one-dimensional, entries with
        variable names 'my_array[1]', 'my_array[2]', ... are extracted.
    permuted : bool, default False
        If True, the chains are combined and the samples are permuted.
    inc_warmup : bool, default False
        If True, include warmup samples.
    logging_disable_level : str, default logging.ERROR
        Logging is disabled below this level. This helps deal with
        annoying warning from PyStan when doing extractions when dtype
        is not specified and permuted is False.

    Returns
    -------
    output : Numpy array
        Numpy array with the samples. The first dimension of
        the array is for the iterations; the second for the number of chains;
        the third for the parameters. Vectors and arrays are expanded to one
        parameter (a scalar) per cell, with names indicating the third dimension.

    Notes
    -----
    .. PyStan 2.17.1.0 has a bug where the `extract()` functionality of
       a StanFit4Model instance does not work as advertised. This
       accommodates that functionality.
    """
    if permuted:
        return fit.extract(pars=par, permuted=True, dtypes=None, inc_warmup=inc_warmup)[
            par
        ]

    if pystan.__version__ >= "2.18":
        with _disable_logging(level=logging_disable_level):
            return fit.extract(
                pars=par, permuted=False, dtypes=None, inc_warmup=inc_warmup
            )

    with _disable_logging(level=logging_disable_level):
        data = fit.extract(permuted=False, inc_warmup=inc_warmup, dtypes=None)

    inds = []
    for k, par_name in enumerate(fit.flatnames):
        try:
            indices = re.search("\[(\d+)(,\d+)*\]", par_name).group()
            base_name = re.split("\[(\d+)(,\d+)*\]", par_name, maxsplit=1)[0]
            if base_name == par:
                inds.append(k)
        except AttributeError:
            if par_name == par:
                inds.append(k)

    if len(inds) == 1:
        return data[:, :, inds[0]]

    return data[:, :, inds]


def hpd(x, mass_frac):
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    x : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(x))

    # Number of total samples taken
    n = len(x)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[: n - n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int + n_samples]])

def posterior_predictive_ranking(fit=None, ppc_name=None, data=None, fractional=False):
    """Compute posterior predictive ranking of each data point.

    Parameters
    ----------
    fit : StanFit4Model instance
        A Stan fit of a model that has posterior predictive check values
        for the data.
    ppc_name : str
        The name of the parameter in the `fit` that has the posterior
        predictive checks.
    data : 1D Numpy array
        The data points that were observed.

    Returns
    -------
    output : 1D Numpy array, same shape as `data`
        The ranking among the posterior predictive samples of each data
        points.
    """
    if fit is None or ppc_name is None or data is None:
        raise RuntimeError("`fit`, `ppc_name`, and `data` must all be specified.")

    ppc = fit.extract(ppc_name)[ppc_name]
    rankings = np.empty(ppc.shape[1])

    for i in range(ppc.shape[1]):
        rankings[i] = np.searchsorted(np.sort(ppc[:, i]), data[i])

    if fractional:
        rankings /= ppc.shape[0]

    return rankings


def _infer_max_treedepth(fit):
    """Extract max treedepth from a StanFit4Model instance."""
    try:
        return fit.stan_args[0]["ctrl"]["sampling"]["max_treedepth"]
    except:
        warnings.warn("Unable to determine max_treedepth. " + "Using value of 10.")
        return 10
        

def _fit_to_az(fit, **kwargs):
    """Convert StanFit4Model to data frame."""
    if "StanFit4Model" in str(type(fit)):
        df = to_dataframe(fit, inc_warmup=False, **kwargs)
    elif type(fit) != pd.core.frame.DataFrame:
        raise RuntimeError("`fit` must be a StanModel or Pandas data frame.")
    else:
        df = fit.loc[fit["warmup"] == 0, :]

    return df
