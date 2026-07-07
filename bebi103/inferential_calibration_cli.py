"""Command-line interface for running inferential calibration analysis."""

import pathlib
import tomllib

import click

from . import stan


def parse_toml_config(toml_path):
    """Parse TOML configuration file and extract inferential calibration parameters.

    Parameters
    ----------
    toml_path : str or pathlib.Path
        Path to the TOML configuration file.

    Returns
    -------
    config : dict
        Dictionary containing all inferential calibration configuration parameters.
    """
    toml_path = pathlib.Path(toml_path)

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    return config


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file):
    """Run inferential calibration analysis from a TOML configuration file.

    All parameters including output file path are specified in the config file.

    CONFIG_FILE: Path to TOML configuration file
    """
    click.echo(f"Reading configuration from: {config_file}")
    config = parse_toml_config(config_file)

    # Get required fields from TOML file
    required_keys = (
        "prior_predictive_stan_file",
        "posterior_stan_file",
        "prior_predictive_model_data",
        "posterior_model_data",
        "measured_data",
    )
    inferential_calibration_kwargs = {}
    for key in required_keys:
        if key in config:
            inferential_calibration_kwargs[key] = config[key]
        else:
            raise RuntimeError(f"{key} must be specified.")

    # Get optional fields
    inferential_calibration_kwargs["var_names"] = config.get("var_names", None)
    inferential_calibration_kwargs["measured_data_dtypes"] = config.get(
        "measured_data_dtypes", {}
    )
    inferential_calibration_kwargs["sampling_kwargs"] = config.get(
        "sampling_kwargs", {}
    )
    inferential_calibration_kwargs["diagnostic_check_kwargs"] = config.get(
        "diagnostic_check_kwargs", {}
    )
    inferential_calibration_kwargs["cores"] = config.get("cores", 1)
    inferential_calibration_kwargs["N"] = config.get("N", 400)
    inferential_calibration_kwargs["n_prior_draws_for_sd"] = config.get(
        "n_prior_draws_for_sd", 1000
    )
    inferential_calibration_kwargs["samples_dir"] = config.get(
        "samples_dir", "inferential_calibration_samples"
    )
    inferential_calibration_kwargs["remove_sample_files"] = config.get(
        "remove_sample_files", True
    )
    inferential_calibration_kwargs["progress_bar"] = config.get("progress_bar", False)
    inferential_calibration_kwargs["output_csv"] = config.get(
        "output_csv", "inferential_calibration_results.csv"
    )
    inferential_calibration_kwargs["overwrite"] = config.get("overwrite", False)

    # Run inferential calibration (results written directly to CSV as workers complete)
    click.echo("Running inferential calibration analysis...")
    click.echo(
        f"Results will be written to: {inferential_calibration_kwargs['output_csv']}\n"
    )

    result_df = stan.inferential_calibration(**inferential_calibration_kwargs)

    click.echo(
        f"\ninferential calibration analysis complete! Results saved to: {inferential_calibration_kwargs['output_csv']}"
    )
    click.echo(f"Total rows: {len(result_df)}")


if __name__ == "__main__":
    main()
