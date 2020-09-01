"""
Generate sample x, y, data of hierarchical structure for bebi103
documentation.
"""

import numpy as np
import pandas as pd


def _generate_sample_data():
    np.random.seed(3252)
    J_1 = 3
    n = np.array([20, 25, 18])
    theta = np.array([3, 7])
    tau = np.array([1, 4])
    sigma = np.array([2, 3])
    rho = 0.6

    sigma_cov = np.array(
        [
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ]
    )

    x = []
    y = []
    trial = []

    for i, n_val in enumerate(n):
        theta_1 = np.random.multivariate_normal(theta, np.diag(tau ** 2))
        x_vals, y_vals = np.random.multivariate_normal(
            theta_1, sigma_cov, size=n_val
        ).transpose()
        x += list(x_vals)
        y += list(y_vals)
        trial += [i + 1] * n_val

    return pd.DataFrame(dict(x=x, y=y, trial=trial))


if __name__ == "__main__":
    df = _generate_sample_data()
    df.to_csv('sample_data.csv', index=False)
