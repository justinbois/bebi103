import numpy as np
import bebi103.utils

def sample_prior_predictive(seed=None, index_1=[0]):
    rng = np.random.default_rng(seed=seed)

    J_1 = max(index_1) + 1
    N = len(index_1)

    theta = rng.normal(5, 5, size=2)
    tau = np.abs(rng.normal(0, 10, size=2))
    sigma = np.abs(rng.normal(0, 10, size=2))

    rho = rng.uniform(-1, 1)

    T = np.diag(tau**2)
    Sigma = np.array(
        [
            [sigma[0]**2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1]**2]
        ]
    )
    
    theta_1 = np.array(
        [
            rng.multivariate_normal(theta, T) 
            for i in range(J_1)
        ]
    )

    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        xy = rng.multivariate_normal(theta_1[index_1[i]], Sigma)
        x[i], y[i] = xy

    return dict(
        theta=theta, 
        tau=tau, 
        sigma=sigma, 
        Sigma=Sigma, 
        theta_1=theta_1, 
        x=x, 
        y=y
    )


