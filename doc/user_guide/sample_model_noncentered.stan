data {
  // Total number of data points
  int N;

  // Number of entries in each level of the hierarchy
  int J_1;

  //Index array to keep track of hierarchical structure
  array[N] int index_1;

  // The measurements
  array[N] real x;
  array[N] real y;
}


transformed data {
  // Data are two-dimensional, so store in a vector
  array[N] vector[2] xy;
  for (i in 1:N) {
    xy[i, 1] = x[i];
    xy[i, 2] = y[i];
  }
}


parameters {
  // Hyperparameters level 0
  vector[2] theta;

  // How hyperparameters vary
  vector<lower=0>[2] tau;

  // Parameters
  vector<lower=0>[2] sigma;
  real<lower=-1, upper=1> rho;

  // Noncentered parameters
  array[J_1] vector[2] theta_1_noncentered;
}


transformed parameters {
  // Covariance matrix for likelihood
  matrix[2, 2] Sigma = [
    [sigma[1]^2,                 rho * sigma[1] * sigma[2]], 
    [rho * sigma[1] * sigma[2],  sigma[2]^2               ]
  ];

  // Center parameters
  array[J_1] vector[2] theta_1;
  for (i in 1:J_1) {
    theta_1[i] = theta + tau .* theta_1_noncentered[i]; 
  }
}


model {
  // Hyperpriors
  theta ~ normal(5, 5);
  tau ~ normal(0, 10);

  // Priors
  theta_1_noncentered ~ multi_normal([0, 0], [[1, 0], [0, 1]]);
  sigma ~ normal(0, 10);
  rho ~ uniform(-1, 1);

  // Likelihood
  for (i in 1:N) {
    xy[i] ~ multi_normal(theta_1[index_1[i]], Sigma);
  }
}


generated quantities {
  array[N] real x_ppc;
  array[N] real y_ppc;
  array[N] real log_lik;

  {
    vector[2] xy_ppc;

    for (i in 1:N) {
      xy_ppc = multi_normal_rng(theta_1[index_1[i]], Sigma);
      log_lik[i] = multi_normal_lpdf(xy_ppc | theta_1[index_1[i]], Sigma);
      x_ppc[i] = xy_ppc[1];
      y_ppc[i] = xy_ppc[2];
    }
  }
}