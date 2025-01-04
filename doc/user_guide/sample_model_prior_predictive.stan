data {
  // Total number of data points
  int N;

  // Number of entries in each level of the hierarchy
  int J_1;

  //Index array to keep track of hierarchical structure
  array[N] int index_1;
}


generated quantities {
  vector[2] theta;
  vector[2] tau;
  vector[2] sigma;

  for (i in 1:2) {
    theta[i] = normal_rng(5, 5);
    tau[i] = abs(normal_rng(0, 10));
    sigma[i] = abs(normal_rng(0, 10));
  }

  real rho = uniform_rng(-1, 1);

  array[J_1] vector[2] theta_1;
  for (i in 1:J_1) {
    for (j in 1:2) {
      theta_1[i, j] = normal_rng(theta[j], tau[j]);
    }
  }

  array[N] real x;
  array[N] real y;

  {
    matrix[2, 2] Sigma = [
      [sigma[1]^2,                 rho * sigma[1] * sigma[2]], 
      [rho * sigma[1] * sigma[2],  sigma[2]^2               ]
    ];

    vector[2] xy;

    for (i in 1:N) {
      xy = multi_normal_rng(theta_1[index_1[i]], Sigma);
      x[i] = xy[1];
      y[i] = xy[2];
    }
  }
}
