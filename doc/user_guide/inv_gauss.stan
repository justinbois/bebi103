functions {
    #include inv_gaussian.stanfunctions
}


data {
    int<lower=0> N;
    array[N] real<lower=0> y;
}


parameters {
    real log10_mu;
    real log10_lambda;
}


transformed parameters {
    real mu = 10^log10_mu;
    real lambda = 10^log10_lambda;
}

model {
    log10_mu ~ normal(-2, 2);
    log10_lambda ~ normal(-2, 2);
    y ~ inv_gaussian(mu, lambda);
}


generated quantities {
    array[N] real y_pred;
    for (i in 1:N) {
        y_pred[i] = inv_gaussian_rng(mu, lambda);
    }
}