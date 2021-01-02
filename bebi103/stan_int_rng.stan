data {
  real real_param_1;
  real real_param_2;
  real real_param_3;
  int int_param_1;
  int int_param_2;
  int int_param_3;

  int n;

  int section;
  int subsection;
}

generated quantities {
  int output[n];

  // Bernoulli
  if (section == 50 && subsection == 1) {
    for (i in 1:n) {
      output[i] = bernoulli_rng(real_param_1);
    }
  }

  // Bernoulli logit
  else if (section == 50 && subsection == 2) {
    for (i in 1:n) {
      output[i] = bernoulli_logit_rng(real_param_1);
    }    
  }

}
