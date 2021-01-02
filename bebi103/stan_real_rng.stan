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
  real output[n];

  // Normal
  if (section == 54 && subsection == 1) {
    for (i in 1:n) {
      output[i] = normal_rng(real_param_1, real_param_2);
    }    
  }

}
