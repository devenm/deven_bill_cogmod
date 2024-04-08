import stan


stan_code = '''
data {
  int<lower=0> N; //trials
  array[N] int<lower=0> C_obs;
  array[N] int<lower=0> S_obs;
  array[N] int<lower=0> F_obs;
  array[N] int<lower=0> M_obs;
  array[N] int<lower=0> U_obs;
  array[N] int<lower=0> N_obs;
  array[N] int<lower=0> AN_obs;
  array[N] int<lower=0> NA_obs;
}

parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=1> b;
  real<lower=0, upper=1> c;
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> e;
  real<lower=0, upper=1> f;
  real<lower=0, upper=1> g;
  real<lower=0, upper=1> h;
}

transformed parameters {
  //derived equations for probabilities
  real C = a*b*c*d*e*f;
  real S = a*b*(1-c)*f;
  real F = (a*b*c)*((1-d)*(1-f)*g + (1-d)*f + d*(1-e)*(1-f)*(1-g) + d*e*(1-f)*g);
  real M = a*b*c*d*(1-e)*f;
  real U = a*(1-b)*f + a*(1-b)*(1-f)*h + a*b*(1-c)*(1-f)*h;
  real N = (a*b*c)*((1-d)*(1-f)*(1-g) + d*(1-e)*(1-f)*(1-g) + d*e*(1-f)*(1-g));
  real AN = a*(1-b)*(1-f)*(1-h) + a*b*(1-c)*(1-f)*(1-h);
  real NA = 1-a;
}

model {
  //priors
  a ~ beta(1, 1);
  b ~ beta(1, 1);
  c ~ beta(1, 1);
  d ~ beta(1, 1);
  e ~ beta(1, 1);
  f ~ beta(1, 1);
  g ~ beta(1, 1);
  h ~ beta(1, 1);

  //Liklihood
  for (i in 1:N) {
    //MPT likelihood
    target += multinomial_lpmf([C_obs[i], S_obs[i], F_obs[i], M_obs[i], U_obs[i], N_obs[i], AN_obs[i], NA_obs[i]]' | 
                                [C, S, F, M, U, N, AN, NA]);
  }
}

generated quantities {
  //gen frequency data
  array[N] int<lower=0> C_gen;
  array[N] int<lower=0> S_gen;
  array[N] int<lower=0> F_gen;
  array[N] int<lower=0> M_gen;
  array[N] int<lower=0> U_gen;
  array[N] int<lower=0> N_gen;
  array[N] int<lower=0> AN_gen;
  array[N] int<lower=0> NA_gen;
  
  for (i in 1:N) {
    //take multinomial distribution
    {
      vector[8] probs = [C, S, F, M, U, N, AN, NA]';
      int counts[8];
      counts = multinomial_rng(probs, sum([C_obs[i], S_obs[i], F_obs[i], M_obs[i], U_obs[i], N_obs[i], AN_obs[i], NA_obs[i]]));
      C_gen[i] = counts[1];
      S_gen[i] = counts[2];
      F_gen[i] = counts[3];
      M_gen[i] = counts[4];
      U_gen[i] = counts[5];
      N_gen[i] = counts[6];
      AN_gen[i] = counts[7];
      NA_gen[i] = counts[8];
    }
  }
}
'''