import pandas as pd
import stan
import matplotlib.pyplot as plt
import arviz as az


data_path = 'problem2_data.csv'  
df = pd.read_csv(data_path)

df['ResponseNum'] = df['Response'].apply(lambda x: 1 if x == 'Old' else 0)
df['WordTypeNum'] = df['Word_Type'].apply(lambda x: 1 if x == 'Old' else 0)


data_for_pystan = {
    'N': len(df),
    'X': df['WordTypeNum'].values,
    'Y': df['ResponseNum'].values
}
print("Data fully prepared.")


one_high_threshold_model_code = """
data {
    int<lower=0> N; //observations
    array[N] int<lower=0,upper=1> X; //actual word type (old/new)
    array[N] int<lower=0,upper=1> Y; //participant responses (old/new)
}
parameters {
    real<lower=0, upper=1> d_raw; 
    real<lower=0, upper=1> g_raw; 
}
transformed parameters {
    real<lower=0, upper=1> d = inv_logit(d_raw); //transformed parameter for d
    real<lower=0, upper=1> g = inv_logit(g_raw); //transformed parameter for g
}
model {
    //priors for d_raw and g_raw
    d_raw ~ normal(0, 1);
    g_raw ~ normal(0, 1); 
    
    //Likelihood
    for (n in 1:N)
        Y[n] ~ bernoulli_logit(X[n] ? (logit(d) + ((1 - logit(d)) * logit(g)) + logit(g)) : ((1 - logit(d)) * (1 - logit(g)) + (1 - logit(g)))); //Decision rules
}
"""

one_ht_posterior = stan.build(one_high_threshold_model_code,data=data_for_pystan,random_seed=42)
fit_1ht = one_ht_posterior.sample(num_chains=4, num_samples=1000)
print('1HT Compiled and fitted.')

two_high_threshold_model_code = """
data {
    int<lower=0> N; 
    array[N] int<lower=0,upper=1> X;
    array[N] int<lower=0,upper=1> Y; 
}
parameters {
    real<lower=0, upper=1> d_raw;
    real<lower=0, upper=1> g_raw; 
}
transformed parameters {
    real<lower=0, upper=1> d = inv_logit(d_raw); 
    real<lower=0, upper=1> g = inv_logit(g_raw); 
}
model {
    d_raw ~ normal(0, 1); // Normal prior for d_raw
    g_raw ~ normal(0, 1); // Normal prior for g_raw

    for (n in 1:N)
        Y[n] ~ bernoulli_logit(X[n] ? (logit(d) + ((1 - logit(d)) * logit(g)) + ( logit(g)) * (1-logit(d)) ) : ((1 - logit(d)) * (1 - logit(g)) + (logit(d)))); //updated decision rule for 2ht
}
"""

two_ht_posterior = stan.build(two_high_threshold_model_code,data=data_for_pystan,random_seed=42)
fit_2ht = one_ht_posterior.sample(num_chains=4, num_samples=1000)
print('2HT Compiled and fitted.')




print("Convergence diagnostics for 1HT model:")
az.plot_trace(fit_1ht)
plt.show()
print("Estimation results for 1HT model:")
print(az.summary(fit_1ht))
print("Convergence diagnostics for 2HT model:")
az.plot_trace(fit_2ht)
plt.show()
print("Estimation results for 2HT model:")
print(az.summary(fit_2ht))