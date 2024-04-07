import pandas as pd
import stan
import matplotlib.pyplot as plt
import arviz as az

# Step 1: Data Preparation
# Load your dataset (assuming a CSV file for this example)
data_path = 'problem2_data.csv'  # Update this with the path to your actual data file
df = pd.read_csv(data_path)

# Convert categorical variables to numerical for analysis
# Assuming 'Response' is 'Old' or 'New', and 'Word Type' is 'Old' or 'New'
df['ResponseNum'] = df['Response'].apply(lambda x: 1 if x == 'Old' else 0)
df['WordTypeNum'] = df['Word_Type'].apply(lambda x: 1 if x == 'Old' else 0)

# Prepare data for PyStan
data_for_pystan = {
    'N': len(df),
    'X': df['WordTypeNum'].values,
    'Y': df['ResponseNum'].values
}
print("Data fully prepared.")
# Step 2: Define the PyStan Model for the 1HT model
one_high_threshold_model_code = """
data {
    int<lower=0> N; // Number of observations
    int<lower=0,upper=1> X[N]; // Actual word type (old/new)
    int<lower=0,upper=1> Y[N]; // Participant responses (old/new)
}
parameters {
    real<lower=0,upper=1> theta; // Threshold
}
model {
    for (n in 1:N)
        Y[n] ~ bernoulli_step(X[n], theta); // Decision rule
}
"""

# Step 3: Compile and Fit the 1HT Model
one_high_threshold_model = stan.build(one_high_threshold_model_code,data=data_for_pystan,random_seed=42)
fit_1ht = one_high_threshold_model.sampling(data=data_for_pystan, iter=1000, chains=4)
print('1HT Compiled and fitted.')



