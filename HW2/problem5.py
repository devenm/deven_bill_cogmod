import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#true parameters for weights of dogs in the park
true_mean = 50   
true_stddev = 9  
#sim data
np.random.seed(42)
sample_size = 100
data = np.random.normal(true_mean, true_stddev, sample_size)
#priors
prior_mean = 55
prior_stddev = 8
#posterior parameters
likelihood_stddev = 7
posterior_stddev = 1 / np.sqrt(1 / prior_stddev**2 + sample_size / likelihood_stddev**2)
posterior_mean = (prior_mean / prior_stddev**2 + np.sum(data) / likelihood_stddev**2) * (1 / (1 / prior_stddev**2 + sample_size / likelihood_stddev**2))
print(posterior_mean,posterior_stddev)
#visualization
plt.figure(figsize=(12, 6))
#prior
plt.subplot(1, 2, 1)
plt.hist(data, bins=20, density=True, alpha=0.5, label='Data')
x_prior = np.linspace(30, 70, 1000)
plt.plot(x_prior, norm.pdf(x_prior, prior_mean, prior_stddev), 'r', label='Prior')
plt.title('Prior and Data')
plt.xlabel('Weight(lbs)')
plt.ylabel('Density')
plt.legend()
#posterior
plt.subplot(1, 2, 2)
plt.hist(data, bins=20, density=True, alpha=0.5, label='Data')
x_posterior = np.linspace(30, 70, 1000)
plt.plot(x_posterior, norm.pdf(x_posterior, posterior_mean, posterior_stddev), 'g', label='Posterior')
plt.title('Posterior and Data')
plt.xlabel('Weight(lbs)')
plt.ylabel('Density')
plt.legend()
#combine and display
plt.tight_layout()
plt.show()