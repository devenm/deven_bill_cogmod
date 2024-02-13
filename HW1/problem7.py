import numpy as np
import matplotlib.pyplot as plt

def calculate_posterior(prior, sensitivity, specificity):
    # Bayes' theorem: P(A|B) = (P(B|A) * P(A)) / P(B)
    # where A: Disease present, B: Positive test
    # P(A|B): Posterior probability, P(B|A): Sensitivity, P(A): Prior probability
    # P(B): Total probability of a positive test, P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)

    # Prior probability of not having the disease
    prior_not_disease = 1 - prior
    # Total probability of a positive test
    total_positive_prob = (sensitivity * prior) + (1 - specificity) * prior_not_disease
    # Posterior probability calculation
    posterior = (sensitivity * prior) / total_positive_prob
    return posterior

# Scenario 1: Varying prior probability, fixed sensitivity and specificity
priors = np.linspace(0.01, 1, 100)
sensitivity = 0.95
specificity = 0.90
posteriors_prior = [calculate_posterior(prior, sensitivity, specificity) for prior in priors]

plt.figure(figsize=(10, 6))
plt.plot(priors, posteriors_prior)
plt.title('Posterior Probability vs. Prior Probability')
plt.xlabel('Prior Probability')
plt.ylabel('Posterior Probability')
plt.grid(True)
plt.show()

#2: Varying sensitivity, fixed prior and specificity
sensitivities = np.linspace(0.5, 1, 100)
prior = 0.01
posteriors_sensitivity = [calculate_posterior(prior, sensitivity, specificity) for sensitivity in sensitivities]

plt.figure(figsize=(10, 6))
plt.plot(sensitivities, posteriors_sensitivity)
plt.title('Posterior Probability vs. Sensitivity')
plt.xlabel('Sensitivity')
plt.ylabel('Posterior Probability')
plt.grid(True)
plt.show()

#3: Varying specificity, fixed prior and sensitivity
specificities = np.linspace(0.5, 1, 100)
posteriors_specificity = [calculate_posterior(prior, sensitivity, specificity) for specificity in specificities]

plt.figure(figsize=(10, 6))
plt.plot(specificities, posteriors_specificity)
plt.title('Posterior Probability vs. Specificity')
plt.xlabel('Specificity')
plt.ylabel('Posterior Probability')
plt.grid(True)
plt.show()