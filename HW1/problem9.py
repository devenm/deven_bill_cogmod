import numpy as np
from scipy.stats import multivariate_normal

def multivariate_normal_density(x, mu, Sigma):
    D = len(x)
    det_Sigma = np.linalg.det(Sigma)
    if det_Sigma == 0:
        raise ValueError("Covariance matrix must be non-singular")
    norm_const = 1.0 / ((2*np.pi) ** (D/2) * det_Sigma ** 0.5)
    diff = x - mu
    inv_Sigma = np.linalg.inv(Sigma)
    exponent = -0.5 * np.dot(np.dot(diff, inv_Sigma), diff)
    return norm_const * np.exp(exponent)


#Spherical Gaussian (zero covariance, shared variance across dimensions)
mu_spherical = np.array([0, 0])
Sigma_spherical = np.array([[2, 0],[0, 2]])
x_spherical = np.array([1, 1])
print("Spherical Gaussian - GPT:", multivariate_normal_density(x_spherical, mu_spherical, Sigma_spherical))
print("Spherical Gaussian - SciPy:", multivariate_normal.pdf(x_spherical, mu_spherical, Sigma_spherical))

#Diagonal Gaussian (zero covariance, different variance for each dimension)
mu_diagonal = np.array([0, 0])
Sigma_diagonal = np.array([[2, 0],[0, 3]])
x_diagonal = np.array([1, 1])
print("Diagonal Gaussian - GPT:", multivariate_normal_density(x_diagonal, mu_diagonal, Sigma_diagonal))
print("Diagonal Gaussian - SciPy:", multivariate_normal.pdf(x_diagonal, mu_diagonal, Sigma_diagonal))

#Full-covariance Gaussian (non-zero covariance, different variance for each dimension)
mu_full = np.array([0, 0])
Sigma_full = np.array([[2, 1],[1, 3]])
x_full = np.array([1, 1])
print("Full-covariance Gaussian - GPT:", multivariate_normal_density(x_full, mu_full, Sigma_full))
print("Full-covariance Gaussian - SciPy:", multivariate_normal.pdf(x_full, mu_full, Sigma_full))