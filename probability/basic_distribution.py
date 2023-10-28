# Bernoulli Distribution:
# x: Binary outcome (0 or 1 for failure or success respectively).

# Binomial Distribution:
# x: Number of successes in N trials. E.g., x=5 in N=10 means 5 successes.

# Categorical Distribution:
# x: One-hot encoded vector for an outcome. E.g., 3 outcomes & 2nd occurs => x = [0, 1, 0].

# Multinomial Distribution:
# x: Vector with each element indicating the number of times a specific outcome occurs in N trials.
# E.g., 10 trials, 3 outcomes, occurrences: 2, 3, 5 => x = [2, 3, 5].

# Poisson Distribution:
# x: Number of events in a fixed interval (time or space).

# Negative Binomial Distribution:
# x: Number of failures until the rth success.

# Gaussian (Normal) Distribution:
# x: Real-valued outcome. Function gives probability density for a mean (μ) and standard deviation (σ).

# Half-Normal Distribution:
# y: Non-negative real-valued outcome. Function gives probability density for a standard deviation (σ).

# Student's t-Distribution:
# x: Real-valued outcome. Function returns probability density for ν degrees of freedom.

# Note:
# Discrete Distributions: Specific outcomes or counts (Bernoulli, Binomial, Categorical, Multinomial, Poisson, Negative Binomial).
# Continuous Distributions: Real-valued outcomes, returning densities (Gaussian, Half-Normal, Student's t).


import numpy as np
from scipy.stats import binom, bernoulli, poisson, nbinom, norm, t
from scipy.special import comb

# Bernoulli Distribution
def bernoulli_distribution(p, x):
    return bernoulli.pmf(x, p)

# Binomial Distribution
def binomial_distribution(N, p, x):
    return binom.pmf(x, N, p)

# Categorical Distribution (assuming x is one-hot encoded)
def categorical_distribution(theta, x):
    return np.prod(theta**x)

# Multinomial Distribution
def multinomial_distribution(N, theta, x):
    coefficient = comb(N, *x)
    return coefficient * np.prod(theta**x)

# Poisson Distribution
def poisson_distribution(lmbda, x):
    return poisson.pmf(x, lmbda)

# Negative Binomial Distribution
def negative_binomial_distribution(r, p, x):
    return nbinom.pmf(x, r, p)

# Gaussian (Normal) Distribution
def gaussian_distribution(mu, sigma, x):
    return norm.pdf(x, mu, sigma)

# Half-Normal Distribution
def half_normal_distribution(sigma, y):
    if y < 0:
        return 0
    return (np.sqrt(2) / (sigma * np.sqrt(np.pi))) * np.exp(-y**2 / (2 * sigma**2))

# Student's t-Distribution
def student_t_distribution(nu, x):
    return t.pdf(x, nu)

def gaussian_pdf(x, mu, sigma2):
    coefficient = 1.0 / np.sqrt(2 * np.pi * sigma2)
    exponential_term = np.exp(- (x - mu)**2 / (2 * sigma2))
    return coefficient * exponential_term



if __name__ == "__main__":
    # You can test the functions here
    print(bernoulli_distribution(0.5, 1))
    print(binomial_distribution(10, 0.5, 5))
    print(categorical_distribution(np.array([0.2, 0.3, 0.5]), np.array([0, 1, 0])))
    print(multinomial_distribution(10, np.array([0.2, 0.3, 0.5]), np.array([2, 3, 5])))
    print(poisson_distribution(5, 3))
    print(negative_binomial_distribution(3, 0.5, 3))
    print(gaussian_distribution(0, 1, 0))
    print(half_normal_distribution(1, 0.5))
    print(student_t_distribution(2, 0.5))
    mu = 0
    sigma2 = 1
    x = 0
    print(gaussian_pdf(x, mu, sigma2)) 
