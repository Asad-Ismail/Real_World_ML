'''
Bayesian model selection:
This method leverages Bayes' rule to determine the most probable model given the data. This involves computing the marginal likelihood of the data under each model, which evaluates the average likelihood of the data across all possible parameter values, weighted by the prior.

The concept of marginal likelihood (or evidence) is introduced. It's essentially the probability of the observed data across all possible parameter values. This is important as it helps to weigh models not just on how well they fit the data, but also on their complexity. This is crucial to prevent overfitting.
5. Bayesian Occam's razor:
The term refers to the idea that given two models, the simpler model is favored unless the complex model provides a substantially better fit to the data. This is a regularization effect embedded in the Bayesian framework.



'''

import numpy as np
from scipy.stats import beta

# Assuming the outcomes are encoded as 0 (tails) and 1 (heads)
outcomes = np.array([0, 1, 0, 1, 1])  # example outcomes
N = len(outcomes)
heads = np.sum(outcomes)
tails = N - heads

# Prior for biased coin (e.g., Beta distribution)
alpha_prior, beta_prior = 1, 1  # Uniform prior

# Marginal likelihood for fair coin model
p_data_given_M0 = 0.5 ** N

# Marginal likelihood for biased coin model (using Beta-binomial conjugacy)
p_data_given_M1 = beta.pdf(0.5, heads + alpha_prior, tails + beta_prior) / beta.pdf(0.5, alpha_prior, beta_prior)

# Model comparison (greater value indicates better model)
print(f"Marginal Likelihood for M0: {p_data_given_M0}")
print(f"Marginal Likelihood for M1: {p_data_given_M1}")

# You'd then compare these values to determine which model is more likely given the data.
