'''
Marginal Likelihood:
In Bayesian model selection, the marginal likelihood (also known as evidence) is
the probability of observing the data under all possible parameter values. 
t's computationally challenging to calculate this directly due to the involved integrals.
Variational Bayes:
A method to approximate the posterior distribution.
It optimizes a family of distributions to be close to the true posterior.
It's used to approximate the evidence lower bound (ELBO), which can be used to approximate the evidence.


Variational Inference (VI):
 VI is a method to approximate the true posterior with a simpler, 
 tractable distribution (e.g., a Gaussian).
The idea is to find the distribution (from a family of simpler distributions) that is closest to the true posterior.

KL Divergence and ELBO:
The closeness between the true posterior and the approximated one is
often measured using the Kullback-Leibler (KL) divergence. 
The objective in VI is to minimize this divergence.
The Evidence Lower BOund (ELBO) is derived from the marginal likelihood of the data,
and maximizing the ELBO is equivalent to minimizing the KL divergence
between the true and approximated posterior.

ELBO Breakdown: The ELBO consists of two terms:

The expected log likelihood of the data. This term encourages the model to fit the data well,
similar to the objective in traditional neural networks.
A regularization term, which is the KL divergence between the approximated posterior and the prior. 
This term regularizes the approximated posterior to be close to the prior.
Mean Squared Loss: This is a specific type of loss function used for regression tasks. 
In traditional (non-Bayesian) neural networks, mean squared loss can be used to train the model. 
However, in the Bayesian context, it's just a part of the overall objective.
 The expected log likelihood term in the ELBO could be a mean squared loss if we're dealing with a
 regression problem with Gaussian noise.
'''



import torch
import torch.nn as nn
import torch.distributions as dist

# Generate synthetic data
n_samples = 100
X = torch.linspace(-3, 3, n_samples).view(-1, 1)
true_weights = torch.tensor([1.5]).view(-1, 1)
y = X @ true_weights + 0.5 * torch.randn(n_samples, 1)

# Bayesian Linear Regression Model
class BayesianLR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLR, self).__init__()
        self.weights_mean = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.weights_logvar = nn.Parameter(torch.zeros(input_dim, output_dim))
        
    def forward(self, x):
        weights_sample = self.weights_mean + torch.exp(0.5 * self.weights_logvar) * torch.randn_like(self.weights_logvar)
        return x @ weights_sample

# Training the Bayesian model using Variational Inference
model = BayesianLR(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    log_likelihood = dist.Normal(output, 0.5).log_prob(y).sum()
    kl_divergence = -0.5 * torch.sum(1 + model.weights_logvar - model.weights_mean.pow(2) - model.weights_logvar.exp())
    loss = kl_divergence - log_likelihood
    loss.backward()
    optimizer.step()

# Bayesian LOO using Importance Sampling
log_p_loo = []
for i in range(n_samples):
    X_loo = torch.cat([X[:i], X[i+1:]])
    y_loo = torch.cat([y[:i], y[i+1:]])
    
    with torch.no_grad():
        output_loo = model(X_loo)
        log_p = dist.Normal(output_loo, 0.5).log_prob(y_loo).sum()
        #For each observation in y_loo, we calculate its log likelihood under the model's predictions output_loo
        log_p_loo.append(log_p.item())

avg_log_p_loo = sum(log_p_loo) / n_samples
print(f"Average Bayesian LOO estimate: {avg_log_p_loo}")


'''
# Bayesian LOO using Importance Sampling
log_p_loo = []

# Number of importance samples
n_importance_samples = 100

for i in range(n_samples):
    X_loo = torch.cat([X[:i], X[i+1:]])
    y_loo = torch.cat([y[:i], y[i+1:]])
    
    importance_weights = []
    log_p_values = []
    
    for _ in range(n_importance_samples):
        with torch.no_grad():
            output_loo = model(X_loo)
            log_p = dist.Normal(output_loo, 0.5).log_prob(y_loo).sum().item()
            log_p_values.append(log_p)
    
    # Calculate importance weights
    max_log_p = max(log_p_values)
    importance_weights = [torch.exp(torch.tensor(lp - max_log_p)) for lp in log_p_values]
    normalized_weights = [w / sum(importance_weights) for w in importance_weights]
    
    # Weighted average using importance weights
    weighted_log_p = sum(w * lp for w, lp in zip(normalized_weights, log_p_values))
    
    log_p_loo.append(weighted_log_p)

avg_log_p_loo = sum(log_p_loo) / n_samples
print(f"Average Bayesian LOO estimate with Importance Sampling: {avg_log_p_loo}")

'''