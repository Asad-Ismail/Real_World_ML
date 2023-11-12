## We will study different kinds of Generative models

### Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs) are a type of generative model that learn to encode data into a lower-dimensional latent space and reconstruct it back to the original space. They are trained to maximize the Evidence Lower Bound (ELBO).

## ELBO (Evidence Lower Bound)

The ELBO is the objective function for VAEs and is composed of two terms:

1. The reconstruction loss, which encourages the decoder to accurately reconstruct the input data.
2. The KL divergence between the approximate posterior and the prior over latent variables, which acts as a regularizer.

## Issues with Standard VAEs

Standard VAEs can sometimes produce blurry images and have a poor latent space representation due to the powerful decoder networks ignoring the latent codes.

## InfoVAE

InfoVAE introduces a mutual information term into the VAE objective to preserve information between the latent variables and the observations, aiming to improve the learned latent representation.

## MMD VAE

MMD VAE replaces the KL divergence with Maximum Mean Discrepancy (MMD), a tractable alternative that measures the distance between the aggregated posterior and the prior.

## Mutual Information Loss

The mutual information loss aims to maximize the information between latent variables and observations, which can be approximated using variational bounds.

## Pseudo Code for Mutual Information Loss

```python
def mutual_information_loss(q_z_given_x, q_z, z_samples):
    # Compute log probabilities
    log_q_z_given_x = q_z_given_x.log_prob(z_samples)
    log_q_z = q_z.log_prob(z_samples)
    
    # Mutual information is the expectation of the difference
    mi_loss = (log_q_z_given_x - log_q_z).mean()
    return mi_loss
```


## β-VAEs
 Introduce a trade-off parameter to balance the latent space capacity and reconstruction quality.
Adversarial Autoencoders: Use a discriminator to enforce a prior on the latent space, leading to an adversarial training scheme.

## MULTIMODAL VAEs

Handling missing modalities


```
import torch
from torch.distributions import Normal

# Define the prior distribution p(z) as a standard Gaussian
prior = Normal(torch.zeros(1), torch.ones(1))

# Expert 1 (e.g., image modality)
expert_1_mean = torch.tensor([0.5])
expert_1_precision = torch.tensor([1.0])  # Precision is the inverse of variance
expert_1 = Normal(expert_1_mean, 1 / expert_1_precision.sqrt())

# Expert 2 (e.g., text modality)
expert_2_mean = torch.tensor([0.2])
expert_2_precision = torch.tensor([1.5])
expert_2 = Normal(expert_2_mean, 1 / expert_2_precision.sqrt())

# Combine the experts by multiplying their densities and normalize to get the posterior
combined_precision = expert_1_precision + expert_2_precision
combined_mean = (expert_1_mean * expert_1_precision + expert_2_mean * expert_2_precision) / combined_precision

# The combined Gaussian representing the posterior q_phi(z|X)
posterior = Normal(combined_mean, 1 / combined_precision.sqrt())

print(f"Combined Posterior Mean: {posterior.mean}")
print(f"Combined Posterior Variance: {posterior.variance}")
```