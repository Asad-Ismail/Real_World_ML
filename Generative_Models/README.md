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

## VAE Posterior Collapse

## Posterior Collapse
Posterior collapse occurs when the variational posterior `q_ϕ(z|x)` becomes identical to the prior `p_θ(z)`, often a standard Gaussian distribution `N(0, I)`. When this happens, the KL divergence term `D_KL(q_ϕ(z|x) || p_θ(z))` in the Evidence Lower Bound (ELBO) becomes zero. This might seem like an optimal scenario, but it's actually problematic. It means that the model ignores the latent variables `z` when generating the data, which defeats the purpose of having a latent space to capture meaningful representations of the data.

In other words, the decoder `p_θ(x|z)` becomes too powerful and learns to reconstruct the data without relying on the latent variables `z`. As a result, the latent variables `z` fail to capture any useful information about the data, and the VAE does not learn a useful latent representation.

## KL Annealing
KL annealing is a technique used to prevent posterior collapse. The idea is to gradually increase the weight of the KL divergence term in the ELBO loss function during training. Initially, this weight (denoted as `β`) is set to zero, which means that the VAE is trained to only maximize the likelihood of the data, similar to a standard autoencoder. Gradually, `β` is increased towards 1, which brings the model closer to the VAE objective, where both the likelihood of the data and the KL divergence are considered.

By slowly increasing `β`, the model starts by learning good reconstructions and then gradually begins to take into account the structure of the latent space. This helps in learning a more balanced model where the latent space captures meaningful information while still providing good reconstructions.

## Cyclical Annealing
Cyclical annealing is an extension of KL annealing where the process of increasing `β` is repeated multiple times in cycles. Each cycle starts with `β` close to 0 and increases to 1 throughout the cycle. The idea is that each cycle uses the latent representations learned in previous cycles to warm-start the optimization. This repeated cycling can lead to a more stable learning of meaningful latent representations as it prevents the model from settling too quickly into a local optimum where the posterior collapses.

The text suggests that cyclical annealing can be more effective than simple KL annealing because it allows the model to refine its understanding of the latent space over multiple cycles, potentially leading to a richer and more informative latent representation.

In summary, KL annealing and cyclical annealing are techniques to address the problem of posterior collapse in VAEs, helping to ensure that the latent space encodes meaningful information about the data.

The key is that we want the posterior to be similar to the prior but not identical. If the posterior is identical to the prior (i.e., the KL divergence is zero), the latent variable z does not contain any information about the input 
x and is not utilized effectively. This is the scenario of posterior collapse, where the latent variables are not used, and the model essentially becomes a standard autoencoder.
On the other hand, if the posterior is too different from the prior (i.e., the KL divergence is large), the model may overfit to the training data, and the latent space may not generalize well to unseen data.
The goal, therefore, is to find a sweet spot where the posterior is close enough to the prior to benefit from its regularizing effect (thus ensuring a well-formed and general latent space) but still contains sufficient information about the input data to be useful for reconstruction. This balancing act is typically achieved by carefully tuning the VAE objective function, potentially using techniques like KL annealing to dynamically adjust the emphasis on the KL divergence term during training.


## Hierarichal VAEs
In a hierarchical VAE, there are multiple layers of latent variables arranged in a hierarchy. The higher layers tend to capture more abstract, global features of the data, while the lower layers capture more detailed, specific features.

"Conditionally independent given the higher layer" means that once you know the value of the latent variables in the higher layer, the latent variables in the lower layer are independent of each other. The dependencies between lower-layer variables are only through their common 'ancestor' variables in the higher layer.

Here's an example to illustrate this concept:

Imagine a hierarchical model trying to generate an image of a face.
At the top layer of the hierarchy, one latent variable might represent the presence of a face versus no face.
The next layer might have latent variables for different parts of the face: one for the eyes, one for the nose, and one for the mouth.
Given the top layer (the presence of a face), the positions and shapes of the eyes, nose, and mouth are independent of each other. The eye variable doesn't directly influence the mouth variable; it only does so indirectly through the higher-level face variable.
This conditional independence allows the model to factorize the complex joint distribution of data into simpler distributions that are easier to manage and model. Each branch of the "tree" in the hierarchy operates independently of the others, given the state of the parent nodes, and this is what gives the hierarchical structure a tree-like form.

In practice, this allows for more efficient computation and learning, as each set of lower-layer variables can be dealt with separately, parallelizing parts of the inference and generation processes. It also means that the model can learn a rich representation of the data, capturing complex dependencies without needing an excessively large number of parameters.


## PixelCNN Mask Types

PixelCNN models generate images pixel by pixel in an autoregressive manner. To ensure the model respects the autoregressive property, different mask types are applied to the convolutional layers:

### Mask Type A
- **Used in the First Layer**: Ensures that the prediction for each pixel does not include information from itself or any future pixels.
- **Enforces Causality**: The first layer predicts the value of each pixel based solely on the pixels that have already been observed.

### Mask Type B
- **Used in Subsequent Layers**: After the first layer, allows the model to use the current pixel's value for predicting itself.
- **Conditioning on Current Pixel**: Subsequent layers can condition on the current pixel's value, capturing the context more accurately.

