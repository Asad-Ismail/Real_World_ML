# MLE vs MAP in Python

# Maximum Likelihood Estimation (MLE):
# - Estimates model parameters by maximizing the likelihood of observed data.
# - Formula: θ_MLE = argmax_θ P(data | θ)
# - Does not incorporate prior information about the parameters.
# - Focuses solely on observed data to estimate parameters.

# Maximum a Posteriori (MAP) Estimation:
# - Estimates model parameters by maximizing the posterior distribution.
# - Formula: θ_MAP = argmax_θ P(θ | data) = argmax_θ P(data | θ) * P(θ)
# - Incorporates prior information about the parameters using a prior distribution P(θ).
# - Balances information from observed data and prior beliefs.

# Example: Estimating the probability of a coin landing heads.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Sample data: 1 for heads, 0 for tails
data = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])

# MLE estimation for the probability of getting heads:
def mle(data):
    return np.mean(data)  # Average of data gives probability of heads

theta_mle = mle(data)
print(f"MLE Estimate for Theta (probability of heads): {theta_mle:.2f}")

# MAP estimation for the probability of getting heads, with a prior belief:
def map_estimate(data, prior_heads, prior_tails):
    # Number of observed heads and tails
    observed_heads = np.sum(data)
    observed_tails = len(data) - observed_heads
    
    # MAP estimate combines observed data with prior beliefs
    return (observed_heads + prior_heads) / (observed_heads + observed_tails + prior_heads + prior_tails)

# Assuming a prior belief: 6 heads observed in 10 trials in past experiments
theta_map = map_estimate(data, prior_heads=6, prior_tails=4)
print(f"MAP Estimate for Theta (probability of heads): {theta_map:.2f}")


# Generating some sample data: y = 2x + noise
x = np.linspace(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.1, size=x.shape)
x_tensor = torch.FloatTensor(x).view(-1, 1)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# A simple feed-forward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.fc(x)

# MLE training
def train_mle(model, data, targets, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Using Mean Squared Error for regression
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# MAP training (incorporating L2 regularization as a prior)
def train_map(model, data, targets, epochs=1000, lr=0.01, weight_decay=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Training a model using MLE
model_mle = SimpleNN()
train_mle(model_mle, x_tensor, y_tensor)

# Training another model using MAP
model_map = SimpleNN()
train_map(model_map, x_tensor, y_tensor)

# Checking parameters after training
print("Weights after MLE training:", model_mle.fc.weight.item())
print("Weights after MAP training:", model_map.fc.weight.item())

