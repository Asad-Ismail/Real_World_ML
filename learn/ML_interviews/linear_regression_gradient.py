import torch
import numpy as np

X_np = np.array([[1, 1], [1, 2], [1, 3]])
y_np = np.array([1, 2, 3]).reshape(-1, 1)


X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

m, n = X.shape

theta = torch.zeros((n, 1), requires_grad=True, dtype=torch.float32)

# Forward pass
y_hat = X @ theta
loss = (1/m) * torch.sum((y_hat - y)**2)

grad_torch = torch.autograd.grad(loss, theta)[0]
print(grad_torch)

grad_np= 2/m*(X.T @ (y_hat.detach().numpy()-y.numpy()))
print(grad_np)

assert np.allclose(grad_torch.numpy(), grad_np)