import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features, 1, 1))
        self.shift = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.scale + self.shift
    
class LayerNorm(nn.Module):
    def __init__(self, features_dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features_dim))
        self.shift = nn.Parameter(torch.zeros(features_dim))
        self.eps = eps

    def forward(self, x):
        # normalize across feature dimension(s), per sample
        mean = x.mean(dim=(-1), keepdim=True)
        var = x.var(dim=(-1), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.scale + self.shift
    


import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        # normalize across spatial dimensions (H, W), per channel, per sample
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.scale + self.bias




if __name__ == "__main__":
    # Input of shape B,C,H,W
    B=2
    C=3
    H=224
    W=224
    S=128

    #x= torch.randn(B, C, H, W)
    #print(f"Input shape {x.shape}")
    #bn = BatchNorm2d(C)
    #y = bn(x)
    #print(f"Output shape {y.shape}")


    x= torch.randn(B, C, S)
    print(f"Input shape {x.shape}")
    ln = LayerNorm(C)
    y = ln(x)
    print(f"Output shape {y.shape}")