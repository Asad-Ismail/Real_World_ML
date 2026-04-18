import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int = 1024, model_dim: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim)
        )

        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


if __name__ == "__main__":
    encoder = SinusoidalPositionalEncoding(max_len=16, model_dim=8)
    sample = torch.zeros(2, 5, 8)
    encoded = encoder(sample)
    print("Encoded shape:", encoded.shape)
    print(encoded[0, :2])
