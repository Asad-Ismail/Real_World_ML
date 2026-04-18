import numpy as np
import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask = torch.rand_like(x) > self.p
        return x * mask / (1 - self.p)


class Convolution2D:
    def __init__(self, input_ch: int, output_ch: int, kernel_sz: int, padding: int = 0, stride: int = 1) -> None:
        self.weights = np.random.randn(output_ch, input_ch, kernel_sz, kernel_sz) * 0.1
        self.bias = np.zeros(output_ch)
        self.padding = padding
        self.stride = stride
        self.out_c = output_ch

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )

        batch_size, _, input_h, input_w = x.shape
        kernel_h, kernel_w = self.weights.shape[2:]
        out_h = (input_h - kernel_h) // self.stride + 1
        out_w = (input_w - kernel_w) // self.stride + 1
        out = np.zeros((batch_size, self.out_c, out_h, out_w))

        for out_channel in range(self.out_c):
            kernel = self.weights[out_channel]
            bias = self.bias[out_channel]
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + kernel_h
                    w_start = j * self.stride
                    w_end = w_start + kernel_w
                    window = x[:, :, h_start:h_end, w_start:w_end]
                    out[:, out_channel, i, j] = np.sum(window * kernel[None, ...], axis=(1, 2, 3)) + bias
        return out


def max_pooling(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    batch_size, channels, height, width = x.shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    out = np.zeros((batch_size, channels, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            out[:, :, i, j] = x[:, :, h_start:h_end, w_start:w_end].max(axis=(2, 3))
    return out


class MLP:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        self.W1 = np.random.randn(in_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, out_dim) * 0.1
        self.b2 = np.zeros(out_dim)
        self.cache = {}
        self.grads = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x @ self.W1 + self.b1
        b = np.maximum(0, a)
        c = b @ self.W2 + self.b2
        self.cache = {"x": x, "a": a, "b": b, "c": c}
        return c

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache["x"]
        a = self.cache["a"]
        b = self.cache["b"]

        dW2 = b.T @ dout
        db2 = dout.sum(axis=0)
        db = dout @ self.W2.T
        da = (a > 0) * db
        dW1 = x.T @ da
        db1 = da.sum(axis=0)
        dx = da @ self.W1.T

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return dx

    def train_step(self, x: np.ndarray, target: np.ndarray, lr: float = 1e-2) -> float:
        predictions = self.forward(x)
        loss = np.mean((predictions - target) ** 2)
        dout = 2 * (predictions - target) / target.shape[0]
        self.backward(dout)

        self.W1 -= lr * self.grads["W1"]
        self.b1 -= lr * self.grads["b1"]
        self.W2 -= lr * self.grads["W2"]
        self.b2 -= lr * self.grads["b2"]
        return float(loss)


if __name__ == "__main__":
    sample = np.random.randn(2, 1, 8, 8)
    conv = Convolution2D(input_ch=1, output_ch=2, kernel_sz=3, padding=1, stride=1)
    conv_out = conv.forward(sample)
    pooled = max_pooling(conv_out, kernel_size=2, stride=2)
    print("Convolution output shape:", conv_out.shape)
    print("Pooled output shape:", pooled.shape)

    mlp = MLP(in_dim=4, hidden_dim=8, out_dim=1)
    x = np.random.randn(16, 4)
    y = np.random.randn(16, 1)
    loss = mlp.train_step(x, y)
    print("One MLP training step loss:", loss)
