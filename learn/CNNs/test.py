

import torch.nn as nn
import numpy as np


class dropout(nn.Module):
    def __init__(self,p=0.2):
        super().__init__()
        self.p=p

    def forward(self,x):
        mask=torch.rand_like(x)>self.p
        x=x*mask/(1-self.p)
        return x


class convolutions:

    def __init__(self,input_ch,output_ch,kernel_sz,padding,stride) -> None:
        self.weights = np.random.randn(output_ch, input_ch, kernel_sz, kernel_sz)
        self.bias = np.zeros(1,output_ch)
        self.padding = padding
        self.stride = stride
        self.input_shape = input_ch
        self.out_c = output_ch

    def forward(self,x):
        b, cin,in_h, in_w = x.shape
        out_h = (in_h - self.weights.shape[2] + 2 * self.padding) // self.stride + 1
        out_w = (in_w - self.weights.shape[3] + 2 * self.padding) // self.stride + 1
        out = np.zeros((b, self.weights.shape[0], out_h, out_w))
        for c in range(self.out_c):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.weights.shape[2]
                    w_start = j * self.stride
                    w_end = w_start + self.weights.shape[3]
                    out[:, c, i, j] = (x[:, :, h_start:h_end, w_start:w_end]*self.weights[c,...].unsqeueeze(0)).sum(axes=(1,2,3)  + self.bias
        return out


def max_pooling(x, kernel_size, stride):
    b, c, h, w = x.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    out = np.zeros((b, c, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size
            out[:, :, i, j] = x[:, :, h_start:h_end, w_start:w_end].max(axis=(2, 3))
    return out



class mlp:;
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.W1 =np.random.randn(in_dim, hidden_dim)
        self.b1 =np.random.randn(hidden_dim)
        self.W2 =np.random.randn(hidden_dim, out_dim)
        self.b2 =np.random.randn(out_dim)

    def forward(self, x):
       # x has shape (batch_size, in_dim)
        a=x @ self.W1 + self.b1
        b=np.maximum(0,x)
        c=x @ self.W2 + self.b2
        self.cache={'a':a, 'b':b, 'c':c}
        return x

    def backward(self, dout):
        x, a, b, c = self.cache["x"], self.cache["a"], self.cache["b"], self.cache["c"]

        # Grad wrt W2 and b2
        dW2 = b.T @ dout                 # (H, D_out)
        db2 = dout.sum(axis=0)           # (D_out,)

        # Grad wrt hidden activations b
        db = dout @ self.W2.T            # (B, H)

        # Backprop through ReLU
        da = (a > 0) * db                 # (B, H)

        # Grad wrt W1 and b1
        dW1 = x.T @ da                   # (D_in, H)
        db1 = da.sum(axis=0)             # (H,)

        # Grad wrt input x
        dx = da @ self.W1.T              # (B, D_in)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return dx

    def train(self,x,target):
        y=self.forward(x)
        loss = 1/2*((y-target)**2)
        # target and y has shape Bxdim
        dloss = y-target

        dout = self.loss.backward()
        self.backward(dout)