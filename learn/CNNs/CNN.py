
import torch

def naive_Convolution(x:torch.Tensor, k:torch.Tensor):
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = k.shape

    # Calculate output dimensions
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1

    # Create output tensor
    out = torch.zeros((batch_size, out_channels, out_height, out_width))

    # Perform convolution
    for i in range(out_height):
        for j in range(out_width):
            x_slice = x[:, :, i:i+kernel_height, j:j+kernel_width]
            for c in range(out_channels):
                out[:, c, i, j] = (x_slice * k[c]).sum(dim=(1, 2, 3))

    return out


def img2col_Convolution(x:torch.Tensor, k:torch.Tensor):
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = k.shape
    x = torch.nn.functional.pad(x, (0, kernel_width - 1, 0, kernel_height - 1))
    x_unf = x.unfold(2, kernel_height, 1).unfold(3, kernel_width, 1)
    x_unf = x_unf.contiguous().view(batch_size, in_channels, -1, kernel_height, kernel_width)

    out = (x_unf * k.view(1, out_channels, 1, kernel_height, kernel_width)).sum(dim=(1, 2, 3, 4))
    return out

if __name__=="__main__":
    x = torch.randn(1, 3, 5, 5)
    k = torch.randn(2, 3, 3, 3)
    print(naive_Convolution(x, k))