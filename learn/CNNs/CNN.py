
import torch
import torch.nn.functional as F

def naive_convolution(x: torch.Tensor, k: torch.Tensor):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = k.shape

    # Symmetric padding for SAME conv
    pad_h = (kernel_height - 1) // 2, (kernel_height - 1) - (kernel_height - 1) // 2
    pad_w = (kernel_width - 1) // 2, (kernel_width - 1) - (kernel_width - 1) // 2

    # PyTorch pad order = (left, right, top, bottom)
    x = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))

    out_height, out_width = height, width
    out = torch.zeros((batch_size, out_channels, out_height, out_width))

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
    # Symmetric padding for SAME conv
    pad_h = (kernel_height - 1) // 2, (kernel_height - 1) - (kernel_height - 1) // 2
    pad_w = (kernel_width - 1) // 2, (kernel_width - 1) - (kernel_width - 1) // 2

    # PyTorch pad order = (left, right, top, bottom)
    x = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))
    x_unf = x.unfold(2, kernel_height, 1).unfold(3, kernel_width, 1)
    x_unf = x_unf.permute(0,2,3,1,4,5)
    x_unf = x_unf.contiguous().view(batch_size, -1, in_channels*kernel_height*kernel_width)
    k = k.view(out_channels, -1)

    out = (x_unf @ k.T).permute(0,2,1).view(batch_size, out_channels, height, width)
    return out

if __name__=="__main__":
    x = torch.randn(1, 3, 5, 5)
    k = torch.randn(11, 3, 3, 3)
    naive_res = naive_convolution(x, k) 
    img2col_res = img2col_Convolution(x, k)

    assert naive_res.shape == (1,11, 5, 5)
    assert img2col_res.shape == (1, 11, 5, 5)
    assert torch.allclose(naive_res, img2col_res, atol=1e-6)
    print("All tests passed!")