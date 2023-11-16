## Also used to train VQVAE top level latent code to sample new samples

class MaskedConv2d(nn.Conv2d):
    # Masked convolution
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kernel_height, kernel_width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, kernel_height // 2, kernel_width // 2:] = 0
            self.mask[:, :, kernel_height // 2 + 1:] = 0
        elif mask_type == 'B':
            self.mask[:, :, kernel_height // 2, kernel_width // 2 + 1:] = 0
            self.mask[:, :, kernel_height // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, input_shape, num_layers, num_classes):
        super(PixelCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Define the first layer as Mask A type
        self.layers.append(MaskedConv2d('A', in_channels=input_shape[0], 
                                        out_channels=num_classes, kernel_size=7, padding=3))
        
        # Add subsequent layers as Mask B type
        for _ in range(num_layers - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(MaskedConv2d('B', in_channels=num_classes, 
                                            out_channels=num_classes, kernel_size=7, padding=3))
        
        # Final layer without masking to output logits
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=num_classes, 
                                     out_channels=input_shape[0], kernel_size=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Instantiate the PixelCNN model
num_layers = 12  # Define the number of layers in the PixelCNN
num_classes = 256  # Define the number of classes (size of the codebook)
input_shape = (1, 28, 28)  # Define the shape of the input (e.g., MNIST images)
pixelcnn = PixelCNN(input_shape, num_layers, num_classes)
