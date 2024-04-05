import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class VDVAEEncoder(nn.Module):
    def __init__(self, num_levels, input_channels=3):
        super(VDVAEEncoder, self).__init__()
        self.res_blocks = nn.ModuleList()
        self.mean_blocks = nn.ModuleList()
        self.logvar_blocks = nn.ModuleList()
        
        channels = [input_channels] + [64 * 2**i for i in range(num_levels)]  # Increasing channels

        for i in range(num_levels):
            self.res_blocks.append(ResBlock(channels[i], channels[i+1]))
            # Assuming a flattened and pooled representation for the mean and logvar
            self.mean_blocks.append(nn.Linear(channels[i+1] * 4 * 4, 256))  # Example dimension
            self.logvar_blocks.append(nn.Linear(channels[i+1] * 4 * 4, 256))

    def forward(self, x):
        latents = []
        for i, res_block in enumerate(self.res_blocks):
            x = F.avg_pool2d(res_block(x), 2)  # Downsample
            x_flat = torch.flatten(x, 1)
            mean = self.mean_blocks[i](x_flat)
            logvar = self.logvar_blocks[i](x_flat)
            latents.append((mean, logvar))
        return latents

class VDVAEDecoder(nn.Module):
    def __init__(self, num_levels):
        super(VDVAEDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # Upsampling
        self.res_blocks = nn.ModuleList()
        
        # In reverse order compared to the encoder
        channels = [256] + [64 * 2**(num_levels-i-1) for i in range(num_levels)]

        for i in range(num_levels):
            self.res_blocks.append(ResBlock(channels[i], channels[i+1]))

    def forward(self, latents):
        x = self.sample(latents[-1])  # Starting with the lowest level latent
        for i in reversed(range(len(self.res_blocks))):
            x = self.upsample(x)
            x = self.res_blocks[i](x)
            if i > 0:  # Combine current level's latent info, avoiding index error
                x += self.sample(latents[i-1])
        return x
    
    def sample(self, latent):
        mean, logvar = latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class VDVAE(nn.Module):
    def __init__(self, num_levels):
        super(VDVAE, self).__init__()
        self.encoder = VDVAEEncoder(num_levels)
        self.decoder = VDVAEDecoder(num_levels)
    
    def forward(self, x):
        latents = self.encoder(x)
        recon_x = self.decoder(latents)
        return recon_x


input_img = torch.randn(1, 1, 28, 28)  
num_levels = 3  
vdvae = VDVAE(num_levels)
reconstructed_img = vdvae(input_img)