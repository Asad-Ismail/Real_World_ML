import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, nz, n_classes, ngf, nc):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(n_classes, nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), x.size(1), 1, 1)
        c = self.embed(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        x = torch.cat([x, c], 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, n_classes, ndf):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(n_classes, n_classes)
        self.main = nn.Sequential(
            nn.Conv2d(nc + n_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.embed(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], 1)
        return self.main(x).view(-1, 1)

import torch

# Hyperparameters
epochs = 100
lr = 0.0002
beta1 = 0.5
nz = 100
ngf = 64
ndf = 64
nc = 3
n_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the generator and discriminator
netG = Generator(nz, n_classes, ngf, nc).to(device)
netD = Discriminator(nc, n_classes, ndf).to(device)

# Set up the loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        labels = labels.to(device)
        
        # Update the discriminator with real images
        netD.zero_grad()
        output = netD(real_images, labels)
        real_labels = torch.full_like(output, 1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        
        # Update the discriminator with fake images
        noise = torch.randn(real_images.size(0), nz, device=device)
        fake_labels = torch.randint(0, n_classes, (real_images.size(0),), device=device)
        fake_images = netG(noise, fake_labels)
        output = netD(fake_images.detach(), fake_labels)
        fake_labels = torch.full_like(output, 0)
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # Update the generator
        netG.zero_grad()
        output = netD(fake_images, fake_labels)
        real_labels.fill_(1)
        errG = criterion(output, real_labels)
        errG.backward()
        optimizerG.step()
        
        # Print the losses
        if i % 100 == 0:
            print(f"[{epoch}/{epochs}] [{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}")

print("Training completed.")
