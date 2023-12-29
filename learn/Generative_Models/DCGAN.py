import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
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
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Set up the loss function and optimizers
criterion = nn.BCELoss()

lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
num_epochs = 50
nz = 100

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train the discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        output = discriminator(real_images).view(-1, 1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        output = discriminator(fake_images.detach()).view(-1, 1)
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # Train the generator
        generator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)  # Using real_labels since we want to minimize the loss
        output = discriminator(fake_images).view(-1, 1)
        errG = criterion(output, real_labels)
        errG.backward()

        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Step [{i}/{len(train_loader)}] | d_loss: {errD.item()}, g_loss: {errG.item()}")

