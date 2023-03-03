import torch
import torch.nn as nn
import torchvision

# Define the transformation to apply to each image
transform = transforms.Compose(
    [transforms.ToTensor()]
    ) 

# Load the training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

# Define the autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        # Non Linearity that makes autoencoder very powerful
        x = torch.relu(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

# Instantiate the autoencoder
input_size = 784 # MNIST images are 28x28, so the input size is 784
hidden_size = 64
model = Autoencoder(input_size, hidden_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 10
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        recon = model(img)
        loss = criterion(recon, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
