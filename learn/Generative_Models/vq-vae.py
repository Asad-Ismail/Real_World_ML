import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
import matplotlib.pyplot as plt
from datasets import get_cifar10_loaders

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize the embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1)
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.reshape(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight through estimator for gradient flow as argmin is non differentiable
        #In forward pass: quantized = quantized
        #In backward pass: gradient flows through inputs as if we did nothing
        #This tricks the network into learning despite the non-differentiable operation
        
        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2)
        return quantized, loss, encoding_indices.reshape(input_shape[:-1])

class Encoder(nn.Module):
    def __init__(self, backbone_name='resnet18', embedding_dim=64):
        super().__init__()
        # Load pretrained backbone from timm
        self.backbone = timm.create_model(backbone_name, features_only=True, pretrained=True)
        
        # Get the output channels of the last layer
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        last_channels = features[-1].shape[1]
        
        # Add final convolution to get desired embedding dimension
        self.final_conv = nn.Conv2d(last_channels, embedding_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.final_conv(features[-1])

class Decoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, 
                 num_embeddings=512,
                 embedding_dim=64,
                 backbone_name='resnet18',
                 commitment_cost=0.25):
        super().__init__()
        
        self.encoder = Encoder(backbone_name=backbone_name, embedding_dim=embedding_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.vector_quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, encoding_indices

    def encode(self, x):
        z = self.encoder(x)
        _, _, encoding_indices = self.vector_quantizer(z)
        return encoding_indices

    def decode(self, encoding_indices):
        # Convert indices to one-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], 
                              self.vector_quantizer.num_embeddings,
                              encoding_indices.shape[1],
                              encoding_indices.shape[2],
                              device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Multiply with embedding weights
        quantized = torch.matmul(encodings.permute(0, 2, 3, 1), 
                                self.vector_quantizer.embedding.weight)
        quantized = quantized.permute(0, 3, 1, 2)
        
        return self.decoder(quantized)

def train_step(model, optimizer, images, device):
    images = images.to(device)
    optimizer.zero_grad()
    # Forward pass
    reconstructions, vq_loss, _ = model(images)
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructions, images)
    # Total loss
    total_loss = recon_loss + vq_loss
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), recon_loss.item(), vq_loss.item()

def train_vqvae(model, train_loader, num_epochs, device, learning_rate=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            loss, recon_loss, vq_loss = train_step(model, optimizer, images, device)
            total_train_loss += loss
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}, '
                      f'Recon Loss: {recon_loss:.4f}, VQ Loss: {vq_loss:.4f}')
        
        avg_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

def inference_step(model, images, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        reconstructions, _, encoding_indices = model(images)
        # Get discrete codes
        codes = encoding_indices.cpu().numpy()
        return reconstructions.cpu(), codes

def visualize_results(original, reconstruction, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        # Show original
        axes[0, i].imshow(original[i].permute(1, 2, 0).clip(0, 1))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        # Show reconstruction
        axes[1, i].imshow(reconstruction[i].permute(1, 2, 0).clip(0, 1))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose dataset 
    print(f"Getting Dataset!!")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    # train_loader, test_loader = get_fashion_mnist_loaders(BATCH_SIZE)
    
    # Initialize smaller model for toy dataset
    model = VQVAE(
        num_embeddings=256,  # Reduced from 512
        embedding_dim=32,    # Reduced from 64
        backbone_name='resnet18',
        commitment_cost=0.25
    )
    
    # Training
    train_vqvae(model, train_loader, NUM_EPOCHS, DEVICE)
    
    # Test reconstruction on test set
    test_images, _ = next(iter(test_loader))
    reconstructions, codes = inference_step(model, test_images, DEVICE)
    visualize_results(test_images, reconstructions)