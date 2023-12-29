## Semi-supervised Learning
## Semi-supervised learning is a machine learning approach that involves training a model on a small amount of labeled data
## supplemented by a larger amount of unlabeled data. The idea is to leverage the unlabeled data to improve the model's performance
## on the labeled task.
## We pass y label in input to encdoer to
## It enables the encoder to learn a more discriminative latent space, which can improve the classification performance.
## It can help in guiding the generative process of the decoder to create more accurate and class-conditional reconstructions of the input data.
'''
For generation, particularly when generating data conditioned on specific classes, having a latent space 
that's informed by class labels ensures that the VAE can generate more distinct and class-accurate samples.
For example, if you have a VAE trained on images of digits (like the MNIST dataset) and you want to generate 
images of the digit "5", a classifier-informed latent space will guide the generator to produce variations of "5s",
as opposed to a more ambiguous set of digits. This is because the encoder has learned to associate certain regions of 
the latent space with the label "5", and the decoder has learned to construct "5s" from those regions.
In the semi-supervised VAE, the classification loss ensures that the generative model does not ignore the class structure inherent in the labeled data.
Thus, it helps in learning a generative process that is better aligned with the true data distribution, improving the fidelity and variety of the generated
samples.
'''
import torch
import torch.nn.functional as F

def compute_reconstruction_loss(x, reconstructed_x):
    # Assuming the data is continuous and using MSE. For binary data, use F.binary_cross_entropy
    return F.mse_loss(reconstructed_x, x, reduction='sum')

def compute_classification_loss(y_true, y_pred_logits):
    # Cross-entropy loss for classification
    return F.cross_entropy(y_pred_logits, y_true, reduction='sum')

def compute_kl_divergence(z_mean, z_log_var):
    # KL divergence between the posterior q(z|x) and the prior p(z)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())



def compute_marginalization_loss(x, encoder, decoder):
    # Loop over all possible classes (assuming y is categorical with a finite set of classes)
    marginalization_loss = 0
    for y in range(num_classes):
        # Convert class label to one-hot encoding
        y_one_hot = F.one_hot(torch.tensor(y), num_classes=num_classes)
        y_one_hot = y_one_hot.float().unsqueeze(0).to(x.device)
        
        # Encode x to get q(z|x)
        z_mean, z_log_var = encoder(x, y_one_hot)
        z = encoder.sample(z_mean, z_log_var)
        
        # Decode z to get p(x|z,y)
        reconstructed_x = decoder(z, y_one_hot)
        
        # Compute the reconstruction loss for this class
        class_reconstruction_loss = compute_reconstruction_loss(x, reconstructed_x)
        
        # Compute the KL divergence
        kl_divergence = compute_kl_divergence(z_mean, z_log_var)
        
        # Combine the reconstruction and KL divergence loss
        class_loss = class_reconstruction_loss + kl_divergence
        
        # Weight the loss by the predicted probability of the class
        # Note: encoder.classify(x) should output class probabilities
        class_prob = torch.sigmoid(encoder.classify(x))[0, y]
        marginalization_loss += class_prob * class_loss
    
    return marginalization_loss


def semi_supervised_vae_objective(x_labeled, y_labeled, x_unlabeled, encoder, decoder, alpha):
    # Encoder provides q(z|x, y) for labeled data
    z_labeled = encoder(x_labeled, y_labeled)
    
    # Decoder provides p(x|z, y) for labeled data
    reconstruction_loss = compute_reconstruction_loss(x_labeled, decoder(z_labeled, y_labeled))
    
    # Classification loss for labeled data
    classification_loss = compute_classification_loss(y_labeled, encoder.classify(x_labeled))
    
    # For unlabeled data, marginalize over y
    marginalization_loss = compute_marginalization_loss(x_unlabeled, encoder, decoder)
    
    # The overall objective combines losses for labeled and unlabeled data
    objective = reconstruction_loss + alpha * classification_loss + marginalization_loss
    return -objective  # We negate the objective because we want to maximize it



class Encoder(nn.Module):
    def __init__(self, input_dim, label_dim, latent_dim):
        super().__init__()
        # Define the network architecture here
        # Typically a few fully connected layers
        # input_dim + label_dim combines data and label information
        self.fc = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output layers for the mean and log variance of z
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        # An additional output layer for classification
        self.classifier = nn.Linear(hidden_dim, label_dim)
    
    def forward(self, x, y=None):
        # If labels are available, concatenate them with the input
        if y is not None:
            x = torch.cat((x, y), dim=1)
        # Otherwise, just use the input
        h = self.fc(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var, self.classifier(h)
    
    def sample(self, z_mean, z_log_var):
        # Reparameterization trick to sample from q(z|x,y)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps*std



class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, output_dim):
        super().__init__()
        # Define the network architecture here
        # Typically a few fully connected layers
        # latent_dim + label_dim combines latent representation and label information
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output layer to reconstruct the input
        self.reconstruction = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, y):
        # Concatenate z and y to inform the reconstruction
        zy = torch.cat((z, y), dim=1)
        h = self.fc(zy)
        return self.reconstruction(h)
