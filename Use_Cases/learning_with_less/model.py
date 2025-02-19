from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.projection(x)
    

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder,input_dim=512):
        super().__init__()
        self.encoder = base_encoder
        # Remove the final fc layer
        self.encoder.fc = nn.Identity()
        self.encoder.fc.in_features = input_dim
        # Add projection head
        self.projection_head = ProjectionHead(input_dim=input_dim)  

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)  

def linear_rampup(current_step, rampup_length=90000):
    """
    Linear ramp-up function for lambda_u parameter
    
    Args:
        current_step: Current training step
        rampup_length: Number of steps over which to ramp up (default: 16000)
    
    Returns:
        rampup_value: Value between 0 and 1 based on current step
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current_step, 0, rampup_length)
        return float(current) / rampup_length   

def sharpen(p, T=0.5):
    """
    Applies temperature sharpening to a probability distribution.
    
    Args:
        p (Tensor): Input tensor of shape (batch_size, num_classes).
        T (float): Temperature parameter. As T → 0, the distribution becomes one-hot.
        
    Returns:
        Tensor: Sharpened probability distribution.
    """
    p_sharpen = p ** (1.0 / T)
    p_sharpen = p_sharpen / torch.sum(p_sharpen, dim=1, keepdim=True)
    return p_sharpen

def compute_pairwise_distances(labels):
    """
    Compute pairwise distances between labels and convert to sampling probabilities.
    
    Args:
        labels (Tensor): Shape [batch_size, 1] for regression values
        
    Returns:
        Tensor: Normalized probability matrix [batch_size, batch_size]
    """
    # Compute pairwise squared distances between labels
    diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [batch_size, batch_size, 1]
    distances = torch.sum(diff ** 2, dim=2)  # [batch_size, batch_size]
    
    # Convert distances to probabilities using Gaussian kernel
    sigma = torch.std(labels) * 0.5  # Adaptive sigma based on label distribution
    probabilities = torch.exp(-distances / (2 * sigma ** 2))
    
    # Zero out self-connections
    mask = 1 - torch.eye(labels.size(0), device=labels.device)
    probabilities = probabilities * mask
    
    # Normalize probabilities per row
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    
    return probabilities

'''
def compute_pairwise_distances(labels, k=5):
    diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=2)
    
    # Compute local sigma for each point based on k-nearest neighbors
    k = min(k, labels.size(0) - 1)
    local_distances, _ = torch.topk(distances, k + 1, largest=False)
    local_sigma = local_distances[:, k].unsqueeze(1)  # Use k-th nearest neighbor
    
    probabilities = torch.exp(-distances / (2 * local_sigma ** 2))
    mask = 1 - torch.eye(labels.size(0), device=labels.device)
    probabilities = probabilities * mask
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    
    return probabilities
'''
def cmixup_data(x1, y1, x2, y2, alpha=0.75, device='cuda'):
    """
    Applies C-Mixup to a pair of examples.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    #lam = max(lam, 1 - lam)  # Ensure x' is closer to x1
    lam = torch.tensor(lam).to(device)
    
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

def mixup_data(x1, y1, x2, y2, alpha=0.75, device='cuda'):
    """
    Applies MixUp to a pair of examples.
    
    For a pair (x1, p1) and (x2, p2), the mixed example is computed as:
      x' = λ₀ * x1 + (1 - λ₀) * x2
      p' = λ₀ * p1 + (1 - λ₀) * p2,
    where λ ~ Beta(α, α) and λ₀ = max(λ, 1 - λ). This extra step ensures that the mixed input 
    is closer to the first sample.
    
    Args:
        x1 (Tensor): First input tensor.
        y1 (Tensor): First label (one-hot for classification or continuous value for regression).
        x2 (Tensor): Second input tensor.
        y2 (Tensor): Second label.
        alpha (float): Beta distribution parameter.
        device (str): Computation device.
                    
    Returns:
        Tuple[Tensor, Tensor]: Mixed input and mixed target.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    #lam = max(lam, 1 - lam)  # Enforce that x' is closer to x1
    lam = torch.tensor(lam).to(device)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


def apply_augmentations(x, num_augmentations=1, augment_fn=None):
    if augment_fn is None:
        if num_augmentations == 1:
            return x
        else:
            return [x for _ in range(num_augmentations)]
    else:
        if  isinstance(x, list):
            # this has to happen for k augmentations
            return [torch.stack([augment_fn(i) for i in x]) for _ in range(num_augmentations) ]
        else:
            raise ValueError("x must be a list")

def cmixmatch(labeled_batch, unlabeled_batch, model, augment_fn=None, T=0.5, K=2, 
              alpha=0.75, device='cuda'):
    """
    Enhanced MixMatch with C-Mixup for regression tasks.
    
    Args:
        labeled_batch: Tuple of (images, targets) where targets are continuous values
        unlabeled_batch: Unlabeled images
        model: Neural network model
        augment_fn: Data augmentation function
        T: Temperature parameter (not used in regression)
        K: Number of augmentations for unlabeled data
        alpha: Beta distribution parameter
        device: Computation device
        normalization_stats: Optional (mean, std) for normalization
    """
    x_l, y_l = labeled_batch
    
    # Augment labeled data once
    x_l_aug = apply_augmentations(x_l, num_augmentations=1, augment_fn=augment_fn)
    x_l_aug = [x.to(device) for x in x_l_aug]
    
    # Process unlabeled data: apply K augmentations
    u = unlabeled_batch
    u_aug_list = apply_augmentations(u, num_augmentations=K, augment_fn=augment_fn)
    u_aug_list = [x.to(device) for x in u_aug_list]

    # Compute predictions for each augmented copy and average them over K augmentations
    predictions = []
    for k in range(K):
        u_aug = u_aug_list[k]
        with torch.no_grad():
            preds = model(u_aug)
            predictions.append(preds)
    
    # Average predictions for unlabeled data
    guessed_labels = torch.mean(torch.stack(predictions, dim=0), dim=0)
    
    # Compute pairwise distances and sampling probabilities for labeled data
    labeled_probs = compute_pairwise_distances(y_l)
    
    # Sample pairs for labeled data using computed probabilities
    batch_size = x_l_aug[0].size(0)
    sampled_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        sampled_indices[i] = torch.multinomial(labeled_probs[i], 1)
    
    # Mix labeled data using C-Mixup
    x_l_shuffled = x_l_aug[0][sampled_indices]
    y_l_shuffled = y_l[sampled_indices]
    mixed_x_l, mixed_y_l = cmixup_data(x_l_aug[0], y_l, x_l_shuffled, y_l_shuffled, 
                                      alpha=alpha, device=device)
    
    # Compute pairwise distances for unlabeled data using guessed labels
    unlabeled_probs = compute_pairwise_distances(guessed_labels)
    
    # Sample pairs for unlabeled data
    batch_size_u = u_aug_list[0].size(0)
    sampled_indices_u = torch.zeros(batch_size_u, dtype=torch.long, device=device)
    for i in range(batch_size_u):
        sampled_indices_u[i] = torch.multinomial(unlabeled_probs[i], 1)
    
    # Mix unlabeled data using C-Mixup
    u_mix = u_aug_list[0].to(device)
    u_shuffled = u_mix[sampled_indices_u]
    q_shuffled = guessed_labels[sampled_indices_u]
    mixed_u, mixed_q = cmixup_data(u_mix, guessed_labels, u_shuffled, q_shuffled, 
                                  alpha=alpha, device=device)
    
    return (mixed_x_l, mixed_y_l), (mixed_u, mixed_q)
    
    #return (x_l_aug[0], y_l), (u_aug_list[0],guessed_labels)


def mixmatch(labeled_batch, unlabeled_batch, model, augment_fn=None, T=0.5, K=2, alpha=0.75, mode='classification', device='cuda'):
    x_l, y_l = labeled_batch
    
    # Augment labeled data once
    x_l_aug = apply_augmentations(x_l, num_augmentations=1, augment_fn=augment_fn)
    x_l_aug = [x.to(device) for x in x_l_aug]
    
    # Process unlabeled data: apply K augmentations
    u = unlabeled_batch
    u_aug_list = apply_augmentations(u, num_augmentations=K, augment_fn=augment_fn)
    u_aug_list = [x.to(device) for x in u_aug_list]

    # Compute predictions for each augmented copy and average them over K augmentations
    predictions = []
    for k in range(K):
        u_aug = u_aug_list[k]
        with torch.no_grad():
            preds = model(u_aug)
            predictions.append(preds)
    preds_stack = torch.stack(predictions, dim=0)
    preds_avg = torch.mean(preds_stack, dim=0)

    # For classification, sharpen the averaged predictions; for regression, use the average directly.
    if mode == 'classification':
        guessed_labels = sharpen(preds_avg, T)
    else:
        guessed_labels = preds_avg
    
    # Use one augmented copy the first one in this case from the unlabeled data for mixing.
    u_mix = u_aug_list[0]
    
    all_inputs = torch.cat([x_l_aug[0], u_mix])
    all_targets = torch.cat([y_l, guessed_labels])
    
    # Shuffle the combined batch
    shuffle_idx = torch.randperm(all_inputs.size(0))
    shuffled_inputs = all_inputs[shuffle_idx]
    shuffled_targets = all_targets[shuffle_idx]

        
    # Split shuffled data for mixing
    n_labeled = x_l_aug[0].size(0)
    
    # MixUp labeled data with first part of shuffled data
    mixed_x_l, mixed_y_l = mixup_data(
        x_l_aug[0], y_l,
        shuffled_inputs[:n_labeled],
        shuffled_targets[:n_labeled],
        alpha=alpha, device=device
    )
    
    # MixUp unlabeled data with remaining shuffled data
    mixed_u, mixed_q = mixup_data(
        u_mix, guessed_labels,
        shuffled_inputs[n_labeled:],
        shuffled_targets[n_labeled:],
        alpha=alpha, device=device
    )
    
    return (mixed_x_l, mixed_y_l), (mixed_u, mixed_q)

def semi_supervised_loss(labeled_output, labeled_target, unlabeled_output, unlabeled_target, lambda_u=100, criterion=nn.MSELoss()):
    """
    Computes the combined semi-supervised loss.
    
    For classification tasks:
      L_X =  L2 or cross-entropy loss between true one-hot labels and model predictions, and
      L_U = L2 (Brier) loss between guessed labels and model predictions on the unlabeled examples.
      
    For regression tasks:
      The mean squared error (MSE) is used for both L_X and L_U.
    
    The total loss is defined by:
      L = L_X + λ_U * L_U
      
    Args:
        labeled_output (Tensor): Model predictions for the labeled batch.
        labeled_target (Tensor): Ground-truth labels (one-hot for classification) for the labeled batch.
        unlabeled_output (Tensor): Model predictions for the unlabeled batch.
        unlabeled_target (Tensor): Guessed labels for the unlabeled batch.
        lambda_u (float): Weight for the unlabeled loss.
        mode (str): 'classification' or 'regression'.
        
    Returns:
        Tuple[Tensor, Tensor, Tensor]: Total loss, labeled loss, and unlabeled loss.
    """

    loss_x = criterion(labeled_output, labeled_target)
    loss_u = criterion(unlabeled_output, unlabeled_target)
    
    total_loss = loss_x + lambda_u * loss_u
    return total_loss, loss_x, loss_u

def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper"""
    batch_size = z1.shape[0]

    # Concatenate representations for all pairs
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                          representations.unsqueeze(0), 
                                          dim=2)
    
    # Remove diagonal similarities (self-similarity)
    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    
    # Remove diagonal elements
    mask = (~torch.eye(2 * batch_size, dtype=bool)).to(z1.device)
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)
    
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temperature
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)
    
    return F.cross_entropy(logits, labels)


def build_model(model=None,self_supervised=False):
    # Load pretrained ResNet34 and replace its final classification layer with a MLP head for regression
    if model is None:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        #model = models.resnet34(weights=None)
    if hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
    else:  
        num_ftrs = model.fc.in_features
    if self_supervised:
        model = SimCLRModel(model,input_dim=num_ftrs)
    else :
        # Add regression head for supervised learning and semi-supervised learning
        mlp_head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        if hasattr(model, 'fc'):
            model.fc = mlp_head
        else:
        # If there's no fc layer, add the mlp_head to the model's output
            model = nn.Sequential(model, mlp_head)
    
    return model