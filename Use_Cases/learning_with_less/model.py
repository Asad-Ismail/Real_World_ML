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
        # Add projection head
        self.projection_head = ProjectionHead(input_dim=input_dim)  

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)  # Normalize
    

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

def mixup_data(x1, y1, x2, y2, alpha=0.75, mode='classification', device='cuda'):
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
        mode (str): 'classification' or 'regression'. (Currently, mixing is identical for both, 
                    making it easy to substitute other strategies later.)
        device (str): Computation device.
                    
    Returns:
        Tuple[Tensor, Tensor]: Mixed input and mixed target.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # Enforce that x' is closer to x1
    lam = torch.tensor(lam).to(device)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y



def apply_augmentations(x, num_augmentations=1, augment_fn=None):
    """
    Applies an augmentation function to x multiple times.
    
    Args:
        x (Tensor): Input batch.
        num_augmentations (int): Number of augmented copies to generate.
        augment_fn (callable): Function to apply. If None, the function returns x (or a list of copies of x).
        
    Returns:
        If num_augmentations == 1: returns the augmented tensor.
        If num_augmentations > 1: returns a list of augmented tensors.
    """
    if augment_fn is None:
        if num_augmentations == 1:
            return x
        else:
            return [x for _ in range(num_augmentations)]
    else:
        if num_augmentations == 1:
            return augment_fn(x)
        else:
            return [augment_fn(x) for _ in range(num_augmentations)]

def mixmatch(labeled_batch, unlabeled_batch, model, augment_fn=None, T=0.5, K=2, alpha=0.75, mode='classification', device='cuda'):
    """
    Processes a batch of labeled and unlabeled data using the MixMatch algorithm.
    
    The procedure is as follows:
      1. Augment the labeled batch once.
      2. For the unlabeled batch, perform K rounds of augmentation.
      3. Compute model predictions for each augmentation of unlabelled data, average the predictions, and apply
         a sharpening function (for classification tasks) to obtain guessed labels.
      4. Perform MixUp separately on the augmented labeled examples and the unlabeled examples.
    
    Args:
        labeled_batch (tuple): (x_l, y_l) where x_l are inputs and y_l are one-hot labels (or regression targets).
        unlabeled_batch (Tensor): Batch of unlabeled inputs.
        model (nn.Module): The model used to guess labels for unlabeled data.
        augment_fn (callable): Data augmentation function to apply to the inputs.
        T (float): Sharpening temperature.
        K (int): Number of augmentations for each unlabeled example.
        alpha (float): Beta parameter for the MixUp operation.
        mode (str): 'classification' or 'regression'.
        device (str): Device for computation.
        
    Returns:
        Tuple: Processed labeled batch (mixed_x_l, mixed_y_l) and processed unlabeled batch (mixed_u, mixed_q).
    """
    # set model to evaluation mode for predictions
    model.eval()
    # Unpack and send labeled data to device
    x_l, y_l = labeled_batch
    x_l = x_l.to(device)
    y_l = y_l.to(device)
    
    # Augment labeled data once
    x_l_aug = apply_augmentations(x_l, num_augmentations=1, augment_fn=augment_fn)
    
    # Process unlabeled data: apply K augmentations
    u = unlabeled_batch.to(device)
    u_aug_list = apply_augmentations(u, num_augmentations=K, augment_fn=augment_fn)
    
    # Compute predictions for each augmented copy and average them over K augmentations
    predictions = []
    for k in range(K):
        u_aug = u_aug_list[k]
        with torch.no_grad():
            preds = model(u_aug)
            if mode == 'classification':
                preds = torch.softmax(preds, dim=1)
            predictions.append(preds)
    preds_stack = torch.stack(predictions, dim=0)
    preds_avg = torch.mean(preds_stack, dim=0)
    # set model to training mode for backpropagation
    model.train()
    # For classification, sharpen the averaged predictions; for regression, use the average directly.
    if mode == 'classification':
        guessed_labels = sharpen(preds_avg, T)
    else:
        guessed_labels = preds_avg
    
    # Use one augmented copy (e.g. the first) from the unlabeled data for mixing.
    u_mix = u_aug_list[0]
    
    # Group augmented labeled and unlabeled data
    # X̂ = (x_l_aug, y_l) and Ũ = (u_mix, guessed_labels)
    # Perform MixUp for each group separately.
    
    # MixUp for labeled data
    batch_size = x_l_aug.size(0)
    index_l = torch.randperm(batch_size).to(device)
    x_l_shuffled = x_l_aug[index_l]
    y_l_shuffled = y_l[index_l]
    mixed_x_l, mixed_y_l = mixup_data(x_l_aug, y_l, x_l_shuffled, y_l_shuffled, alpha=alpha, mode=mode, device=device)
    
    # MixUp for unlabeled data
    batch_size_u = u_mix.size(0)
    index_u = torch.randperm(batch_size_u).to(device)
    u_shuffled = u_mix[index_u]
    q_shuffled = guessed_labels[index_u]
    mixed_u, mixed_q = mixup_data(u_mix, guessed_labels, u_shuffled, q_shuffled, alpha=alpha, mode=mode, device=device)
    
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
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
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
        model.fc = mlp_head

    return model