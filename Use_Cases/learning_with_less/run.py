import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from model import nt_xent_loss, cmixmatch, mixmatch 
from model import semi_supervised_loss, build_model, linear_rampup, build_model_uncertainty
from data import prepare_dataset, create_dataloaders
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained ResNet34 Age Regression on UTKFace")
    parser.add_argument("--train_pct", type=float, default=5.0, help="Percentage of dataset to use for training (0-100)")
    parser.add_argument("--val_pct", type=float, default=3.0, help="Percentage of dataset to use for validation (0-100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable Weights and Biases logging")
    parser.add_argument("--wandb_project", type=str, default="learn_with_lesser", help="Weights and Biases project name")

    parser.add_argument("--mode", type=str, default="supervised", 
                      choices=["supervised", "self_supervised", "semi_supervised"],
                      help="Training mode")
    parser.add_argument("--temperature", type=float, default=0.5, 
                      help="Temperature for contrastive loss")
    parser.add_argument("--lambda_u", type=float, default=10.0, 
                      help="Weight for unsupervised loss in semi-supervised learning")
    parser.add_argument("--semi_supervised_mode", type=str, default="mixmatch", 
                      choices=["mixmatch", "cmixmatch"],
                      help="Semi-supervised learning mode")

    parser.add_argument("--waitepochs", type=int, default=10, 
                      help="Wait epoch before stopping training if val loss does not improves")

    parser.add_argument("--sequential_training", default=False, action="store_true", 
                      help="First do self-supervised then semi-supervised")
    parser.add_argument("--self_supervised_epochs", type=int, default=1,
                      help="Number of self-supervised training epochs")
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, device, args, transform=None, unlabel_loader=None, global_step=0):
    """
    Unified training function that handles all training modes: supervised, self-supervised, and semi-supervised
    
    Args:
        model: The neural network model
        train_loader: DataLoader for labeled training data
        criterion: Loss function (used for supervised and semi-supervised)
        optimizer: The optimizer
        device: Training device (CPU/GPU)
        args: Training arguments including mode and hyperparameters
        transform: Data augmentation transform (for self-supervised and semi-supervised)
        unlabel_loader: DataLoader for unlabeled data (for self-supervised and semi-supervised)
    
    Returns:
        epoch_loss: Average loss for the epoch
        additional_metrics: Dictionary containing any additional metrics
    """
    model.train()
    running_loss = 0.0
    additional_metrics = {}
    num_batches = 0
    grad_clip_value = args.grad_clip_value if hasattr(args, 'grad_clip_value') else 1.0
    
    if args.mode == "supervised":
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs = batch["image"].to(device)
            targets = batch["age"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            global_step += 1

    elif args.mode == "self_supervised":
        # Self-supervised training using contrastive learning
        for batch in tqdm(unlabel_loader, desc="Training", leave=False):
            # Generate two random augmentations
            views1 = [transform(img) for img in batch["image"]]
            views2 = [transform(img) for img in batch["image"]]
            
            view1 = torch.stack(views1).to(device)
            view2 = torch.stack(views2).to(device)
            
            optimizer.zero_grad()
            z1 = model(view1)
            z2 = model(view2)
            
            loss = nt_xent_loss(z1, z2, args.temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            global_step += 1

    elif args.mode == "semi_supervised":
        print(f"Semi-supervised training!!")
        # Semi-supervised training using MixMatch variants
        train_iter = iter(train_loader)
        
        for unlabel_batch in tqdm(unlabel_loader, desc="Training", leave=False):
            # Get next labeled batch
            try:
                labeled_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                labeled_batch = next(train_iter)
            
            x_l, y_l = labeled_batch["original_image"], labeled_batch["age"].to(device)
            u = unlabel_batch["image"]
            
            current_lambda = args.lambda_u * linear_rampup(global_step)
            optimizer.zero_grad()
            
            # select semi-supervised method
            if args.semi_supervised_mode == "mixmatch":
                (labeled_inputs, true_labels), (unlabeled_inputs, guessed_labels) = mixmatch(
                    labeled_batch=(x_l, y_l),
                    unlabeled_batch=u,
                    model=model,
                    augment_fn=transform,
                    T=0.5, K=2, alpha=0.75,
                    mode='regression',
                    device=device
                )
            else:  # cmixmatch
                (labeled_inputs, true_labels), (unlabeled_inputs, guessed_labels) = cmixmatch(
                    labeled_batch=(x_l, y_l),
                    unlabeled_batch=u,
                    model=model,
                    augment_fn=transform,
                    T=0.5, K=2, alpha=0.75,
                    device=device
                )
            
            output_l = model(labeled_inputs)
            output_u = model(unlabeled_inputs)
            
            loss, loss_x, loss_u = semi_supervised_loss(
                labeled_output=output_l,
                labeled_target=true_labels,
                unlabeled_output=output_u,
                unlabeled_target=guessed_labels,
                lambda_u=current_lambda,
                criterion=criterion
            )
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            additional_metrics.update({
                'labeled_loss': loss_x.item(),
                'unlabeled_loss': loss_u.item(),
                'current_lambda': current_lambda
            })
            num_batches += 1
            global_step += 1
    
    epoch_loss = running_loss / num_batches
    return epoch_loss, additional_metrics, global_step


def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            targets = batch["age"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, device, args, transform=None, unlabel_loader=None,vis_val=True):
    """
    Main training loop that uses the unified train_one_epoch function
    """
    best_loss = float('inf')
    global_step = 0
    wait_epoch = args.waitepochs
    epochs_without_improvement = 0  

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        # Training phase
        train_loss, additional_metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            transform=transform,
            unlabel_loader=unlabel_loader,
            global_step=global_step
        )
        
        # Validation phase (skip for self-supervised)
        if args.mode != "self_supervised":
            val_loss = eval_model(model, val_loader, criterion, device)
            current_loss = val_loss
        else:
            current_loss = train_loss
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), f"best_model_{args.mode}.pth")
            epochs_without_improvement = 0  # Reset counter if there's improvement
        else:
            epochs_without_improvement += 1  # Increment counter if no improvement

        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "global_step": global_step
        }
        if vis_val:
            metrics["val_loss"]= current_loss
        metrics.update(additional_metrics)
        
        print(f"Epoch {epoch}/{args.epochs} -> Train Loss: {train_loss:.4f}", end="")
        if args.mode != "self_supervised":
            print(f" | Val Loss: {current_loss:.4f}", end="")
        if args.mode == "semi_supervised":
            print(f" | Current λ: {additional_metrics['current_lambda']:.4f}", end="")
        print("")
        print(f"Best Val loss so far {best_loss:.4f}", end="")
        
        if args.use_wandb:
            wandb.log(metrics)
        
        # Check for early stopping
        if epochs_without_improvement >= wait_epoch:
            print(f"\nEarly stopping triggered after {wait_epoch} epochs without improvement.")
            break

    print(f"Training complete!, best validation loss: {best_loss:.4f}")


def main():
    args = parse_args()

    # Initialize weights & biases logging if enabled
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    ds_train, ds_val, ds_unlabel = prepare_dataset(args.train_pct, args.val_pct)
    train_loader, unlabel_loader, val_loader, augment_transform = create_dataloaders(ds_train, ds_val, ds_unlabel, args.batch_size)

    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If sequential do self supervied followed by semi supervised
    if args.sequential_training:
        print(f"Starting sequential training!!")
        # 1. Self-supervised pretraining
        print("Starting first phase of training with self-supervised pretraining ...")
        self_supervised_model = build_model(self_supervised=True).to(device)
        criterion_ss = nn.MSELoss()  # Not used for contrastive learning
        optimizer_ss = optim.Adam(self_supervised_model.parameters(), lr=args.lr)
        
        # Temporarily change args for self-supervised phase
        original_mode = args.mode
        original_epochs = args.epochs
        args.mode = "self_supervised"
        args.epochs = args.self_supervised_epochs
        
        train_model(
            self_supervised_model, train_loader, val_loader, 
            criterion_ss, optimizer_ss, device, args, 
            augment_transform, unlabel_loader,vis_val=False
        )
        
        # 2. Transfer encoder weights and initialize semi-supervised model
        print(f"Starting second phase of training with pretrained encoder {original_mode} ...")
        # Get the pretrained encoder
        pretrained_encoder = self_supervised_model.encoder
        
        # Build new model for semi-supervised learning
        semi_supervised_model = build_model(model=pretrained_encoder, 
                                         self_supervised=False).to(device)
        
        # Restore original args for semi-supervised phase
        args.mode = original_mode
        args.epochs = original_epochs
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(semi_supervised_model.parameters(), lr=args.lr)
        
        train_model(
            semi_supervised_model, train_loader, val_loader, 
            criterion, optimizer, device, args, 
            augment_transform, unlabel_loader
        )
    else:
        if args.mode == "self_supervised":
            model = build_model(self_supervised=True).to(device)
        else:
            model = build_model_uncertainty(self_supervised=False).to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
        # call train_model
        train_model(model, train_loader, val_loader, criterion, optimizer, device, args, augment_transform, unlabel_loader)

if __name__ == "__main__":
    main()