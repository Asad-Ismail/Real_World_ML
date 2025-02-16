import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset, concatenate_datasets
import wandb
from model import nt_xent_loss, cmixmatch, mixmatch 
from model import semi_supervised_loss, build_model, linear_rampup
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained ResNet34 Age Regression on UTKFace")
    parser.add_argument("--train_pct", type=float, default=20.0, help="Percentage of dataset to use for training (0-100)")
    parser.add_argument("--val_pct", type=float, default=7.0, help="Percentage of dataset to use for validation (0-100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable Weights and Biases logging")
    parser.add_argument("--wandb_project", type=str, default="learn_with_less", help="Weights and Biases project name")

    parser.add_argument("--mode", type=str, default="supervised", 
                      choices=["supervised", "self_supervised", "semi_supervised"],
                      help="Training mode")
    parser.add_argument("--temperature", type=float, default=0.5, 
                      help="Temperature for contrastive loss")
    parser.add_argument("--lambda_u", type=float, default=100.0, 
                      help="Weight for unsupervised loss in semi-supervised learning")
    parser.add_argument("--semi_supervised_mode", type=str, default="cmixmatch", 
                      choices=["mixmatch", "cmixmatch"],
                      help="Semi-supervised learning mode")
    
    return parser.parse_args()

def prepare_dataset(train_pct, val_pct):
    dataset = load_dataset("deedax/UTK-Face-Revised")
    ds_train = dataset["train"]
    ds_val = dataset["valid"]

    ds_full = concatenate_datasets([ds_train, ds_val])
    total_size = len(ds_full)
    print(f"Total dataset size: {total_size}")

    # Shuffle and then subset the dataset
    ds_full = ds_full.shuffle(seed=42)
    num_train = int((train_pct / 100) * total_size)
    num_val = int((val_pct / 100) * total_size)

    ds_train = ds_full.select(range(num_train))
    ds_val = ds_full.select(range(num_train, num_train + num_val))

    ds_unlabel = ds_full.select(range(num_train + num_val, total_size))

    print(f"Using {len(ds_train)} samples for training, {len(ds_val)} samples for validation and {len(ds_unlabel)} samples for unsupervised learning")
    
    return ds_train, ds_val, ds_unlabel

def get_transforms(train=True):
    """Returns two types of transforms: basic and augmented"""
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    if train:
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return basic_transform, augment_transform
    return basic_transform, None


def transform_supervised(example, transform):
    # Apply the transformation to the image
    original_images = example["image"]
    images = [transform(img) for img in example["image"]]
    # Stack the transformed images into a batch tensor
    images = torch.stack(images)
    # Convert age to a float tensor (vector of size 1) for regression tasks
    ages = torch.tensor(example["age"],dtype=torch.float32).unsqueeze(1)
    return {"image": images, "age": ages, "original_image": original_images}



def collate_supervised(batch):
    """
    Collate function for supervised data that handles both transformed and original images
    """
    collated = {
        "image": torch.stack([item["image"] for item in batch]),
        "original_image": [sample["original_image"] for sample in batch],
        "age": torch.stack([item["age"] for item in batch])
    }
    return collated

def collate_unsupervised(batch):
    """
    Collate function for unsupervised data that preserves original images
    """
    collated = {
        "image": [sample["image"] for sample in batch]
    }
    return collated



def create_dataloaders(ds_train, ds_val, ds_unlabel, batch_size):
    # Get transformations (could later inject augmentation for training)
    transform_train, augment_transform = get_transforms(train=True)
    transform_val, _ = get_transforms(train=False)

    ds_train = ds_train.with_transform(lambda x: transform_supervised(x, transform_train))
    ds_val = ds_val.with_transform(lambda x: transform_supervised(x, transform_val))


    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_supervised)
    unlabel_loader = DataLoader(ds_unlabel, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_unsupervised)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,collate_fn=collate_supervised)

    return train_loader, unlabel_loader, val_loader, augment_transform


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

    if args.mode == "supervised":
        print("Supervised training")
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
        print("Self-supervised training")
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



def train_model(model, train_loader, val_loader, criterion, optimizer, device, args, transform=None, unlabel_loader=None):
    """
    Main training loop that uses the unified train_one_epoch function
    """
    best_loss = float('inf')
    global_step = 0

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
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "current_loss": current_loss,
            "global_step": global_step
        }
        metrics.update(additional_metrics)
        
        print(f"Epoch {epoch}/{args.epochs} -> Train Loss: {train_loss:.4f}", end="")
        if args.mode != "self_supervised":
            print(f" | Val Loss: {current_loss:.4f}", end="")
        if args.mode == "semi_supervised":
            print(f" | Current λ: {additional_metrics['current_lambda']:.4f}", end="")
        print("")
            
        if args.use_wandb:
            wandb.log(metrics)
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
    
    if args.mode == "self_supervised":
        model = build_model(self_supervised=True).to(device)
    else:
        model = build_model(self_supervised=False).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # call train_model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args, augment_transform, unlabel_loader)

if __name__ == "__main__":
    main()