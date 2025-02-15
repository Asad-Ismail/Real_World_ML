import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms, models
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained ResNet34 Age Regression on UTKFace")
    parser.add_argument("--train_pct", type=float, default=20.0, help="Percentage of dataset to use for training (0-100)")
    parser.add_argument("--val_pct", type=float, default=5.0, help="Percentage of dataset to use for validation (0-100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable Weights and Biases logging")
    parser.add_argument("--wandb_project", type=str, default="learn_with_less", help="Weights and Biases project name")
    return parser.parse_args()

def prepare_dataset(train_pct, val_pct):
    # Load the UTKFace dataset from Hugging Face
    dataset = load_dataset("utkface")
    # Assumption: the dataset has a single split named "train"
    ds_full = dataset["train"]
    total_size = len(ds_full)
    print(f"Total dataset size: {total_size}")

    # Shuffle and then subset the dataset
    ds_full = ds_full.shuffle(seed=42)
    num_train = int((train_pct / 100) * total_size)
    num_val = int((val_pct / 100) * total_size)

    ds_train = ds_full.select(range(num_train))
    ds_val = ds_full.select(range(num_train, num_train + num_val))

    print(f"Using {len(ds_train)} samples for training and {len(ds_val)} samples for validation")
    return ds_train, ds_val

def get_transforms(train=True):
    # For ResNet, we use standard ImageNet statistics
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def transform_example(example, transform):
    # Apply the transformation to the image
    example["image"] = transform(example["image"])
    # Convert age to a float tensor (vector of size 1) for regression tasks
    example["age"] = torch.tensor([float(example["age"])])
    return example

def create_dataloaders(ds_train, ds_val, batch_size):
    # Get transformations (could later inject augmentation for training)
    transform_train = get_transforms(train=True)
    transform_val = get_transforms(train=False)
    # Set per-example transforms using dataset's with_transform method
    ds_train = ds_train.with_transform(lambda x: transform_example(x, transform_train))
    ds_val = ds_val.with_transform(lambda x: transform_example(x, transform_val))
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_model():
    # Load pretrained ResNet34 and adjust the final layer for regression
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Output a single value
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch["image"].to(device)
        targets = batch["age"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

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

def main():
    args = parse_args()

    # Initialize weights & biases logging if enabled
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Prepare dataset and dataloaders
    ds_train, ds_val = prepare_dataset(args.train_pct, args.val_pct)
    train_loader, val_loader = create_dataloaders(ds_train, ds_val, args.batch_size)

    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()