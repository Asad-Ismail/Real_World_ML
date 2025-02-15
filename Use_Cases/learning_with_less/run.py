import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset, concatenate_datasets
import wandb
from model import nt_xent_loss, mixmatch, semi_supervised_loss, build_model
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained ResNet34 Age Regression on UTKFace")
    parser.add_argument("--train_pct", type=float, default=20.0, help="Percentage of dataset to use for training (0-100)")
    parser.add_argument("--val_pct", type=float, default=5.0, help="Percentage of dataset to use for validation (0-100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable Weights and Biases logging")
    parser.add_argument("--wandb_project", type=str, default="learn_with_less", help="Weights and Biases project name")
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

    print(f"Using {len(ds_train)} samples for training, {len(ds_val)} samples for validation and
          {len(ds_unlabel)} samples for unsupervised learning")
    
    return ds_train, ds_val, ds_unlabel

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
    images = [transform(img) for img in example["image"]]
    # Stack the transformed images into a batch tensor
    images = torch.stack(images)
    # Convert age to a float tensor (vector of size 1) for regression tasks
    ages = torch.tensor(example["age"],dtype=torch.float32).unsqueeze(1)
    return {"image": images, "age": ages}

def create_dataloaders(ds_train, ds_val, ds_unlabel, batch_size):
    # Get transformations (could later inject augmentation for training)
    transform_train = get_transforms(train=True)
    transform_val = get_transforms(train=False)
    ds_train = ds_train.with_transform(lambda x: transform_example(x, transform_train))
    ds_unlabel = ds_unlabel.with_transform(lambda x: transform_example(x, transform_train))
    ds_val = ds_val.with_transform(lambda x: transform_example(x, transform_val))
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    unlabel_loader = DataLoader(ds_unlabel, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return train_loader, unlabel_loader, val_loader


def train_self_supervised(model, batch, optimizer, device, temperature,transform):
    # Get two random augmentations of the same images
    views = [transform(img) for img in batch["image"]]
    view1 = torch.stack(views).to(device)
    views = [transform(img) for img in batch["image"]]
    view2 = torch.stack(views).to(device)
    
    optimizer.zero_grad()
    z1 = model(view1)
    z2 = model(view2)
    
    loss = nt_xent_loss(z1, z2, temperature)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    return loss.item()

def train_semisupervised(model, supervised_batch, unsupervised_batch, criterion, optimizer, device):

    x_l, y_l = supervised_batch["image"], supervised_batch["age"]
    u = unsupervised_batch["image"]#

    (labeled_inputs, true_labels), (unlabeled_inputs, guessed_labels) = mixmatch(
        labeled_batch=(x_l, y_l),
        unlabeled_batch=u,
        model=model,
        augment_fn=your_augmentation_fn,  
        T=0.5, K=2, alpha=0.75,
        mode='regression',            
        device=device
    )
    # Compute predictions for the processed data
    output_l = model(labeled_inputs)
    output_u = model(unlabeled_inputs)

    # Compute the combined loss
    loss, loss_x, loss_u = semi_supervised_loss(
        labeled_output=output_l,
        labeled_target=true_labels,
        unlabeled_output=output_u,
        unlabeled_target=guessed_labels,
        lambda_u=100,
        criterion=criterion
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_supervised(model, batch, criterion, optimizer, device):
    inputs = batch["image"].to(device)
    targets = batch["age"].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        loss = train_supervised(model, batch, criterion, optimizer, device)
        running_loss += loss.item() 
    epoch_loss = running_loss / len(dataloader)
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

    ds_train, ds_val, ds_unlabel = prepare_dataset(args.train_pct, args.val_pct)
    train_loader, unlabel_loader, val_loader = create_dataloaders(ds_train, ds_val, ds_unlabel, args.batch_size)

    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training",total=args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_model(model, val_loader, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_pretrained.pth")

        print(f"Epoch {epoch}/{args.epochs} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    
    if args.use_wandb:
        wandb.finish()
    print(f"Training complete!, best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()