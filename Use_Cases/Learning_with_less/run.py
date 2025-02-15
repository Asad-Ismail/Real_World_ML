import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def prepare_dataset(train_pct, val_pct):
    # Import concatenate_datasets from datasets library
    from datasets import load_dataset, concatenate_datasets
    
    # Load the UTKFace dataset from Hugging Face
    dataset = load_dataset("deedax/UTK-Face-Revised")
    
    # Load both train and validation splits
    ds_train = dataset["train"]
    ds_val = dataset["valid"]
    
    # Combine using concatenate_datasets
    ds_full = concatenate_datasets([ds_train, ds_val])
    
    # Shuffle and then subset the dataset
    ds_full = ds_full.shuffle(seed=42)
    total_size = len(ds_full)
    
    # Calculate sizes for new train/val splits based on percentages
    num_train = int((train_pct / 100) * total_size)
    num_val = int((val_pct / 100) * total_size)
    
    # Create new train/val splits
    ds_train = ds_full.select(range(num_train))
    ds_val = ds_full.select(range(num_train, num_train + num_val))
    
    print(f"Total dataset size: {total_size}")
    print(f"Using {len(ds_train)} samples for training and {len(ds_val)} samples for validation")
    
    return ds_train, ds_val 

def transform_example(example, transform):
    # Handle list of PIL images
    images = [transform(img) for img in example["image"]]
    # Stack the transformed images into a batch tensor
    images = torch.stack(images)
    
    # Convert age values to float tensor
    ages = torch.tensor([float(age) for age in example["age"]], dtype=torch.float32)
    
    return {
        "image": images,
        "age": ages
    }

def create_dataloaders(ds_train, ds_val, batch_size):
    # Get transformations (could later inject augmentation for training)
    transform_train = get_transforms(train=True)
    transform_val = get_transforms(train=False)
    
    # Set per-example transforms using dataset's with_transform method
    ds_train = ds_train.with_transform(lambda x: transform_example(x, transform_train))
    ds_val = ds_val.with_transform(lambda x: transform_example(x, transform_val))
    
    # DataLoader will handle batching after the transforms
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader 

def get_transforms(train=True):
    # For ResNet, we use standard ImageNet statistics
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]) 