import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset, concatenate_datasets

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
        #transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        
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