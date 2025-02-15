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