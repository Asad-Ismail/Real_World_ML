import logging
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


LOGGER = logging.getLogger(__name__)
IMAGE_SIZE = 64


class SupervisedImageRegressionDataset(Dataset):
    def __init__(self, images: Sequence[Image.Image], targets: Sequence[float], transform):
        self.images = list(images)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        original_image = self.images[index].copy()
        transformed_image = self.transform(original_image)
        target = torch.tensor([self.targets[index]], dtype=torch.float32)
        return {
            "image": transformed_image,
            "age": target,
            "original_image": original_image,
        }


class UnsupervisedImageDataset(Dataset):
    def __init__(self, images: Sequence[Image.Image]):
        self.images = list(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return {"image": self.images[index].copy()}


def _digits_to_pil_images() -> tuple[list[Image.Image], np.ndarray]:
    dataset = load_digits()
    images = []
    for image in dataset.images:
        image = (image / image.max()) * 255.0
        pil_image = Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
        images.append(pil_image)
    targets = dataset.target.astype(np.float32)
    return images, targets


def _load_utkface_examples() -> tuple[list[Image.Image], np.ndarray]:
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The optional 'datasets' package is required for the UTK-Face example."
        ) from error

    dataset = load_dataset("deedax/UTK-Face-Revised")
    combined = concatenate_datasets([dataset["train"], dataset["valid"]]).shuffle(seed=42)
    images = [example["image"].convert("RGB") for example in combined]
    targets = np.asarray([float(example["age"]) for example in combined], dtype=np.float32)
    return images, targets


def _load_examples(dataset_source: str) -> tuple[list[Image.Image], np.ndarray, str]:
    if dataset_source == "digits":
        images, targets = _digits_to_pil_images()
        return images, targets, "digits"

    if dataset_source == "utkface":
        images, targets = _load_utkface_examples()
        return images, targets, "utkface"

    try:
        images, targets = _load_utkface_examples()
        LOGGER.info("Loaded UTK-Face dataset for the limited-data example.")
        return images, targets, "utkface"
    except Exception as error:
        LOGGER.warning(
            "Falling back to the built-in digits dataset because UTK-Face was unavailable: %s",
            error,
        )
        images, targets = _digits_to_pil_images()
        return images, targets, "digits"


def prepare_dataset(
    train_pct: float,
    val_pct: float,
    dataset_source: str = "auto",
    seed: int = 42,
):
    if train_pct <= 0 or val_pct <= 0:
        raise ValueError("train_pct and val_pct must both be greater than zero.")
    if train_pct + val_pct >= 100:
        raise ValueError("train_pct + val_pct must be less than 100.")

    images, targets, resolved_source = _load_examples(dataset_source)
    total_size = len(images)
    print(f"Resolved dataset source: {resolved_source}")
    print(f"Total dataset size: {total_size}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_size)

    num_train = max(1, int((train_pct / 100.0) * total_size))
    num_val = max(1, int((val_pct / 100.0) * total_size))

    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]
    unlabel_idx = indices[num_train + num_val :]

    train_images = [images[idx] for idx in train_idx]
    val_images = [images[idx] for idx in val_idx]
    unlabel_images = [images[idx] for idx in unlabel_idx]
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]

    print(
        "Using "
        f"{len(train_images)} labeled train, "
        f"{len(val_images)} validation, and "
        f"{len(unlabel_images)} unlabeled samples"
    )

    return (train_images, train_targets), (val_images, val_targets), unlabel_images


def get_transforms(train: bool = True):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if not train:
        return eval_transform, None

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=18, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, augment_transform


def collate_supervised(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "original_image": [item["original_image"] for item in batch],
        "age": torch.stack([item["age"] for item in batch]),
    }


def collate_unsupervised(batch):
    return {"image": [sample["image"] for sample in batch]}


def create_dataloaders(ds_train, ds_val, ds_unlabel, batch_size: int):
    train_images, train_targets = ds_train
    val_images, val_targets = ds_val

    transform_train, augment_transform = get_transforms(train=True)
    transform_val, _ = get_transforms(train=False)

    train_dataset = SupervisedImageRegressionDataset(train_images, train_targets, transform_train)
    val_dataset = SupervisedImageRegressionDataset(val_images, val_targets, transform_val)
    unlabel_dataset = UnsupervisedImageDataset(ds_unlabel)

    drop_last_train = len(train_dataset) >= batch_size
    drop_last_unlabel = len(unlabel_dataset) >= batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        collate_fn=collate_supervised,
    )
    unlabel_loader = DataLoader(
        unlabel_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_unlabel,
        collate_fn=collate_unsupervised,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_supervised,
    )

    return train_loader, unlabel_loader, val_loader, augment_transform
