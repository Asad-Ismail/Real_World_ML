import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(32),  # Native CIFAR size
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_fashion_mnist_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize to match CIFAR
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
