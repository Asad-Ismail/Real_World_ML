import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
import time
from collections import defaultdict
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image

class SyntheticSegmentationDataset(Dataset):
    """Generate synthetic segmentation data for fast iterations"""
    
    def __init__(self, size=1000, image_size=(512, 512), num_classes=10, transform=None):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random colorful image
        image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        
        # Generate segmentation mask with geometric shapes
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        # Add random geometric shapes
        for _ in range(np.random.randint(3, 8)):
            class_id = np.random.randint(1, self.num_classes)
            center = (np.random.randint(50, self.image_size[1]-50), 
                     np.random.randint(50, self.image_size[0]-50))
            
            shape_type = np.random.choice(['circle', 'rectangle'])
            
            if shape_type == 'circle':
                radius = np.random.randint(20, 80)
                y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
                circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                mask[circle_mask] = class_id
            else:  # rectangle
                w, h = np.random.randint(40, 120), np.random.randint(40, 120)
                x1, y1 = max(0, center[0]-w//2), max(0, center[1]-h//2)
                x2, y2 = min(self.image_size[1], center[0]+w//2), min(self.image_size[0], center[1]+h//2)
                mask[y1:y2, x1:x2] = class_id
        
        # Convert to PIL Images
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask

class ResNet50Segmentation(nn.Module):
    """ResNet-50 based segmentation model with FCN-style head"""
    
    def __init__(self, num_classes=10):
        super(ResNet50Segmentation, self).__init__()
        
        # Load pretrained ResNet-50 and remove the final layers
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Simple decoder (FCN-style)
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Extract features using ResNet-50 backbone
        features = self.backbone(x)
        
        # Apply decoder
        out = self.decoder(features)
        
        # Upsample to original size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out

class SegmentationMetrics:
    """Calculate segmentation metrics including IoU and pixel accuracy"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """Update metrics with batch predictions and targets"""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if t < self.num_classes and p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_iou(self):
        """Compute mean IoU and per-class IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - intersection)
        
        iou = intersection / (union + 1e-8)
        mean_iou = np.nanmean(iou[1:])  # Exclude background class
        return mean_iou, iou
    
    def compute_pixel_accuracy(self):
        """Compute pixel accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-8)

def create_data_loaders(batch_size=8, num_workers=2):
    """Create synthetic train and validation data loaders"""
    
    # Simple transforms for synthetic data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SyntheticSegmentationDataset(
        size=800,  # Small for fast iterations
        image_size=(256, 256),  # Smaller image size for speed
        num_classes=10,
        transform=transform
    )
    
    val_dataset = SyntheticSegmentationDataset(
        size=200,
        image_size=(256, 256),
        num_classes=10,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device, metrics):
    """Train for one epoch"""
    model.train()
    metrics.reset()
    
    running_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        pred = torch.argmax(outputs, dim=1)
        metrics.update(pred, targets)
        
        running_loss += loss.item()
        
        # Print progress every 20 batches
        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Calculate metrics
    mean_iou, _ = metrics.compute_iou()
    pixel_acc = metrics.compute_pixel_accuracy()
    avg_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time
    
    return {
        'loss': avg_loss,
        'mean_iou': mean_iou,
        'pixel_accuracy': pixel_acc,
        'time': epoch_time
    }

def validate_epoch(model, val_loader, criterion, device, metrics):
    """Validate for one epoch"""
    model.eval()
    metrics.reset()
    
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            pred = torch.argmax(outputs, dim=1)
            metrics.update(pred, targets)
            
            running_loss += loss.item()
    
    # Calculate metrics
    mean_iou, _ = metrics.compute_iou()
    pixel_acc = metrics.compute_pixel_accuracy()
    avg_loss = running_loss / len(val_loader)
    
    return {
        'loss': avg_loss,
        'mean_iou': mean_iou,
        'pixel_accuracy': pixel_acc
    }

def profile_model(model, train_loader, criterion, optimizer, device):
    """Profile the training process"""
    print(" Profiling training performance...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        
        start_time= time.monotonic()
        
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if batch_idx>20:
                break
            with record_function("forward"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        end_time= time.monotonic()
    # Print top operations
    #print("\n⚡ TOP CPU OPERATIONS:")
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    prof.export_chrome_trace("profile_trace.json")
    print(prof.key_averages())
    
    print(f"Wall time is {end_time-start_time}s")
    #if torch.cuda.is_available():
    #    print("\nTOP GPU OPERATIONS:")
    #    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

def main():
    # Simple configuration for fast iterations
    config = {
        'batch_size':64,      # Larger batch for efficiency
        'num_epochs': 5,       # Few epochs for quick testing
        'learning_rate': 0.001,
        'num_workers': 2,      # Reduce workers to avoid overhead
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 10
    }
    
    print("🚀 ResNet-50 Synthetic Segmentation Training")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = ResNet50Segmentation(num_classes=config['num_classes'])
    model = model.to(config['device'])
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Metrics
    train_metrics = SegmentationMetrics(config['num_classes'])
    val_metrics = SegmentationMetrics(config['num_classes'])
    
    # Profile first
    profile_model(model, train_loader, criterion, optimizer, config['device'])

    exit()
    
    # Training
    print("\nStarting Training...")
    history = defaultdict(list)
    best_miou = 0.0
    
    total_start = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, 
            config['device'], train_metrics
        )
        
        # Validate
        val_results = validate_epoch(
            model, val_loader, criterion, config['device'], val_metrics
        )
        
        # Print results
        print(f"Train: Loss {train_results['loss']:.4f}, mIoU {train_results['mean_iou']:.3f}, Acc {train_results['pixel_accuracy']:.3f} ({train_results['time']:.1f}s)")
        print(f"Val:   Loss {val_results['loss']:.4f}, mIoU {val_results['mean_iou']:.3f}, Acc {val_results['pixel_accuracy']:.3f}")
        
        # Save best model
        if val_results['mean_iou'] > best_miou:
            best_miou = val_results['mean_iou']
            torch.save(model.state_dict(), 'best_synthetic_model.pth')
            print(f"💾 New best model saved! mIoU: {best_miou:.3f}")
        
        # Store history
        history['train_loss'].append(train_results['loss'])
        history['val_loss'].append(val_results['loss'])
        history['train_miou'].append(train_results['mean_iou'])
        history['val_miou'].append(val_results['mean_iou'])
    
    total_time = time.time() - total_start
    
    # Results
    print(f"\n Training completed in {total_time:.1f}s")
    print(f"Best validation mIoU: {best_miou:.3f}")

if __name__ == "__main__":
    main()