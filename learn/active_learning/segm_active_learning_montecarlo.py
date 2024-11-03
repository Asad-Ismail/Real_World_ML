'''

To implement model variance for active learning with image segmentation, 
you can use Monte Carlo dropout. In Monte Carlo dropout, even during inference/testing, 
ropout is turned on. By running the forward pass multiple times and applying dropout each time,
you get different predictions on each pass.
The variance of these predictions can serve as a measure of model uncertainty.
Here's how you can modify the earlier code to incorporate model variance:
Use Dropout in Your Model:
Ensure your segmentation model (e.g., U-Net) uses dropout layers. Even if dropout wasn't used during training, you can add it just for the purpose of uncertainty estimation.

Modify the Inference Loop:
Instead of a single forward pass for each image, do multiple forward passes with dropout enabled.

Compute Variance:
For each pixel, compute the variance across the Monte Carlo samples.

'''
import torch
import torch.nn.functional as F
import numpy as np

# Set seed for reproducibility
torch.manual_seed(0)

# Mock datasets replace with your datasets
unlabeled_dataset = list(range(10)) 

# Parameters
mc_samples = 10  # Number of Monte Carlo samples
n_queries = 5 

class SegmentationModelWithDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)  

    def forward(self, img):
        confidence = torch.randn(1, 2, 2, 2)  
        confidence = self.dropout(confidence)  
        return {'confidence': confidence}

model = SegmentationModelWithDropout()

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

# Enable dropout during inference
model.eval()
model.apply(apply_dropout) 

uncertainties = []
for img in unlabeled_dataset:
    mc_predictions = []
    for _ in range(mc_samples):
        segms = model(img)  # Forward pass with dropout enabled
        probs = F.softmax(segms['confidence'], dim=1)
        mc_predictions.append(probs)
        
    mc_predictions = torch.stack(mc_predictions)  #(mc_samples, N, C, H, W)

    # Compute variance across the Monte Carlo samples for each pixel
    variance = mc_predictions.var(dim=0).mean(dim=[0, 1, 2, 3]).item()  # Mean variance across all pixels and classes
    uncertainties.append(variance)

# Active learning loop
selected_indices = np.argsort(uncertainties)[::-1][:min(len(uncertainties), n_queries)]
print(f'Images to label are: {selected_indices}')

print("Active learning completed!")
