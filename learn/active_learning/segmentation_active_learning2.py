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
import numpy as np


# Assume DataLoader and SegmentationModel from the earlier example
# SegmentationModel should contain dropout layers

model = SegmentationModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(dataset):
    # ... typical segmentation model training loop ...

# Active learning loop
n_queries = 5
for _ in range(n_queries):
    # Train on labeled data
    train_model(labeled_dataset)

    # Predict on unlabeled data and get uncertainties
    uncertainties = []

    # Number of Monte Carlo samples
    n_mc_samples = 10

    for img in unlabeled_dataset:
        mc_samples = []
        for _ in range(n_mc_samples):
            model.train()  # Ensure dropout is turned on
            logits = model(img)
            probs = torch.nn.functional.softmax(logits, dim=1)
            mc_samples.append(probs)
        
        mc_samples = torch.stack(mc_samples)
        # Variance across the Monte Carlo samples for each pixel
        variance = mc_samples.var(dim=0).sum(dim=1)  # Sum variances for a single image uncertainty score
        uncertainties.append(variance.item())

    # Select top N most uncertain images
    N = 10
    most_uncertain_indices = np.argsort(uncertainties)[-N:]

    # Query expert and update datasets (as earlier)

print("Active learning with model variance completed!")
