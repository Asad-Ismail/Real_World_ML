import torch
import numpy as np
import random

# Mock datasets replace with your dataset
unlabeled_dataset = list(range(10))


class SegmentationModel:
    def __init__(self):
        pass
    def parameters(self):
        return [torch.nn.Parameter(torch.randn(2, 2))]
    
    def __call__(self, img):
        # each mask has shape (N,C,h,w) N is size of batch
        segm = {'confidence':torch.randn(1,21,24,24)}
        return segm

# Initialize
model = SegmentationModel()

# Get detection scores for unlabelled dataset
uncertainties = []
for img in unlabeled_dataset:
    segms = model(img)
    probs = torch.nn.functional.softmax(segms['confidence'],dim=1).max(dim=1)[0]
    entropies = -torch.sum(probs * torch.log(1e-9+probs), dim=[0,1,2]) 
    uncertainties.append(entropies.item())


# Active learning loop
n_queries = 5
selected_indices=np.argsort(uncertainties)[::-1][:min(len(uncertainties),n_queries)]
print(f'Images to labels are {selected_indices}')
## Save most uncertain images and label them
    
print("Active learning completed!")