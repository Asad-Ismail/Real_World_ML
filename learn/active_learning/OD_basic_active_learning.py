import torch
import numpy as np
import random

# Mock datasets replace with your datasets
labeled_dataset = list(range(10))
unlabeled_dataset = list(range(10))

# Mock Object Detection Model replace with actual detection model
class ObjectDetectionModel:
    def __init__(self):
        pass
    def parameters(self):
        return [torch.nn.Parameter(torch.randn(2, 2))]
    
    def __call__(self, img):
        num_detections = random.randint(1, 5)
        detections = [{'confidence': random.random()} for _ in range(num_detections)]
        return detections

# Initialize
model = ObjectDetectionModel()

# Get detection scores for unlabelled dataset
uncertainties = []
for img in unlabeled_dataset:
    detections = model(img)
    # can replace with any aggrefation function mean, min, max 
    agg_confidence = np.mean([det['confidence'] for det in detections])
    uncertainties.append(agg_confidence)


# Active learning loop
n_queries = 5
threshold = 0.5
uncertainties = np.array(uncertainties)
most_uncertain_indices = np.where(np.abs(uncertainties - threshold) < 0.1)[0]
selected_indices=sorted(most_uncertain_indices, reverse=True)[:min(len(most_uncertain_indices),n_queries)]
print(f'Images to labels are {selected_indices}')
## Save most uncertain images and label them
    
print("Active learning completed!")
