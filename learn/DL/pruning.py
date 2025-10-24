import torch
import torch.nn as nn

class TinyNN(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ) for _ in range(layers)
        ])

    def forward(self, x):
        intermediate_activations = []
        for layer in self.layers:
            x = layer(x)
            intermediate_activations.append(x)
        return x, intermediate_activations  


net = TinyNN()

net.train()
## Prune 90% of weights in each layer
ratio_prune = 0.9
for name, param in net.named_parameters():
    if "weight" in name:  # only prune weights, not biases
        num_pruned = int(param.numel() * ratio_prune)
        flat = param.view(-1)
        _, indices = torch.topk(flat.abs(), num_pruned, largest=False)
        flat[indices] = 0.0
## train again
net.train()


## Pruned based on activations
for x,y in val_data_loader:
    act_scores=[]
    with torch.no_grad():
        _, intermediate_activations = net(x)
    # Compute importance scores (e.g., L1 norm)
   act_scores.append[torch.norm(act, p=1) for act in intermediate_activations]

act_scores= torch.tensor(act_scores).average(axis=0)
## prune each layer based on average activation score
for i, layer in enumerate(net.layers):
    if i < len(act_scores):
        num_pruned = int(act_scores[i].numel() * ratio_prune)
        flat = act_scores[i].view(-1)
        _, indices = torch.topk(flat.abs(), num_pruned, largest=False)
        flat[indices] = 0.0
