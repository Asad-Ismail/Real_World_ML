import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TinyNN(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10, layers: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = input_dim
        for _ in range(layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
            current_dim = hidden_dim
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        hidden_activations = []
        for block in self.blocks:
            x = block(x)
            hidden_activations.append(x)
        logits = self.classifier(x)
        return logits, hidden_activations


def magnitude_prune_(model: nn.Module, ratio_prune: float = 0.9) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            threshold = torch.quantile(param.abs().view(-1), ratio_prune)
            param[param.abs() <= threshold] = 0.0


def collect_activation_scores(model: TinyNN, dataloader: DataLoader) -> list[torch.Tensor]:
    layer_sums = [torch.zeros(block[0].out_features) for block in model.blocks]
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            _, activations = model(inputs)
            for index, act in enumerate(activations):
                layer_sums[index] += act.abs().mean(dim=0).cpu()
            total_batches += 1

    if total_batches == 0:
        return layer_sums
    return [scores / total_batches for scores in layer_sums]


def activation_prune_(model: TinyNN, activation_scores: list[torch.Tensor], ratio_prune: float = 0.25) -> None:
    with torch.no_grad():
        for block, scores in zip(model.blocks, activation_scores):
            linear = block[0]
            num_pruned = max(1, int(scores.numel() * ratio_prune))
            _, indices = torch.topk(scores, num_pruned, largest=False)
            linear.weight[indices] = 0.0
            linear.bias[indices] = 0.0


if __name__ == "__main__":
    inputs = torch.randn(64, 784)
    targets = torch.randint(0, 10, (64,))
    val_data_loader = DataLoader(TensorDataset(inputs, targets), batch_size=16)

    net = TinyNN()
    magnitude_prune_(net, ratio_prune=0.9)
    activation_scores = collect_activation_scores(net, val_data_loader)
    activation_prune_(net, activation_scores, ratio_prune=0.25)

    logits, activations = net(inputs[:8])
    print("Logit shape:", logits.shape)
    print("Collected activation score tensors:", [scores.shape for scores in activation_scores])
