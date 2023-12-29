import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet(10, 50, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Sample data (just for the demonstration)
x_sample = torch.randn(100, 10)
y_sample = torch.randint(0, 2, (100,)).long()

# 2. Compute the FIM
def compute_fim(model, data, criterion):
    model.eval()
    fim = {}
    for name, param in model.named_parameters():
        param_grad = torch.autograd.grad(criterion(model(data), y_sample), param, create_graph=True)[0]
        fim[name] = param_grad @ param_grad.t()
    return fim

criterion = nn.CrossEntropyLoss()
fim = compute_fim(model, x_sample, criterion)

# 3. Prune based on FIM values
threshold = 1e-4  # threshold for pruning
for name, param in model.named_parameters():
    if name in fim:
        # Identify low Fisher information parameters
        mask = fim[name].diag() > threshold
        # Prune those parameters by zeroing out
        param.data *= mask.float()

# 4. Fine-tune if necessary (this is a simple demonstration, so we skip fine-tuning)

