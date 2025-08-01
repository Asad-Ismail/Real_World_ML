import torch
import torch.nn as nn


## Few ways to enforce ordering
## Frame as regression task and use MSE standard cross entropy does not enforce it
## Use Coral
## Coral equation P(y>k∣x)=σ(z(x)−bk)

class Coral(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes - 1)
        )
        self.biases = nn.Parameter(torch.randn(num_classes - 1))

    def forward(self, x):
        # first x determines the scoring logits of inouts
        x = self.features(x)
        # subtraction of biases encourages learning the thresholds
        x = x - self.biases  
        return torch.sigmoid(x)

    def loss_calculation(self, pred, labels):
        """
        pred: (batch_size, num_classes - 1)
        labels: (batch_size,) integers from 0 to num_classes-1
        """
        ## we can also enfore more strictness using 
        ## reg = torch.mean(F.relu(probs[:, 1:] - probs[:, :-1]))

        batch_size, num_thresholds = pred.shape
        expanded_labels = torch.zeros((batch_size, num_thresholds), device=pred.device)
        for i in range(num_thresholds):
            expanded_labels[:, i] = (labels > i).float()
        loss_fn = nn.BCELoss()
        return loss_fn(pred, expanded_labels)

    def predict(self, preds):
        """
        preds: (batch_size, num_classes - 1) with values in (0, 1)
        returns: predicted class labels (batch_size,)
        """
        return torch.sum(preds > 0.5, dim=1)


if __name__ == "__main__":
    bs = 12
    input_dim = 32
    num_classes = 3

    x = torch.randn((bs, input_dim))
    y_true = torch.randint(0, num_classes, (bs,))

    model = Coral(input_dim=input_dim, num_classes=num_classes)
    y_pred = model(x)

    print(f"Input shape: {x.shape}, Output shape: {y_pred.shape}")
    loss = model.loss_calculation(y_pred, y_true)
    print(f"Loss: {loss.item():.4f}")

    y_pred_class = model.predict(y_pred)
    print(f"Predicted labels: {y_pred_class}")
