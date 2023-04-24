import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        embeds = self.in_embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        avg_embeds = torch.mean(embeds, dim=1)      # (batch_size, embedding_dim)
        logits = self.out_embeddings(avg_embeds)    # (batch_size, vocab_size)
        return logits


class CBOWDataset(Dataset):
    def __init__(self, text: str, window_size: int, min_count: int = 1):
        tokens = text.lower().split()
        self.token_freqs = Counter(tokens)
        self.token_to_idx = {token: idx for idx, (token, _) in enumerate(self.token_freqs.items()) if self.token_freqs[token] >= min_count}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)

        # Create context-target pairs
        self.pairs = []
        for i, token in enumerate(tokens):
            if token not in self.token_to_idx:
                continue
            target_idx = self.token_to_idx[token]
            context_indices = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(len(tokens), i + window_size + 1)))
            context_indices = [idx for idx in context_indices if tokens[idx] in self.token_to_idx]

            if len(context_indices) > 0:
                self.pairs.extend([(target_idx, self.token_to_idx[tokens[idx]]) for idx in context_indices])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        return target, context

def collate_fn(batch: List[torch.Tensor]):
    targets, contexts = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.long)
    contexts = torch.tensor(contexts, dtype=torch.long)
    return targets, contexts

# Example usage
text = "I love learning about artificial intelligence and natural language processing"
window_size = 2
batch_size = 4

dataset = CBOWDataset(text, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#for targets, contexts in dataloader:
#    print("Targets:", targets)
#    print("Contexts:", contexts)
#    break


# Hyperparameters
embedding_dim = 50
learning_rate = 0.01
epochs = 20

# Initialize the CBOW model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CBOWModel(dataset.vocab_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    for targets, contexts in dataloader:
        targets = targets.to(device)
        contexts = contexts.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(contexts)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

print("Training finished!")

