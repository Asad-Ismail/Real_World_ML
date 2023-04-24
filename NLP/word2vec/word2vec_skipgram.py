import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random

# Preprocess your text data
data = "the car sat on the mat"
tokens = data.split()
unique_words = set(tokens)

# Hyperparameters
embedding_dim = 300
window_size = 3
batch_size = 256
epochs = 5

# Build vocabulary
word_counts = Counter(tokens)
word2idx = {word: i for i, word in enumerate(unique_words)}
idx2word = {i: word for i, word in enumerate(unique_words)}

# Generate training data
def generate_skip_grams(tokens, window_size):
    data = []
    for i, word in enumerate(tokens):
        context_indices = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(len(tokens), i + window_size + 1)))
        for index in context_indices:
            data.append((word2idx[word], word2idx[tokens[index]]))
    return data

skip_grams = generate_skip_grams(tokens, window_size)

# Define dataset class
class SkipGramDataset(Dataset):
    def __init__(self, skip_grams):
        self.skip_grams = skip_grams

    def __len__(self):
        return len(self.skip_grams)

    def __getitem__(self, idx):
        return self.skip_grams[idx]

# DataLoader
dataset = SkipGramDataset(skip_grams)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Skip-Gram model
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        in_embeds = self.in_embeddings(target)
        out_embeds = self.out_embeddings(context)
        scores = torch.matmul(in_embeds, out_embeds.t())
        return scores

# Initialize the model, optimizer, and loss function
model = SkipGramModel(len(unique_words), embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        target, context = batch
        target = target.to(device)
        context = context.to(device)

        optimizer.zero_grad()
        logits = model(target, context)
        print(f"Logits shape is {logits.shape}")
        print(f"target shape is {target.shape}")
        print(f"Context shape is {context.shape}")
        
        print(f"Target is {target}")
        
        print(f"Context is {context}")

        target_labels = context.view(-1)
        loss = criterion(logits, target_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Extract word embeddings
word_embeddings = model.in_embeddings.weight.cpu().detach().numpy()
