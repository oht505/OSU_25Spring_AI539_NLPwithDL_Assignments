import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        #TODO

        # x: (B, L, d_model)
        B, L, d_model = x.size()
        device = x.device

        pe = torch.zeros(L, d_model, device=device)
        t = torch.arange(L, dtype=torch.float, device=device).unsqueeze(1)
        f_j = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(t * f_j)
        pe[:, 1::2] = torch.cos(t * f_j)

        pe.unsqueeze(0)

        return x + pe

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=True)
        self.classifier = nn.Linear(d_model, vocab_size)

    def generateCausalMask(self, L):
        #TODO
        mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        return mask

    def forward(self, x):
        #TODO

        # x = (B, L)
        B, L = x.shape

        # Embedding
        x = self.embeddings(x)

        # Add position
        x = self.position(x)

        # Casual mask
        mask = self.generateCausalMask(L).to(x.device)

        # TF encoder
        x = self.encoder(x, mask=mask, is_causal=True)

        # Output projection
        output = self.classifier(x)

        return output
