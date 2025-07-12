import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


##################################
#  Q5 / Q11
##################################

class PoSGRU(nn.Module) :

    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True, embed_init=None):
        super().__init__()

        # TODO
        # Embedding Layer
        if embed_init is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embed_init,
                freeze=False,
                padding_idx=1
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=1
            )

        # Mapping embed_dim to hidden_dim
        self.linear_mapping = nn.Linear(embed_dim, hidden_dim)

        # GRU flag
        self.use_gru = num_layers > 0

        # Residual flag
        self.residual = residual and self.use_gru

        # Number of layers
        self.num_layers = num_layers

        # Bidirectional GRU
        if self.use_gru:
            assert hidden_dim % 2 == 0, "hidden_dim should be even number"
            self.gru_layers = nn.ModuleList([
                nn.GRU(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim//2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True)
                for _ in range(num_layers)
            ])
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Classifier module
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # TODO
        # Word embedding
        x = self.embedding(x)

        # Mapping embeddings to hidden_dim
        h = self.linear_mapping(x)

        # GRU layers
        if self.use_gru:
            out = h
            for gru in self.gru_layers:
                new_out, _ = gru(out)
                out = out + new_out if self.residual else new_out
        else:
            B, T, H = h.shape
            out = self.mlp(h.view(-1, H)).view(B, T, H)

        # Classifier
        logits = self.classifier(out)
        return logits
