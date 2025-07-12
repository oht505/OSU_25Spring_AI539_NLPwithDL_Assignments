from torch import nn
import torch

##################################
#  Q2
##################################

class ParityLSTM(nn.Module) :

    def __init__(self, hidden_dim=16):
        super().__init__()
        #TODO
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, x_lens):
        #TODO

        # LSTM layer
        out, _ = self.lstm(x)

        # Find last hidden (x_lens[i]-1)  
        idxs = (x_lens - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last_hidden = out.gather(dim=1, index=idxs).squeez(1)

        logits = self.classifier(last_hidden)
        return logits