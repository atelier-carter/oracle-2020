import torch
from torch import nn
import torch.nn.functional as F


torch.cuda.current_device()
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class PhilosophyRNN(nn.Module):  
    def __init__(self, tokens, hidden_size, n_layers, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(self.chars), 
                            self.hidden_size, 
                            n_layers, 
                            dropout=dropout_prob,
                            bidirectional=False,
                            batch_first=True
                           )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.lin = nn.Linear(self.hidden_size, len(self.chars))
        
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.lin(out)
        return out, hidden
        

    def init_hidden(self, batch_size=1, device=device):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device), 
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))