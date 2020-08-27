import os
import re
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


torch.cuda.current_device()
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


dir_path = os.path.dirname(os.path.realpath(__file__))
full_path = os.path.join(dir_path, 'data/shakespeare.txt')

with open(full_path, 'r', encoding='utf8') as f:
    text = f.read()


chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}


def one_hot_encoder(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

hidden_size = 512
n_layers = 2
dropout_prob = 0.5

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint(os.path.join(dir_path, 'weights/shakespeare_model.pth'))
model.eval()


def predict(model, char, temperature, hidden=None, top_k=None):
        x = np.array([[model.char2int[char]]])
        x = one_hot_encoder(x, len(model.chars))
        
        inputs = torch.from_numpy(x)
        inputs = inputs.to(device)
        
        hidden = tuple([each.data for each in hidden])
        output, hidden = model(inputs, hidden)

        p = F.softmax(output/temperature, dim=1).data
        p = p.cpu()

        if top_k is None:
            top_ch = np.arange(len(model.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())
        return model.int2char[char], hidden



def sample(model, size, prime, temperature, top_k=None):
    model.to(device)  
    model.eval()
    chars = [ch for ch in prime]

    hidden = model.init_hidden(1)
    for ch in prime:
        char, hidden = predict(model, ch, temperature, hidden, top_k=top_k)

    chars.append(char)

    for ii in range(size):
        char, hidden = predict(model, chars[-1], temperature, hidden, top_k=top_k)
        chars.append(char)
    
    text = ''.join(chars)
    return text