import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

CUDA = torch.cuda.is_available()

def to_np(x):
    return x.detach().cpu().numpy()

def to_tensor(x, requires_grad=False):
    x = torch.from_numpy(x)
    if CUDA:
        x = x.cuda()
    
    if requires_grad:
        return x.clone().contiguous().detach().requires_grad_(True)
    else:
        return x.clone().contiguous().detach()

    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28, 28),
            nn.Tanh(),
            nn.Linear(28, 28),
            nn.Tanh(),
            nn.Linear(28, 8),
            nn.Tanh()
        )

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.layers.forward(X)
    
    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)
    
    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(to_tensor(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(to_tensor(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_np(v).flatten() for v in
                                   self.parameters()]))
        