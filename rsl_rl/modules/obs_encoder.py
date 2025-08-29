import torch
import torch.nn as nn

class ObsEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=8):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
