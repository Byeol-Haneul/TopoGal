import torch
from torch import nn
from network import get_activation

class TPCF_NN(nn.Module):
    def __init__(
        self, 
        input_dim,                   
        output_dim  = 2,                  
        hidden_dims =[128, 64, 32],  
        activation  ="relu",           
        dropout     =0.1                  
    ):
        super().__init__()
        self.activation = get_activation(activation)

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))           
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
