import torch
from torch import nn
from topomodelx.nn.combinatorial.hmc import HMC

class Network(nn.Module):
    def __init__(self, channels_per_layer, final_output_layer):
        super().__init__()
        '''
        Base Model: Higher Order Attention Network for Mesh Classification ()
        x{0, 1, 2, 3} = {nodes, edges, tetra, tetra clusters} features
        x0: (x, y, z, Mstar, Rstar)
        x1: (Edge distance)
        x2: (tetra volume)
        x3: (Merged tetrav volume) Currently not used.

        References
        ----------
        .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
            Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
            (2023) https://arxiv.org/abs/2206.00606.
        '''
        self.base_model = HMC(channels_per_layer)
        
        # Compute the penultimate layer output size
        penultimate_layer = channels_per_layer[-1][-1][0]
        
        # Adaptive max pooling to ensure the output is of fixed size
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, penultimate_layer))
        
        # Define fully connected layers with LeakyReLU activations
        self.fc1 = nn.Linear(penultimate_layer * 3, 512)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(512, 256)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc3 = nn.Linear(256, 128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc4 = nn.Linear(128, final_output_layer)

    def forward(
        self,
        x_0, x_1, x_2,
        neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2,
        neighborhood_0_to_1, neighborhood_1_to_2
    ) -> torch.Tensor:
        # Forward pass through the base model
        x_0, x_1, x_2 = self.base_model(
            x_0, x_1, x_2,
            neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2,
            neighborhood_0_to_1, neighborhood_1_to_2
        )
        
        # Global max pooling
        x_0 = self.global_max_pool(x_0.unsqueeze(0)).squeeze(0)
        x_1 = self.global_max_pool(x_1.unsqueeze(0)).squeeze(0)
        x_2 = self.global_max_pool(x_2.unsqueeze(0)).squeeze(0)
        
        # Concatenate features from different inputs
        x = torch.cat((x_0, x_1, x_2), dim=1)
        
        # Forward pass through fully connected layers with LeakyReLU activations
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        
        return x
