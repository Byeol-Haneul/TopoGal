import torch
from torch import nn
from topomodelx.nn.combinatorial.hmc import HMC
from .layer import AugmentedHMC


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
        self.base_model = AugmentedHMC(channels_per_layer)
        
        # Compute the penultimate layer output size
        penultimate_layer = channels_per_layer[-1][-1][0]
        num_aggregators = 3 # max, min, avg
        num_ranks = 4       # rank 0~3
        
        # Define fully connected layers with LeakyReLU activations
        self.fc1 = nn.Linear(penultimate_layer * num_ranks * num_aggregators, 512)  # Adjust input size for concatenated features
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(512, 256)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc3 = nn.Linear(256, 128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc4 = nn.Linear(128, final_output_layer)

    def forward(
        self,
        x_0, x_1, x_2, x_3,
        neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3,
        neighborhood_0_to_1, neighborhood_1_to_2, neighborhood_2_to_3
    ) -> torch.Tensor:
        # Forward pass through the base model
        x_0, x_1, x_2, x_3 = self.base_model(
            x_0, x_1, x_2, x_3,
            neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3,
            neighborhood_0_to_1, neighborhood_1_to_2, neighborhood_2_to_3
        )
        
        def global_aggregations(x):
            #x_sum = torch.sum(x, dim=0, keepdim=True)
            x_avg = torch.mean(x, dim=0, keepdim=True)
            x_max, _ = torch.max(x, dim=0, keepdim=True)
            x_min, _ = torch.min(x, dim=0, keepdim=True)
            x = torch.cat((x_avg, x_max, x_min), dim=1)
            return x
        
        # Apply global aggregations to each feature set
        x_0 = global_aggregations(x_0)
        x_1 = global_aggregations(x_1)
        x_2 = global_aggregations(x_2)
        x_3 = global_aggregations(x_3)
        
        # Concatenate features from different inputs
        x = torch.cat((x_0, x_1, x_2, x_3), dim=1)

        # Forward pass through fully connected layers with LeakyReLU activations
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        
        return x
