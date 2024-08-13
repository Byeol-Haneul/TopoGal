import torch
from torch import nn
from topomodelx.nn.combinatorial.hmc import HMC
from .layer import AugmentedHMC
from .HierLayer import HierHMC


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
        num_aggregators = 3 # sum, max, min, avg
        num_ranks_pooling = 5      # rank 0~4
        
        # Global feature size
        global_feature_size = 4  # x_0.shape[0], x_1.shape[0], x_2.shape[0], x_3.shape[0]

        # Define fully connected layers with LeakyReLU activations
        self.fc1 = nn.Linear(
            penultimate_layer * num_ranks_pooling * num_aggregators + global_feature_size, 512
        )  # Adjust input size for concatenated features + global features
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(512, 256)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc3 = nn.Linear(256, 128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc4 = nn.Linear(128, final_output_layer)

    def forward(self, batch) -> torch.Tensor:
        # Extract from Dict
        x_0 = batch['x_0']
        x_1 = batch['x_1']
        x_2 = batch['x_2']
        x_3 = batch['x_3']
        x_4 = batch['x_4']

        n0_to_0 = batch['n0_to_0']
        n1_to_1 = batch['n1_to_1']
        n2_to_2 = batch['n2_to_2']
        n3_to_3 = batch['n3_to_3']

        n0_to_1 = batch['n0_to_1']
        n1_to_2 = batch['n1_to_2']
        n2_to_3 = batch['n2_to_3']
        n3_to_4 = batch['n3_to_4']
       
        global_feature = batch['global_feature']
        # Forward pass through the base model
        x_0, x_1, x_2, x_3, x_4 = self.base_model(
            x_0, x_1, x_2, x_3, x_4, 
            n0_to_0, n1_to_1, n2_to_2, n3_to_3,
            n0_to_1, n1_to_2, n2_to_3, n3_to_4, 
        )
        
        def global_aggregations(x):
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
        x_4 = global_aggregations(x_4)
       
        # Concatenate features from different inputs along with global features
        x = torch.cat((x_0, x_1, x_2, x_3, x_4, global_feature), dim=1)

        # Forward pass through fully connected layers with LeakyReLU activations
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        return x


class HierNetwork(nn.Module):
    def __init__(self, channels_per_layer, final_output_layer):
        super().__init__()
        self.base_model = HierHMC(channels_per_layer)
        
        # Compute the penultimate layer output size
        penultimate_layer = channels_per_layer[-1][-1][0]
        num_aggregators = 3         # sum, max, min, avg
        num_ranks_pooling = 1       # rank 0,1,2,3
        
        # Global feature size
        global_feature_size = 4  # x_0.shape[0], x_1.shape[0], x_2.shape[0], x_3.shape[0]

        # Define fully connected layers with LeakyReLU activations
        self.fc1 = nn.Linear(
            penultimate_layer * num_ranks_pooling * num_aggregators + global_feature_size, 512
        )  # Adjust input size for concatenated features + global features
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(512, 256)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc3 = nn.Linear(256, 128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc4 = nn.Linear(128, final_output_layer)

    def forward(self, batch) -> torch.Tensor:
        # Extract from Dict
        # features
        x_0 = batch['x_0']
        x_1 = batch['x_1']
        x_2 = batch['x_2']
        x_3 = batch['x_3']
        x_4 = batch['x_4']

        # (Co)Adjacency
        n0_to_0 = batch['n0_to_0']
        n1_to_1 = batch['n1_to_1']
        n2_to_2 = batch['n2_to_2']
        n3_to_3 = batch['n3_to_3']
        n4_to_4 = batch['n4_to_4']

        # Incidence
        n0_to_1 = batch['n0_to_1']
        n0_to_2 = batch['n0_to_2']
        n0_to_3 = batch['n0_to_3']
        n0_to_4 = batch['n0_to_4']
        n1_to_2 = batch['n1_to_2']
        n1_to_3 = batch['n1_to_3']
        n1_to_4 = batch['n1_to_4']
        n2_to_3 = batch['n2_to_3']
        n2_to_4 = batch['n2_to_4']
        n3_to_4 = batch['n3_to_4']
       
        # global feature
        global_feature = batch['global_feature']

        # Forward pass through the base model
        x_0, x_1, x_2, x_3, x_4 = self.base_model(
            x_0, x_1, x_2, x_3, x_4, 
            n0_to_0, n1_to_1, n2_to_2, n3_to_3, n4_to_4,
            n0_to_1, n1_to_2, n2_to_3, n3_to_4
        )
        
        def global_aggregations(x):
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
        
        # Concatenate features from different inputs along with global features
        x = torch.cat((x_3, global_feature), dim=1)

        # Forward pass through fully connected layers with LeakyReLU activations
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        
        return x