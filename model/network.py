import torch
from torch import nn
from .layers import AugmentedHMCLayer, HierLayer, GNNLayer, MasterLayer, TestLayer

class Network(nn.Module):
    def __init__(self, layerType, channels_per_layer, final_output_layer, attention_flag: bool = False, residual_flag: bool = True):
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
        self.layerType = layerType
        self.base_model = CustomHMC(layerType, channels_per_layer, attention_flag=attention_flag, residual_flag=residual_flag)        
        penultimate_layer = channels_per_layer[-1][-1][0]
        num_aggregators = 4         # sum, max, min, avg

        if layerType == "Master":
            num_ranks_pooling = 5
        elif layerType == "Test":
            num_ranks_pooling = 2
        else:
            num_ranks_pooling = 1
        
        # Global feature size
        global_feature_size = 4     # x_0.shape[0], x_1.shape[0], x_2.shape[0], x_3.shape[0]

        # FCL
        self.fc1 = nn.Linear(penultimate_layer * num_ranks_pooling * num_aggregators + global_feature_size, 512)
        self.ln1 = nn.LayerNorm(512)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.fc4 = nn.Linear(128, final_output_layer)


    def forward(self, batch) -> torch.Tensor:
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
            n0_to_1, n0_to_2, n0_to_3, n0_to_4,
            n1_to_2, n1_to_3, n1_to_4,
            n2_to_3, n2_to_4,
            n3_to_4
        )

        def global_aggregations(x):
            x_avg = torch.mean(x, dim=0, keepdim=True)
            x_sum = torch.sum(x, dim=0, keepdim=True)
            x_max, _ = torch.max(x, dim=0, keepdim=True)
            x_min, _ = torch.min(x, dim=0, keepdim=True)
            x = torch.cat((x_avg, x_sum, x_max, x_min), dim=1)
            return x
        
        # Apply global aggregations to each feature set
        x_0 = global_aggregations(x_0)
        x_1 = global_aggregations(x_1)
        x_2 = global_aggregations(x_2)
        x_3 = global_aggregations(x_3)
        x_4 = global_aggregations(x_4)
       
        # Concatenate features from different inputs along with global features
        if self.layerType == "Hier":
            x = torch.cat((x_3, global_feature), dim=1)
        if self.layerType == "Test":
            x = torch.cat((x_0, x_3, global_feature), dim=1)
        elif self.layerType == "Master":
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, global_feature), dim=1)
        else:
            x = torch.cat((x_0, global_feature), dim=1)

        # Forward pass through fully connected layers with LeakyReLU activations
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        return x


class CustomHMC(torch.nn.Module):
    def __init__(
        self,
        layerType,
        channels_per_layer,
        negative_slope=0.2,
        update_func_attention="relu",
        update_func_aggregation="tanh", #"relu"
        attention_flag: bool = False,
        residual_flag: bool = True
    ) -> None:
        def check_channels_consistency():
            """Check that the number of input, intermediate, and output channels is consistent."""
            assert len(channels_per_layer) > 0
            for i in range(len(channels_per_layer) - 1):
                assert channels_per_layer[i][2][0] == channels_per_layer[i + 1][0][0]
                assert channels_per_layer[i][2][1] == channels_per_layer[i + 1][0][1]
                assert channels_per_layer[i][2][2] == channels_per_layer[i + 1][0][2]
                assert channels_per_layer[i][2][3] == channels_per_layer[i + 1][0][3]


        super().__init__()
        check_channels_consistency()

        if layerType == "Normal":
            self.base_layer = AugmentedHMCLayer
        elif layerType == "Hier":
            self.base_layer = HierLayer
            assert len(channels_per_layer) == 1
        elif layerType == "GNN":
            self.base_layer = GNNLayer
        elif layerType == "Master":
            self.base_layer = MasterLayer
        elif layerType == "Test":
            self.base_layer = TestLayer
        else:
            raise Exception("Invalid Model Type. Current Available Options are [Hier, Normal]")

        self.residual_flag = residual_flag
        self.layers = torch.nn.ModuleList(
            [
                self.base_layer(
                    in_channels=in_channels,
                    intermediate_channels=intermediate_channels,
                    out_channels=out_channels,
                    negative_slope=negative_slope,
                    softmax_attention=True, # softmax or row norm.
                    update_func_attention=update_func_attention,
                    update_func_aggregation=update_func_aggregation,
                    attention_flag=attention_flag,
                )
                for in_channels, intermediate_channels, out_channels in channels_per_layer
            ]
        )

    def forward(
        self,
        x_0, x_1, x_2, x_3, x_4,
        neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3, neighborhood_4_to_4,
        neighborhood_0_to_1, neighborhood_0_to_2, neighborhood_0_to_3, neighborhood_0_to_4,
        neighborhood_1_to_2, neighborhood_1_to_3, neighborhood_1_to_4,
        neighborhood_2_to_3, neighborhood_2_to_4,
        neighborhood_3_to_4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer_num, layer in enumerate(self.layers):
            h_0, h_1, h_2, h_3, h_4 = layer(
                x_0, x_1, x_2, x_3, x_4,
                neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3, neighborhood_4_to_4,
                neighborhood_0_to_1, neighborhood_0_to_2, neighborhood_0_to_3, neighborhood_0_to_4,
                neighborhood_1_to_2, neighborhood_1_to_3, neighborhood_1_to_4,
                neighborhood_2_to_3, neighborhood_2_to_4,
                neighborhood_3_to_4
            )

            residual_condition = self.residual_flag and layer_num > 1

            x_0 = h_0 + x_0 if residual_condition else h_0
            x_1 = h_1 + x_1 if residual_condition else h_1
            x_2 = h_2 + x_2 if residual_condition else h_2
            x_3 = h_3 + x_3 if residual_condition else h_3
            x_4 = h_4 + x_4 if residual_condition else h_4

        return x_0, x_1, x_2, x_3, x_4