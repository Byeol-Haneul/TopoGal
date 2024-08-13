import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from topomodelx.nn.combinatorial.hmc_layer import sparse_row_norm, HBNS, HBS
from topomodelx.base.aggregation import Aggregation

class GNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        negative_slope: float,
        softmax_attention=False,
        update_func_attention=None,
        update_func_aggregation=None,
        initialization="xavier_uniform",
    ):
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2, in_channels_3, in_channels_4 = in_channels
        (
            intermediate_channels_0,
            intermediate_channels_1,
            intermediate_channels_2,
            intermediate_channels_3,
            intermediate_channels_4,
        ) = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2, out_channels_3, out_channels_4 = out_channels

        # Level 1
        self.hbs_0_level1 = HBS(
            source_in_channels=in_channels_0,
            source_out_channels=intermediate_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_1_level1 = HBS(
            source_in_channels=in_channels_1,
            source_out_channels=intermediate_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_0_1_level1 = HBNS(
            source_in_channels=in_channels_1,
            source_out_channels=intermediate_channels_1,
            target_in_channels=in_channels_0,
            target_out_channels=intermediate_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.aggr = Aggregation(aggr_func="mean", update_func=update_func_aggregation)

    def forward(
        self,
        x_0, x_1,
        adjacency_0, adjacency_1,
        incidence_1,
    ):
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_1_to_1 = self.hbs_1_level1(x_1, adjacency_1)
        x_0_to_1, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_1)
        
        x_0_level1 = self.aggr([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggr([x_0_to_1, x_1_to_1])

        return x_0_level1, x_1_level1

class GNN(torch.nn.Module):
    def __init__(
        self,
        channels_per_layer,
        negative_slope=0.2,
        update_func_attention="relu",
        update_func_aggregation="relu",
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
        self.layers = torch.nn.ModuleList(
            [
                GNNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    negative_slope=negative_slope,
                    softmax_attention=True, # softmax or row norm.
                    update_func_attention=update_func_attention,
                    update_func_aggregation=update_func_aggregation,
                )
                for in_channels, _, out_channels in channels_per_layer
            ]
        )

    def forward(
        self,
        x_0, x_1,
        neighborhood_0_to_0, neighborhood_1_to_1,
        neighborhood_0_to_1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x_0, x_1 = layer(
                x_0, x_1,
                neighborhood_0_to_0, neighborhood_1_to_1,
                neighborhood_0_to_1
            )

        return x_0, x_1