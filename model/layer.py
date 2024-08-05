import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from topomodelx.nn.combinatorial.hmc_layer import sparse_row_norm, HBS, HBNS
from topomodelx.base.aggregation import Aggregation

class AugmentedHMCLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        intermediate_channels: list[int],
        out_channels: list[int],
        negative_slope: float,
        softmax_attention=False,
        update_func_attention=None,
        update_func_aggregation=None,
        initialization="xavier_uniform",
    ):
        super().__init__()

        assert (
            len(in_channels) == 4
            and len(intermediate_channels) == 4
            and len(out_channels) == 4
        )

        in_channels_0, in_channels_1, in_channels_2, in_channels_3 = in_channels
        (
            intermediate_channels_0,
            intermediate_channels_1,
            intermediate_channels_2,
            intermediate_channels_3,
        ) = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2, out_channels_3 = out_channels

        self.hbs_0_level1 = HBS(
            source_in_channels=in_channels_0,
            source_out_channels=intermediate_channels_0,
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

        self.hbns_1_2_level1 = HBNS(
            source_in_channels=in_channels_2,
            source_out_channels=intermediate_channels_2,
            target_in_channels=in_channels_1,
            target_out_channels=intermediate_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_2_3_level1 = HBNS(
            source_in_channels=in_channels_3,
            source_out_channels=intermediate_channels_3,
            target_in_channels=in_channels_2,
            target_out_channels=intermediate_channels_2,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_0_level2 = HBS(
            source_in_channels=intermediate_channels_0,
            source_out_channels=out_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_0_1_level2 = HBNS(
            source_in_channels=intermediate_channels_1,
            source_out_channels=out_channels_1,
            target_in_channels=intermediate_channels_0,
            target_out_channels=out_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_1_level2 = HBS(
            source_in_channels=intermediate_channels_1,
            source_out_channels=out_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_1_2_level2 = HBNS(
            source_in_channels=intermediate_channels_2,
            source_out_channels=out_channels_2,
            target_in_channels=intermediate_channels_1,
            target_out_channels=out_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_2_level2 = HBS(
            source_in_channels=intermediate_channels_2,
            source_out_channels=out_channels_2,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_2_3_level2 = HBNS(
            source_in_channels=intermediate_channels_3,
            source_out_channels=out_channels_3,
            target_in_channels=intermediate_channels_2,
            target_out_channels=out_channels_2,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_3_level2 = HBS(
            source_in_channels=intermediate_channels_3,
            source_out_channels=out_channels_3,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.aggr = Aggregation(aggr_func="sum", update_func=update_func_aggregation)

    def forward(
        self,
        x_0, x_1, x_2, x_3,
        adjacency_0, adjacency_1, coadjacency_2, coadjacency_3,
        incidence_1, incidence_2, incidence_3,
    ):
        # Computing messages from Higher Order Attention Blocks Level 1
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_0_to_1, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_1)
        x_1_to_2, x_2_to_1 = self.hbns_1_2_level1(x_2, x_1, incidence_2)
        x_2_to_3, x_3_to_2 = self.hbns_2_3_level1(x_3, x_2, incidence_3)

        x_0_level1 = self.aggr([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggr([x_0_to_1, x_2_to_1])
        x_2_level1 = self.aggr([x_1_to_2, x_3_to_2])
        x_3_level1 = self.aggr([x_2_to_3])

        # Computing messages from Higher Order Attention Blocks Level 2
        x_0_to_0 = self.hbs_0_level2(x_0_level1, adjacency_0)
        x_1_to_1 = self.hbs_1_level2(x_1_level1, adjacency_1)
        x_2_to_2 = self.hbs_2_level2(x_2_level1, coadjacency_2)
        x_3_to_3 = self.hbs_3_level2(x_3_level1, coadjacency_3)

        x_0_to_1, _ = self.hbns_0_1_level2(x_1_level1, x_0_level1, incidence_1)
        x_1_to_2, _ = self.hbns_1_2_level2(x_2_level1, x_1_level1, incidence_2)
        x_2_to_3, _ = self.hbns_2_3_level2(x_3_level1, x_2_level1, incidence_3)

        x_0_level2 = self.aggr([x_0_to_0])
        x_1_level2 = self.aggr([x_0_to_1, x_1_to_1])
        x_2_level2 = self.aggr([x_1_to_2, x_2_to_2])
        x_3_level2 = self.aggr([x_2_to_3, x_3_to_3])

        return x_0_level2, x_1_level2, x_2_level2, x_3_level2

class AugmentedHMC(torch.nn.Module):
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
                AugmentedHMCLayer(
                    in_channels=in_channels,
                    intermediate_channels=intermediate_channels,
                    out_channels=out_channels,
                    negative_slope=negative_slope,
                    softmax_attention=True,
                    update_func_attention=update_func_attention,
                    update_func_aggregation=update_func_aggregation,
                )
                for in_channels, intermediate_channels, out_channels in channels_per_layer
            ]
        )

    def forward(
        self,
        x_0, x_1, x_2, x_3,
        neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3,
        neighborhood_0_to_1, neighborhood_1_to_2, neighborhood_2_to_3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x_0, x_1, x_2, x_3 = layer(
                x_0, x_1, x_2, x_3,
                neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3,
                neighborhood_0_to_1, neighborhood_1_to_2, neighborhood_2_to_3,
            )

        return x_0, x_1, x_2, x_3