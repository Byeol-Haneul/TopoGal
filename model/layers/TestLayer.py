import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseLayer import sparse_row_norm, HBNS, HBS
from topomodelx.base.aggregation import Aggregation

class TestLayer(torch.nn.Module):
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
        attention_flag: bool = False
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
            attention_flag=attention_flag
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
            attention_flag=attention_flag
        )

        self.hbs_3_level1 = HBS(
            source_in_channels=in_channels_3,
            source_out_channels=intermediate_channels_3,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
            attention_flag=attention_flag
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
            attention_flag=attention_flag
        )

        # FCLs after msg passing
        self.fc_0 = torch.nn.Linear(intermediate_channels_0, intermediate_channels_0)
        self.fc_3 = torch.nn.Linear(intermediate_channels_0, intermediate_channels_3)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)

        self.aggr = Aggregation(aggr_func="sum", update_func=update_func_aggregation)

    def forward(
        self,
        x_0, x_1, x_2, x_3, x_4,
        adjacency_0, adjacency_1, adjacency_2, adjacency_3, adjacency_4,
        incidence_0_1, incidence_0_2, incidence_0_3, incidence_0_4,
        incidence_1_2, incidence_1_3, incidence_1_4,
        incidence_2_3, incidence_2_4,
        incidence_3_4
    ):
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_3_to_3 = self.hbs_3_level1(x_3, adjacency_3)

        _, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_0_1)
        x_2_to_3, _ = self.hbns_2_3_level1(x_3, x_2, incidence_2_3)
        
        x_0 = self.aggr([x_0_to_0, x_1_to_0])
        x_3 = self.aggr([x_2_to_3, x_3_to_3])

        x_0 = self.leaky_relu(self.fc_0(x_0))
        x_3 = self.leaky_relu(self.fc_3(x_3))
        return x_0, x_1, x_2, x_3, x_4

