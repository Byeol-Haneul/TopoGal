"""Higher-Order Attentional NN Layer for Mesh Classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from model.aggregators import *

from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import *

def sparse_hadamard(A, B):
    return A*B
    '''
    # this is for torch.Tensor, layout == coo
    assert A.get_device() == B.get_device()
    A = A.coalesce()
    B = B.coalesce()

    return torch.sparse_coo_tensor(
        indices=A.indices(),
        values=A.values() * B.values(),
        size=A.shape,
        device=A.get_device(),
    )
    '''

def sparse_row_norm(sparse_tensor):
    row_sum = torch.sparse.sum(sparse_tensor, dim=1)
    values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor._indices(), values, sparse_tensor.shape
    )
    return sparse_tensor.coalesce()


class HBNS(torch.nn.Module):
    def __init__(
        self,
        source_in_channels: int,
        source_out_channels: int,
        target_in_channels: int,
        target_out_channels: int,
        negative_slope: float = 0.2,
        softmax: bool = False,
        update_func: str | None = None,
        initialization: str = "xavier_uniform",
        attention_flag: bool = False,
    ) -> None:
        super().__init__()

        self.initialization = initialization
        self.attention_flag = attention_flag

        self.source_in_channels, self.source_out_channels = (
            source_in_channels,
            source_out_channels,
        )
        self.target_in_channels, self.target_out_channels = (
            target_in_channels,
            target_out_channels,
        )

        self.update_func = update_func

        self.w_s = Parameter(
            torch.Tensor(self.source_in_channels, self.target_out_channels)
        )
        self.w_t = Parameter(
            torch.Tensor(self.target_in_channels, self.source_out_channels)
        )

        self.w_s_cci = torch.nn.ParameterList(
            [Parameter(
                torch.Tensor(self.source_in_channels, self.target_out_channels)
            )] * 2
        )

        self.w_t_cci = torch.nn.ParameterList(
            [Parameter(
                torch.Tensor(self.target_in_channels, self.source_out_channels)
            )] * 2
        )

        self.negative_slope = negative_slope

        self.softmax = softmax

        self.reset_parameters()

        self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels) for _ in range(3)])
        self.target_aggregators = nn.ModuleList([PNAAggregator(target_out_channels, target_out_channels) for _ in range(3)])
        #self.source_aggregators = self.target_aggregators = [default_agg] * 3

    def get_device(self) -> torch.device:
        """Get device on which the layer's learnable parameters are stored."""
        return self.w_s.device

    def reset_parameters(self, gain=1.414) -> None:
        if self.initialization == "xavier_uniform":
            for w in ([self.w_s, self.w_t] + list(self.w_s_cci) + list(self.w_t_cci)):
                torch.nn.init.xavier_uniform_(w, gain=gain)


        elif self.initialization == "xavier_normal":
            for w in ([self.w_s, self.w_t] + list(self.w_s_cci) + list(self.w_t_cci)):
                torch.nn.init.xavier_normal_(w, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized."
                "Should be either xavier_uniform or xavier_normal."
            )

    def update(
        self, message_on_source: torch.Tensor, message_on_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
       
        if self.update_func == "sigmoid":
            message_on_source = torch.sigmoid(message_on_source)
            message_on_target = torch.sigmoid(message_on_target)
        elif self.update_func == "relu":
            message_on_source = torch.nn.functional.relu(message_on_source)
            message_on_target = torch.nn.functional.relu(message_on_target)
        elif self.update_func == "tanh":
            message_on_source = torch.nn.functional.tanh(message_on_source)
            message_on_target = torch.nn.functional.tanh(message_on_target)

        return message_on_source, message_on_target

    def forward(
        self, x_source: torch.Tensor, x_target: torch.Tensor, neighborhood: SparseTensor, cci = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_message, s_message2, s_message3  = torch.mm(x_source, self.w_s), torch.mm(x_source, self.w_s_cci[0]), torch.mm(x_source, self.w_s_cci[1])  # [n_source_cells, d_t_out]
        t_message, t_message2, t_message3  = torch.mm(x_target, self.w_t), torch.mm(x_target, self.w_t_cci[0]), torch.mm(x_target, self.w_t_cci[1])  # [n_target_cells, d_s_out]

        neighborhood_s_to_t = (
            neighborhood.coalesce()
        )  # [n_target_cells, n_source_cells]
        neighborhood_t_to_s = (
            neighborhood.t().coalesce()
        )  # [n_source_cells, n_target_cells]

        # ADD CROSS-CELL INFORMATION
        if cci is not None:
            message_on_source = self.source_aggregators[0](neighborhood_t_to_s, t_message) + self.source_aggregators[1](cci.t(), t_message2) + self.source_aggregators[2](sparse_hadamard(cci.t(), neighborhood_t_to_s), t_message3) 
            message_on_target = self.target_aggregators[0](neighborhood_s_to_t, s_message) + self.target_aggregators[1](cci, s_message2) + self.target_aggregators[2](sparse_hadamard(cci, neighborhood_s_to_t), s_message3)
        else:
            message_on_source = torch.mm(neighborhood_t_to_s, t_message) 
            message_on_target = torch.mm(neighborhood_s_to_t, s_message) 

        if self.update_func is None:
            return message_on_source, message_on_target

        return self.update(message_on_source, message_on_target)


class HBS(torch.nn.Module):
    def __init__(
        self,
        source_in_channels: int,
        source_out_channels: int,
        negative_slope: float = 0.2,
        softmax: bool = False,
        m_hop: int = 1,
        update_func: str | None = None,
        initialization: str = "xavier_uniform",
        attention_flag: bool = False,
    ) -> None:
        super().__init__()

        self.initialization = initialization
        self.attention_flag = attention_flag

        self.source_in_channels = source_in_channels
        self.source_out_channels = source_out_channels

        self.m_hop = m_hop
        self.update_func = update_func

        self.weight = Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))
        self.weight2 = Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))
        self.weight3 = Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))


        self.negative_slope = negative_slope
        self.softmax = softmax

        self.reset_parameters()
        self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels) for _ in range(3)])
        #self.source_aggregators = [default_agg] * 3


    def get_device(self) -> torch.device:
        """Get device on which the layer's learnable parameters are stored."""
        return self.weight[0].device

    def reset_parameters(self, gain: float = 1.414) -> None:

        def reset_p_hop_parameters(weight, weight2, weight3):
            if self.initialization == "xavier_uniform":
                for w in [weight, weight2, weight3]:
                    torch.nn.init.xavier_uniform_(w, gain=gain)

            elif self.initialization == "xavier_normal":
                for w in [weight, weight2, weight3]:
                    torch.nn.init.xavier_normal_(w, gain=gain)
            else:
                raise RuntimeError(
                    "Initialization method not recognized. "
                    "Should be either xavier_uniform or xavier_normal."
                )

        for w, w2, w3 in zip([self.weight], [self.weight2], [self.weight3], strict=True):
            reset_p_hop_parameters(w, w2, w3)

    def update(self, message: torch.Tensor) -> torch.Tensor:
        if self.update_func == "sigmoid":
            return torch.sigmoid(message)
        if self.update_func == "relu":
            return torch.nn.functional.relu(message)
        if self.update_func == "tanh":
            return torch.nn.functional.tanh(message)

        raise RuntimeError(
            "Update function not recognized. Should be either sigmoid, relu or tanh."
        )

    def forward(
        self, x_source: torch.Tensor, neighborhood: SparseTensor, cci = None
    ) -> torch.Tensor:        
        message, message2, message3 = [
            torch.mm(x_source, w) for w in [self.weight, self.weight2, self.weight3]
        ] 

        if cci is not None:
            message, message2, message3 = [agg(_neighbor, _message) 
                                           for agg, _neighbor, _message in 
                                           zip(self.source_aggregators, [neighborhood, cci, sparse_hadamard(neighborhood, cci)], [message, message2, message3])]
        else:
            message = torch.mm(neighborhood, message)


        if cci is not None:
            result = message + message2 + message3
        else:
            result = message

        if self.update_func is None:
            return result
        
        return self.update(result)