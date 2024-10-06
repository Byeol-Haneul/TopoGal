"""Higher-Order Attentional NN Layer for Mesh Classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from model.aggregators import PNAAggregator

def sparse_hadamard(A, B):
    assert A.get_device() == B.get_device()
    A = A.coalesce()
    B = B.coalesce()

    return torch.sparse_coo_tensor(
        indices=A.indices(),
        values=A.values() * B.values(),
        size=A.shape,
        device=A.get_device(),
    )

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

        # Aggregators!!
        #self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels) for _ in range(3)])
        #self.target_aggregators = nn.ModuleList([PNAAggregator(target_out_channels, target_out_channels) for _ in range(3)])


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
        self, x_source: torch.Tensor, x_target: torch.Tensor, neighborhood: torch.Tensor, cci = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cci is not None:
            assert neighborhood.shape == cci.shape

        s_message, s_message2, s_message3 = torch.mm(x_source, self.w_s), torch.mm(x_source, self.w_s_cci[0]), torch.mm(x_source, self.w_s_cci[1])  # [n_source_cells, d_t_out]
        t_message, t_message2, t_message3 = torch.mm(x_target, self.w_t), torch.mm(x_target, self.w_t_cci[0]), torch.mm(x_target, self.w_t_cci[1])  # [n_target_cells, d_s_out]

        neighborhood_s_to_t = (
            neighborhood.coalesce()
        )  # [n_target_cells, n_source_cells]
        neighborhood_t_to_s = (
            neighborhood.t().coalesce()
        )  # [n_source_cells, n_target_cells]

        self.target_indices, self.source_indices = neighborhood_s_to_t.indices()

        neighborhood_s_to_t_att = torch.sparse_coo_tensor(
            indices=neighborhood_s_to_t.indices(),
            values=neighborhood_s_to_t.values(),
            size=neighborhood_s_to_t.shape,
            device=self.get_device(),
        )

        neighborhood_t_to_s_att = torch.sparse_coo_tensor(
            indices=neighborhood_t_to_s.indices(),
            values=neighborhood_t_to_s.values(),
            size=neighborhood_t_to_s.shape,
            device=self.get_device(),
        )

        # ADD CROSS-CELL INFORMATION
        if cci is not None:
            message_on_source = torch.mm(neighborhood_t_to_s_att, t_message) + torch.mm(cci.T, t_message2) + torch.mm(sparse_hadamard(cci.T, neighborhood_t_to_s), t_message3) 
            message_on_target = torch.mm(neighborhood_s_to_t_att, s_message) + torch.mm(cci, s_message2) + torch.mm(sparse_hadamard(cci, neighborhood_s_to_t), s_message3)
        else:
            message_on_source = torch.mm(neighborhood_t_to_s_att, t_message) 
            message_on_target = torch.mm(neighborhood_s_to_t_att, s_message) 

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

        self.weight = torch.nn.ParameterList(
            [
                Parameter(
                    torch.Tensor(self.source_in_channels, self.source_out_channels)
                )
                for _ in range(self.m_hop)
            ]
        )

        self.weight2 = torch.nn.ParameterList(
            [
                Parameter(
                    torch.Tensor(self.source_in_channels, self.source_out_channels)
                )
                for _ in range(self.m_hop)
            ]
        )

        self.weight3 = torch.nn.ParameterList(
            [
                Parameter(
                    torch.Tensor(self.source_in_channels, self.source_out_channels)
                )
                for _ in range(self.m_hop)
            ]
        )

        self.negative_slope = negative_slope
        self.softmax = softmax

        self.reset_parameters()
        #self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels) for _ in range(3)])


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

        for w, w2, w3 in zip(self.weight, self.weight2, self.weight3, strict=True):
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
        self, x_source: torch.Tensor, neighborhood: torch.Tensor, cci = None
    ) -> torch.Tensor:        
        if cci is not None:
            assert neighborhood.shape == cci.shape

        message, message2, message3 = [
            [torch.mm(x_source, w) for w in weights]
            for weights in [self.weight, self.weight2, self.weight3]
        ]  # [m-hop, n_source_cells, d_t_out]

        # Create a torch.eye with the device of x_source
        A_p = torch.eye(x_source.shape[0], device=self.get_device()).to_sparse_coo()

        m_hop_matrices = []

        # Generate the list of neighborhood matrices :math:`A, \dots, A^m`
        for _ in range(self.m_hop):
            A_p = torch.sparse.mm(A_p, neighborhood)
            m_hop_matrices.append(A_p)
        

        message = [
            torch.mm(n_p, m_p)
            for n_p, m_p in zip(m_hop_matrices, message, strict=True)
        ]

        if cci is not None:
            message2 = [
                torch.mm(cci, m_p)
                for n_p, m_p in zip(m_hop_matrices, message2, strict=True)
            ]

            message3 = [
                torch.mm(sparse_hadamard(n_p, cci), m_p)
                for n_p, m_p in zip(m_hop_matrices, message3, strict=True)
            ]
            

        result = torch.zeros_like(message[0])
        if cci is not None:
            for m_p, m_p2, m_p3 in zip(message, message2, message3):
                result += (m_p + m_p2 + m_p3)
        else:
            for m_p in message:
                result += m_p

        if self.update_func is None:
            return result

        return self.update(result)