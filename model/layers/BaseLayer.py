"""Higher-Order Attentional NN Layer for Mesh Classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from topomodelx.base.aggregation import Aggregation
from model.aggregators import PNAAggregator

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

        self.att_weight = Parameter(
            torch.Tensor(self.target_out_channels + self.source_out_channels, 1)
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
            for w in ([self.w_s, self.w_t, self.att_weight.view(-1, 1)] + list(self.w_s_cci) + list(self.w_t_cci)):
                torch.nn.init.xavier_uniform_(w, gain=gain)


        elif self.initialization == "xavier_normal":
            for w in ([self.w_s, self.w_t, self.att_weight.view(-1, 1)] + list(self.w_s_cci) + list(self.w_t_cci)):
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

        s_message, s_message2, s_message3 = torch.matmul(x_source, self.w_s), torch.matmul(x_source, self.w_s_cci[0]), torch.matmul(x_source, self.w_s_cci[1])  # [n_source_cells, d_t_out]
        t_message, t_message2, t_message3 = torch.matmul(x_target, self.w_t), torch.matmul(x_target, self.w_t_cci[0]), torch.matmul(x_target, self.w_t_cci[1])  # [n_target_cells, d_s_out]

        neighborhood_s_to_t = neighborhood
        neighborhood_t_to_s = neighborhood.transpose(1, 2)

        # ADD CROSS-CELL INFORMATION
        if cci is not None:
            message_on_source = torch.matmul(neighborhood_t_to_s, t_message) + torch.matmul(cci.T, t_message2) + torch.matmul(cci.T*neighborhood_t_to_s, t_message3) 
            message_on_target = torch.matmul(neighborhood_s_to_t, s_message) + torch.matmul(cci, s_message2) + torch.matmul(cci*neighborhood_s_to_t, s_message3)
        else:
            message_on_source = torch.matmul(neighborhood_t_to_s, t_message) 
            message_on_target = torch.matmul(neighborhood_s_to_t, s_message) 

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

        self.att_weight = torch.nn.ParameterList(
            [
                Parameter(torch.Tensor(2 * self.source_out_channels, 1))
                for _ in range(self.m_hop)
            ]
        )
        self.negative_slope = negative_slope
        self.softmax = softmax

        self.reset_parameters()
        self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels) for _ in range(3)])


    def get_device(self) -> torch.device:
        """Get device on which the layer's learnable parameters are stored."""
        return self.weight[0].device

    def reset_parameters(self, gain: float = 1.414) -> None:
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization. Default is 1.414.
        """

        def reset_p_hop_parameters(weight, weight2, weight3, att_weight):
            if self.initialization == "xavier_uniform":
                for w in [weight, weight2, weight3, att_weight.view(-1, 1)]:
                    torch.nn.init.xavier_uniform_(w, gain=gain)

            elif self.initialization == "xavier_normal":
                for w in [weight, weight2, weight3, att_weight.view(-1, 1)]:
                    torch.nn.init.xavier_normal_(w, gain=gain)
            else:
                raise RuntimeError(
                    "Initialization method not recognized. "
                    "Should be either xavier_uniform or xavier_normal."
                )

        for w, w2, w3, a in zip(self.weight, self.weight2, self.weight3, self.att_weight, strict=True):
            reset_p_hop_parameters(w, w2, w3, a)

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
            [torch.matmul(x_source, w) for w in weights]
            for weights in [self.weight, self.weight2, self.weight3]
        ]  # [m-hop, n_source_cells, d_t_out]

        # Create a torch.eye with the device of x_source
        A_p = torch.eye(x_source.shape[1], device=self.get_device())

        m_hop_matrices = []

        # Generate the list of neighborhood matrices :math:`A, \dots, A^m`
        for _ in range(self.m_hop):
            A_p = torch.matmul(A_p, neighborhood)
            m_hop_matrices.append(A_p)
        
        message = [
            torch.matmul(n_p, m_p)
            for n_p, m_p in zip(m_hop_matrices, message, strict=True)
        ]

        if cci is not None:
            message2 = [
                torch.matmul(cci, m_p)
                for n_p, m_p in zip(m_hop_matrices, message2, strict=True)
            ]

            message3 = [
                torch.matmul(n_p*cci, m_p)
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