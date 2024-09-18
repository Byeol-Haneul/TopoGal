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
    r"""Normalize a sparse tensor by row dividing each row by its sum.

    Parameters
    ----------
    sparse_tensor : torch.sparse, shape=[n_cells, n_cells]

    Returns
    -------
    _ : torch.sparse, shape=[n_cells, n_cells]
        Normalized by rows sparse tensor.
    """
    row_sum = torch.sparse.sum(sparse_tensor, dim=1)
    values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor._indices(), values, sparse_tensor.shape
    )
    return sparse_tensor.coalesce()


class HBNS(torch.nn.Module):
    r"""Higher Order Attention Block for non-squared neighborhood matrices.

    Let :math:`\mathcal{X}` be a combinatorial complex, we denote by
    :math:`\mathcal{C}^k(\mathcal{X}, \mathbb{R}^d)` the :math:`\mathbb{
    R}`-valued vector space of :math:`d`-dimensional signals over
    :math:`\Sigma^k`, the :math:`k`-th skeleton of :math:`\mathcal{X}`
    subject to a certain total order. Elements of this space are called
    :math:`k`-cochains of :math:`\mathcal{X}`. If :math:`d = 1`, we denote
    it by :math:`\mathcal{C}^k(\mathcal{X})`.

    Let :math:`N: \mathcal{C}^s(\mathcal{X}) \rightarrow \mathcal{C}^t(
    \mathcal{X})` with :math:`s \neq t` be a non-squared neighborhood matrix
    from the :math:`s` th-skeleton of :math:`\mathcal{
    X}` to its :math:`t` th-skeleton. The higher order
    attention block induced by :math:`N` is a cochain map

    ..  math::
        \begin{align}
            \text{HBNS}_N: \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{in}}})
            \times \mathcal{C}^t(\mathcal{X},\mathbb{R}^{d^{t_{in}}})
            \rightarrow \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{t_{out}}})
            \times \mathcal{C}^t(\mathcal{X},\mathbb{R}^{d^{s_{out}}}),
        \end{align}

    where :math:`d^{s_{in}}`, :math:`d^{t_{in}}`, :math:`d^{s_{out}}`,
    and :math:`d^{t_{out}}` are the input and output dimensions of the
    source and target cochains, also denoted as source_in_channels,
    target_in_channels, source_out_channels, and target_out_channels.

    The cochain map :math:`\text{HBNS}_N` is defined as

    ..  math::
        \begin{align}
            \text{HBNS}_N(X_s, X_t) = (Y_s, Y_t),
        \end{align}

    where the source and target output cochain matrices :math:`Y_s` and
    :math:`Y_t` are computed as

     ..  math::
        \begin{align}
            Y_s &= \phi((N^T \odot A_t) X_t W_t), \\
            Y_t &= \phi((N \odot A_s) X_s W_s ).
        \end{align}

    Here, :math:`\odot` denotes the Hadamard product, namely the entry-wise
    product, and :math:`\phi` is a non-linear activation function.
    :math:`W_t` and :math:`W_s` are learnable weight matrices of shapes
    [target_in_channels, source_out_channels] and [source_in_channels,
    target_out_channels], respectively. Attention matrices are denoted as
    :math:`A_t` and :math:`A_s` and have the same dimensions as :math:`N^T`
    and :math:`N`, respectively. The entries :math:`(i, j)` of the attention
    matrices :math:`A_t` and :math:`A_s` are defined as

    ..  math::
        \begin{align}
            A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{\#\text{columns}(N)} e_{i,
            k}}, \\
            A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{\#\text{columns}(N^T)} f_{i,
            k}},
        \end{align}

    where,

    ..  math::
        \begin{align}
            e_{i,j} &= S(\text{LeakyReLU}([(X_s)_jW_s||(X_t)_iW_t]a)),\\
            f_{i,j} &= S(\text{LeakyReLU}([(X_t)_jW_t||(X_s)_iW_s][a[d_{s_{
            out}}:]||a[:d_{s_{out}}])).\\
        \end{align}

    Here, || denotes concatenation and :math:`a` denotes the learnable column
    attention vector of length :math:`d_{s_{out}} + d_{t_{out}}`.
    Given a vector :math:`v`, we denote by :math:`v[:c]` and :math:`v[c:]`
    to the projection onto the first :math:`c` elements and the last
    elements of :math:`v` starting from the :math:`(c+1)`-th element,
    respectively. :math:`S` is the exponential function if softmax is used
    and the identity function otherwise.

    This HBNS class just contains the sparse implementation of the block.

    Notes
    -----
    HBNS layers were introduced in [H23]_, Definition 31 and 33.

    References
    ----------
    .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
        (2023) https://arxiv.org/abs/2206.00606.

    Parameters
    ----------
    source_in_channels : int
        Number of input features for the source cells.
    source_out_channels : int
        Number of output features for the source cells.
    target_in_channels : int
        Number of input features for the target cells.
    target_out_channels : int
        Number of output features for the target cells.
    negative_slope : float
        Negative slope of the LeakyReLU activation function.
    softmax : bool, optional
        Whether to use softmax or sparse_row_norm in the computation of the
        attention matrix. Default is False.
    update_func : {None, 'sigmoid', 'relu'}, optional
        Activation function :math:`\phi` in the computation of the output of
        the layer. If None, :math:`\phi` is the identity function. Default is
        None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of :math:`W_p` and the attention
        vector :math:`a`. Default is 'xavier_uniform'.
    """

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

        # Aggregators!!
        self.source_aggregator = PNAAggregator(source_out_channels, source_out_channels)
        self.target_aggregator = PNAAggregator(target_out_channels, target_out_channels)

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

    def get_device(self) -> torch.device:
        """Get device on which the layer's learnable parameters are stored."""
        return self.w_s.device

    def reset_parameters(self, gain=1.414) -> None:
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization. Default is 1.414.
        """
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
        r"""Update signal features on each cell with an activation function.

        The implemented activation functions are sigmoid, ReLU and tanh.

        Parameters
        ----------
        message_on_source : torch.Tensor, shape=[source_cells,
        source_out_channels]
            Source output signal features before the activation function
            :math:`\phi`.
        message_on_target : torch.Tensor, shape=[target_cells,
        target_out_channels]
            Target output signal features before the activation function
            :math:`\phi`.

        Returns
        -------
        phi(Y_s) : torch.Tensor, shape=[source_cells, source_out_channels]
            Source output signal features after the activation function
            :math:`\phi`.
        phi(Y_t) : torch.Tensor, shape=[target_cells, target_out_channels]
            Target output signal features after the activation function
            :math:`\phi`.
        """
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

    def attention(
        self, s_message: torch.Tensor, t_message: torch.Tensor
    ) -> tuple[torch.sparse.Tensor, torch.sparse.Tensor]:
        r"""Compute attention matrices :math:`A_s` and :math:`A_t`.

        ..  math::
            \begin{align}
                A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{\#\text{columns}(N)}
                e_{i,k}}, \\
                A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{\#\text{columns}(N^T)}
                f_{i,k}},
            \end{align}

        where,

        ..  math::
            \begin{align}
                e_{i,j} &= S(\text{LeakyReLU}([(X_s)_jW_s||(X_t)_iW_t]a)),\\
                f_{i,j} &= S(\text{LeakyReLU}([(X_t)_jW_t||(X_s)_iW_s][a[
                d_{s_{out}}:]||a[:d_{s_{out}}])).
            \end{align}

        Parameters
        ----------
        s_message : torch.Tensor, shape [n_source_cells, target_out_channels]
            Source message tensor. This is the result of the matrix
            multiplication of the cochain matrix :math:`X_s` with the weight
            matrix :math:`W_s`.
        t_message : torch.Tensor, shape [n_target_cells, source_out_channels]
            Target message tensor. This is the result of the matrix
            multiplication of the cochain matrix :math:`X_t` with the weight
            matrix :math:`W_t`.

        Returns
        -------
        A_s : torch.sparse, shape=[target_cells, source_cells].
        A_t : torch.sparse, shape=[source_cells, target_cells].
        """
        s_to_t = torch.cat(
            [s_message[self.source_indices], t_message[self.target_indices]], dim=1
        )

        t_to_s = torch.cat(
            [t_message[self.target_indices], s_message[self.source_indices]], dim=1
        )

        e = torch.sparse_coo_tensor(
            indices=torch.tensor(
                [self.target_indices.tolist(), self.source_indices.tolist()]
            ),
            values=F.leaky_relu(
                torch.matmul(s_to_t, self.att_weight),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(t_message.shape[0], s_message.shape[0]),
            device=self.get_device(),
        )

        f = torch.sparse_coo_tensor(
            indices=torch.tensor(
                [self.source_indices.tolist(), self.target_indices.tolist()]
            ),
            values=F.leaky_relu(
                torch.matmul(
                    t_to_s,
                    torch.cat(
                        [
                            self.att_weight[self.source_out_channels :],
                            self.att_weight[: self.source_out_channels],
                        ]
                    ),
                ),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(s_message.shape[0], t_message.shape[0]),
            device=self.get_device(),
        )

        if self.softmax:
            return torch.sparse.softmax(e, dim=1), torch.sparse.softmax(f, dim=1)

        return sparse_row_norm(e), sparse_row_norm(f)

    def forward(
        self, x_source: torch.Tensor, x_target: torch.Tensor, neighborhood: torch.Tensor, cci = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute forward pass.

        The forward pass of the Higher Order Attention Block for non-squared
        matrices (HBNS) is defined as

        ..  math::
            \begin{align}
                \text{HBNS}_N(X_s, X_t) = (Y_s, Y_t),
            \end{align}

        where the source and target outputs :math:`Y_s` and :math:`Y_t` are
        computed as

        ..  math::
            \begin{align}
                Y_s &= \phi((N^T \odot A_t) X_t W_t), \\
                Y_t &= \phi((N \odot A_s) X_s W_s ).
            \end{align}

        Parameters
        ----------
        x_source : torch.Tensor, shape=[source_cells, source_in_channels]
            Cochain matrix representation :math:`X_s` containing the signal
            features over the source cells.
        x_target : torch.Tensor, shape=[target_cells, target_in_channels]
            Cochain matrix :math:`X_t` containing the signal features over
            the target cells.
        neighborhood : torch.sparse, shape=[target_cells, source_cells]
            Neighborhood matrix :math:`N` inducing the HBNS block.

        Returns
        -------
        _ :math:`Y_s` : torch.Tensor, shape=[source_cells, source_out_channels]
            Output features of the layer for the source cells.
        _ :math:`Y_t` : torch.Tensor, shape=[target_cells, target_out_channels]
            Output features of the layer for the target cells.
        """

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
        if self.attention_flag:
            s_to_t_attention, t_to_s_attention = self.attention(s_message, t_message)

            neighborhood_s_to_t_att = torch.sparse_coo_tensor(
                indices=neighborhood_s_to_t.indices(),
                values=s_to_t_attention.values() * neighborhood_s_to_t.values(),
                size=neighborhood_s_to_t.shape,
                device=self.get_device(),
            )

            neighborhood_t_to_s_att = torch.sparse_coo_tensor(
                indices=neighborhood_t_to_s.indices(),
                values=t_to_s_attention.values() * neighborhood_t_to_s.values(),
                size=neighborhood_t_to_s.shape,
                device=self.get_device(),
            )

        else:
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
            message_on_source = self.source_aggregator(neighborhood_t_to_s_att, t_message) + torch.mm(cci.T, t_message2) + torch.mm(sparse_hadamard(cci.T, neighborhood_t_to_s), t_message3) 
            message_on_target = self.target_aggregator(neighborhood_s_to_t_att, s_message) + torch.mm(cci, s_message2) + torch.mm(sparse_hadamard(cci, neighborhood_s_to_t), s_message3)
        else:
            message_on_source = torch.mm(neighborhood_t_to_s_att, t_message) 
            message_on_target = torch.mm(neighborhood_s_to_t_att, s_message) 

        if self.update_func is None:
            return message_on_source, message_on_target

        return self.update(message_on_source, message_on_target)


class HBS(torch.nn.Module):
    r"""Higher Order Attention Block layer for squared neighborhoods (HBS).

    Let :math:`\mathcal{X}` be a combinatorial complex, we denote by
    :math:`\mathcal{C}^k(\mathcal{X}, \mathbb{R}^d)` the :math:`\mathbb{
    R}`-valued vector space of :math:`d`-dimensional signals over
    :math:`\Sigma^k`, the :math:`k`-th skeleton of :math:`\mathcal{X}`
    subject to a certain total order. Elements of this space are called
    :math:`k`-cochains of :math:`\mathcal{X}`. If :math:`d = 1`, we denote
    it by :math:`\mathcal{C}^k(\mathcal{X})`.

    Let :math:`N\colon \mathcal{C}^s(\mathcal{X}) \rightarrow \mathcal{C}^s(
    \mathcal{X})` be a cochain map endomorphism of the space of signals over
    :math:`\Sigma^s` of :math:`\mathcal{X}`. The matrix representation of
    :math:`N` has shape :math:`n_{cells} \times n_{cells}`, where :math:`n_{
    cells}` denotes the cardinality of :math:`\Sigma^s`.

    The higher order attention block induced by :math:`N` is the cochain map

    ..  math::
        \begin{align}
            \text{HBS}_N\colon \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{
            in}}}) \rightarrow \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{
            out}}}),
        \end{align}

    where :math:`d^{s_{in}}` and :math:`d^{s_{out}}` are the input and
    output dimensions of the HBS block, also denoted as
    source_in_channels and source_out_channels, respectively.

    :math:`\text{HBS}_N` is defined by

    ..  math::
        \phi(\sum_{p=1}^{\text{m_hop}}(N^p \odot A_p) X W_p )

    where :math:`X` is the cochain matrix representation of shape [n_cells,
    source_in_channels] under the canonical basis of :math:`\mathcal{C}^s(
    \mathcal{X},\mathbb{R}^{d^{s_{in}}})`, induced by the total order of
    :math:`\Sigma^s`, that contains the input features for each cell. The
    :math:`\odot` symbol denotes the Hadamard product, namely the entry-wise
    product, and :math:`\phi` is a non-linear activation function.
    :math:`W_p` is a learnable weight matrix of shape [source_in_channels,
    source_out_channels] for each :math:`p`, and :math:`A_p` is an attention
    matrix with the same dimensionality as the input neighborhood matrix
    :math:`N`, i.e., [n_cells, n_cells]. The indices :math:`(i,j)` of the
    attention matrix :math:`A_p` are computed as

    ..  math::
        A_p(i,j) = \frac{e_{i,j}^p}{\sum_{k=1}^{\#\text{columns}(N)} e_{i,k}^p}

    where

    ..  math::
        e_{i,j}^p = S(\text{LeakyReLU}([X_iW_p||X_jW_p]a_p))

    and where || denotes concatenation, :math:`a_p` is a learnable column
    vector of length :math:`2\cdot` source_out_channels, and :math:`S` is the
    exponential function if softmax is used and the identity function
    otherwise.

    This HBS class just contains the sparse implementation of the block.

    Notes
    -----
    HBS layers were introduced in [H23]_, Definitions 31 and 32.

    References
    ----------
    .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
        (2023) https://arxiv.org/abs/2206.00606.

    Parameters
    ----------
    source_in_channels : int
        Number of input features for the source cells.
    source_out_channels : int
        Number of output features for the source cells.
    negative_slope : float
        Negative slope of the LeakyReLU activation function.
    softmax : bool, optional
        Whether to use softmax in the computation of the attention matrix.
        Default is False.
    m_hop : int, optional
        Maximum number of hops to consider in the computation of the layer
        function. Default is 1.
    update_func : {None, 'sigmoid', 'relu', 'tanh'}, optional
        Activation function :math:`phi` in the computation of the output of
        the layer.
        If None, :math:`phi` is the identity function. Default is None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of W_p and :math:`a_p`.
        Default is 'xavier_uniform'.
    """

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

        self.source_aggregator = PNAAggregator(source_out_channels, source_out_channels)

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
        r"""Update signal features on each cell with an activation function.

        Implemented activation functions are sigmoid, ReLU and tanh.

        Parameters
        ----------
        message : torch.Tensor, shape=[n_cells, out_channels]
            Output signal features before the activation function :math:`\phi`.

        Returns
        -------
        _ : torch.Tensor, shape=[n_cells, out_channels]
            Output signal features after the activation function :math:`\phi`.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(message)
        if self.update_func == "relu":
            return torch.nn.functional.relu(message)
        if self.update_func == "tanh":
            return torch.nn.functional.tanh(message)

        raise RuntimeError(
            "Update function not recognized. Should be either sigmoid, relu or tanh."
        )

    def attention(
        self, message: torch.Tensor, A_p: torch.Tensor, a_p: torch.Tensor
    ) -> torch.sparse.Tensor:
        """Compute the attention matrix.

        Parameters
        ----------
        message : torch.Tensor, shape=[n_messages, source_out_channels]
            Message tensor. This is the result of the matrix multiplication
            of the cochain matrix :math:`X` with the learnable weights
            matrix :math:`W_p`.
        A_p : torch.sparse, shape=[n_cells, n_cells]
            Neighborhood matrix to the power p. Indicates how many paths of
            lenght p exist from cell :math:`i` to cell :math:`j`.
        a_p : torch.Tensor, shape=[2*source_out_channels, 1]
            Learnable attention weight vector.

        Returns
        -------
        att_p : torch.sparse, shape=[n_messages, n_messages].
            Represents the attention matrix :math:`A_p`.
        """
        n_messages = message.shape[0]
        source_index_i, source_index_j = A_p.coalesce().indices()
        s_to_s = torch.cat([message[source_index_i], message[source_index_j]], dim=1)
        e_p = torch.sparse_coo_tensor(
            indices=torch.tensor([source_index_i.tolist(), source_index_j.tolist()]),
            values=F.leaky_relu(
                torch.matmul(s_to_s, a_p), negative_slope=self.negative_slope
            ).squeeze(1),
            size=(n_messages, n_messages),
            device=self.get_device(),
        )
        return (
            torch.sparse.softmax(e_p, dim=1) if self.softmax else sparse_row_norm(e_p)
        )

    def forward(
        self, x_source: torch.Tensor, neighborhood: torch.Tensor, cci = None
    ) -> torch.Tensor:
        r"""Compute forward pass.

        The forward pass of the Higher Order Attention Block for squared
        neighborhood matrices is defined as:

        ..  math::
            \text{HBS}_N(X) = \phi(\sum_{p=1}^{\text{m_hop}}(N^p \odot A_p) X
            W_p ).

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_cells, source_in_channels]
            Cochain matrix representation :math:`X` whose rows correspond to
            the signal features over each cell following the order of the cells
            in :math:`\Sigma^s`.
        neighborhood : torch.sparse, shape=[n_cells, n_cells]
            Neighborhood matrix :math:`N`.

        Returns
        -------
        _ : Tensor, shape=[n_cells, source_out_channels]
            Output features of the layer.
        """
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
        
        if self.attention_flag: # We have not updated this since we are not using!!
            att = [
                self.attention(m_p, A_p, a_p)
                for m_p, A_p, a_p in zip(
                    message, m_hop_matrices, self.att_weight, strict=True
                )
            ]

            att_m_hop_matrices = [
                sparse_hadamard(A_p, att_p)
                for A_p, att_p in zip(m_hop_matrices, att, strict=True)
            ]

            message = [
                torch.mm(n_p, m_p)
                for n_p, m_p in zip(att_m_hop_matrices, message, strict=True)
            ]

        else: 
            message = [
                self.source_aggregator(n_p, m_p)
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