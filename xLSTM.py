"""Implementation of mLSTMCell model and sLSTMCell model as described in the paper.

xLSTM: Extended Long Short-Term Memory
https://arxiv.org/abs/2405.04517

This module provides an implementation of the sLSTMCell model, a variant of LSTM cells proposed in the xLSTM paper.

Attributes:
    input_size (int): The size of the input features.
    hidden_size (int): The size of the hidden state.
    bias (bool): Indicates whether bias is included in the calculations.

Methods:
    forward(x, internal_state): Performs a forward pass of the sLSTMCell model.
    init_hidden(batch_size): Initializes the hidden state of the model.

References:
    "xLSTM: Extended Long Short-Term Memory" - https://arxiv.org/abs/2405.04517
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class sLSTMCell(nn.Module):
    """Implements the sLSTMCell model as described in the xLSTM paper.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        bias (bool): Indicates whether bias is included in the calculations.

    Methods:
        forward(x, internal_state): Performs a forward pass of the sLSTMCell model.
        init_hidden(batch_size): Initializes the hidden state of the model.

    References:
        - xLSTM: Extended Long Short-Term Memory
          https://arxiv.org/abs/2405.04517
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()

        # Store the input and hidden size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Combine the Weights and Recurrent weights into a single matrix
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.randn(self.input_size + self.hidden_size, 4 * self.hidden_size)
            ),
            requires_grad=True,
        )
        # Combine the Bias into a single matrix
        if self.bias:
            self.B = nn.Parameter(
                (torch.zeros(4 * self.hidden_size)), requires_grad=True
            )

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Forward pass of the sLSTMCell model

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            internal_state (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the hidden state, cell state, normalization state and stabilization state
        """
        # Unpack the internal state
        h, c, n, m = internal_state  # (batch_size, hidden_size)

        # Combine the weights and the input
        combined = torch.cat((x, h), dim=1)  # (batch_size, input_size + hidden_size)
        # Calculate the linear transformation
        gates = torch.matmul(combined, self.W)  # (batch_size, 4 * hidden_size)

        # Add the bias if included
        if self.bias:
            gates += self.B

        # Split the gates into the input, forget, output and stabilization gates
        z_tilda, i_tilda, f_tilda, o_tilda = torch.split(gates, self.hidden_size, dim=1)

        # Calculate the activation of the states
        z_t = torch.tanh(z_tilda)  # (batch_size, hidden_size)
        # Exponential activation of the input gate
        i_t = torch.exp(i_tilda)  # (batch_size, hidden_size)
        # Exponential activation of the forget gate
        f_t = torch.sigmoid(f_tilda)  # (batch_size, hidden_size)

        # Sigmoid activation of the output gate
        o_t = torch.sigmoid(o_tilda)  # (batch_size, input_size)
        # Calculate the stabilization state
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))  # (batch_size, hidden_size)
        # Calculate the input stabilization state
        i_prime = torch.exp(i_tilda - m_t)  # (batch_size, hidden_size)

        # Calculate the new internal states
        c_t = f_t * c + i_prime * z_t  # (batch_size, hidden_size)
        n_t = f_t * n + i_prime  # (batch_size, hidden_size)

        # Calculate the stabilized hidden state
        h_tilda = c_t / n_t  # (batch_size, hidden_size)

        # Calculate the new hidden state
        h_t = o_t * h_tilda  # (batch_size, hidden_size)
        return h_t, (
            h_t,
            c_t,
            n_t,
            m_t,
        )  # (batch_size, hidden_size), (batch_size, hidden_size), (batch_size, hidden_size), (batch_size, hidden_size)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden state of the model.

        Args:
            batch_size (int): Batch size of the input tensor.

        Returns:
            tuple: Tuple containing the initialized hidden state, cell state, normalization state, and stabilization state.
        """
        return (
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )


class sLSTM(nn.Module):
    """Implements the sLSTM model as described in the xLSTM paper.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of layers in the model.
        bias (bool): Indicates whether bias is included in the calculations.

    Methods:
        forward(x, hidden_states): Performs a forward pass of the sLSTM model.
        init_hidden(batch_size): Initializes the hidden state of the model.

    References:
        - xLSTM: Extended Long Short-Term Memory
          https://arxiv.org/abs/2405.04517
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        """
        Initializes the sLSTM.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.
            num_layers (int): The number of layers in the model.
            bias (bool, optional): Indicates whether bias is included in the calculations. Default is True.
            batch_first (bool, optional): Indicates whether the input tensor is batch first. Default is False.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                sLSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Performs a forward pass of the sLSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_size) if batch_first is False,
                              or (batch_size, seq_len, input_size) if batch_first is True.
            hidden_states (list, optional): List of hidden states for each layer of the model. If None, hidden states are initialized to zero.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size)
            tuple: Tuple containing the hidden states at each layer and each time step.
        """
        # Permute the input tensor if batch_first is True
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # Initialize the hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Check if the hidden states are of the correct length
            if len(hidden_states) != self.num_layers:
                raise ValueError(
                    f"Expected hidden states of length {self.num_layers}, but got {len(hidden_states)}"
                )
            if any(state[0].size(0) != x.size(1) for state in hidden_states):
                raise ValueError(
                    f"Expected hidden states of batch size {x.size(1)}, but got {hidden_states[0][0].size(0)}"
                )

        H, C, N, M = [], [], [], []

        for layer, cell in enumerate(self.cells):
            lh, lc, ln, lm = [], [], [], []
            for t in range(x.size(0)):
                h_t, hidden_states[layer] = (
                    cell(x[t], hidden_states[layer])
                    if layer == 0
                    else cell(H[layer - 1][t], hidden_states[layer])
                )
                lh.append(h_t)
                lc.append(hidden_states[layer][0])
                ln.append(hidden_states[layer][1])
                lm.append(hidden_states[layer][2])

            H.append(torch.stack(lh, dim=0))
            C.append(torch.stack(lc, dim=0))
            N.append(torch.stack(ln, dim=0))
            M.append(torch.stack(lm, dim=0))

        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        N = torch.stack(N, dim=0)
        M = torch.stack(M, dim=0)

        return H[-1], (H, C, N, M)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Initializes the hidden state of the model.

        Args:
            batch_size (int): Batch size of the input tensor.

        Returns:
            list: List containing the initialized hidden states for each layer.
        """
        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]


class mLSTMCell(nn.Module):
    """Implements the mLSTMCell model as described in the xLSTM paper.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        bias (bool): Indicates whether bias is included in the calculations.

    Methods:
        forward(x, internal_state): Performs a forward pass of the mLSTMCell model.
        init_hidden(batch_size): Initializes the hidden state of the model.

    References:
        - xLSTM: Extended Long Short-Term Memory
          https://arxiv.org/abs/2405.04517
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        """Initializes the mLSTMCell model.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.
            bias (bool, optional): Indicates whether bias is included in the calculations. Defaults to True.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize weights and biases
        self.W_i = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_f = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_o = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_q = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_k = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_v = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )

        if self.bias:
            self.B_i = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_f = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_o = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_q = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_k = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_v = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the mLSTMCell model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            internal_state (tuple[torch.Tensor, torch.Tensor]): Tuple containing the covariance matrix, normalization state, and stabilization state.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output tensor and updated internal state.
        """
        # Get the internal state
        C, n, m = internal_state

        #  Calculate the input, forget, output, query, key and value gates
        i_tilda = (
            torch.matmul(x, self.W_i) + self.B_i
            if self.bias
            else torch.matmul(x, self.W_i)
        )
        f_tilda = (
            torch.matmul(x, self.W_f) + self.B_f
            if self.bias
            else torch.matmul(x, self.W_f)
        )
        o_tilda = (
            torch.matmul(x, self.W_o) + self.B_o
            if self.bias
            else torch.matmul(x, self.W_o)
        )
        q_t = (
            torch.matmul(x, self.W_q) + self.B_q
            if self.bias
            else torch.matmul(x, self.W_q)
        )
        k_t = (
            torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size))
            + self.B_k
            if self.bias
            else torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size))
        )
        v_t = (
            torch.matmul(x, self.W_v) + self.B_v
            if self.bias
            else torch.matmul(x, self.W_v)
        )

        # Exponential activation of the input gate
        i_t = torch.exp(i_tilda)
        f_t = torch.sigmoid(f_tilda)
        o_t = torch.sigmoid(o_tilda)

        # Stabilization state
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))
        i_prime = torch.exp(i_tilda - m_t)

        # Covarieance matrix and normalization state
        C_t = f_t.unsqueeze(-1) * C + i_prime.unsqueeze(-1) * torch.einsum(
            "bi, bk -> bik", v_t, k_t
        )
        n_t = f_t * n + i_prime * k_t

        normalize_inner = torch.diagonal(torch.matmul(n_t, q_t.T))
        divisor = torch.max(
            torch.abs(normalize_inner), torch.ones_like(normalize_inner)
        )
        h_tilda = torch.einsum("bkj,bj -> bk", C_t, q_t) / divisor.view(-1, 1)
        h_t = o_t * h_tilda

        return h_t, (C_t, n_t, m_t)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state of the model.

        Args:
            batch_size (int): Batch size of the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized covariance matrix and normalization state.
        """
        return (
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )


class mLSTM(nn.Module):
    """Implements the mLSTM model as described in the xLSTM paper.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of layers in the model.
        bias (bool): Indicates whether bias is included in the calculations.

    Methods:
        forward(x, hidden_states): Performs a forward pass of the sLSTM model.
        init_hidden(batch_size): Initializes the hidden state of the model.

    References:
        - xLSTM: Extended Long Short-Term Memory
          https://arxiv.org/abs/2405.04517
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        """
        Initializes the sLSTM.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.
            num_layers (int): The number of layers in the model.
            bias (bool, optional): Indicates whether bias is included in the calculations. Default is True.
            batch_first (bool, optional): Indicates whether the input tensor is batch first. Default is False.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                mLSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass of the sLSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_size) if batch_first is False,
                              or (batch_size, seq_len, input_size) if batch_first is True.
            hidden_states (list, optional): List of hidden states for each layer of the model. If None, hidden states are initialized to zero.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size)
            tuple: Tuple containing the hidden states at each layer and each time step.
        """
        # Permute the input tensor if batch_first is True
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Check if the hidden states are of the correct length
            if len(hidden_states) != self.num_layers:
                raise ValueError(
                    f"Expected hidden states of length {self.num_layers}, but got {len(hidden_states)}"
                )
            if any(state[0].size(0) != x.size(1) for state in hidden_states):
                raise ValueError(
                    f"Expected hidden states of batch size {x.size(1)}, but got {hidden_states[0][0].size(0)}"
                )

        H, C, N, M = [], [], [], []

        for layer, cell in enumerate(self.cells):
            lh, lc, ln, lm = [], [], [], []
            for t in range(x.size(0)):
                h_t, hidden_states[layer] = (
                    cell(x[t], hidden_states[layer])
                    if layer == 0
                    else cell(H[layer - 1][t], hidden_states[layer])
                )
                lh.append(h_t)
                lc.append(hidden_states[layer][0])
                ln.append(hidden_states[layer][1])
                lm.append(hidden_states[layer][2])

            H.append(torch.stack(lh, dim=0))
            C.append(torch.stack(lc, dim=0))
            N.append(torch.stack(ln, dim=0))
            M.append(torch.stack(lm, dim=0))

        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        N = torch.stack(N, dim=0)
        M = torch.stack(M, dim=0)

        return H[-1], (H, C, N, M)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Initializes the hidden state of the model.

        Args:
            batch_size (int): Batch size of the input tensor.

        Returns:
            list: List containing the initialized hidden states for each layer.
        """
        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]
