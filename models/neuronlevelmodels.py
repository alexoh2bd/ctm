import torch
import math
import torch.nn as nn
import numpy as np
from modules import Identity, Squeeze


class nlm(nn.module):
    """
    g_theta_d (3) in paper
    Parameters for the NLM:
    in_dims
    out_dims
    N:dim_model

    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        dim_model,
        batch: int,
        ticks: int = 0,
        memory_len: int = 10,
        dropout=0,
        T=1.0,
    ):
        super().__init__()
        # N is the number of neurons (d_model), in_dims is the history length (memory_length)

        # If we want to normalize layers
        # self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm, else Identity()
        # self.do_norm=  do_norm

        self.dim_model = dim_model
        self.batch = batch
        self.M = memory_len
        self.T = T

        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.register_parameter(
            "w1",
            nn.Parameter(
                torch.empty((in_dims, out_dims, dim_model)).uniform_(
                    -1 / math.sqrt(in_dims + out_dims),
                    1 / math.sqrt(in_dims + out_dims),
                ),
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "b1",
            nn.Parameter(torch.zeros((1, dim_model, out_dims)), requires_grad=True),
        )

        self.register_parameter("T", nn.Parameter(torch.Tensor([T])))
        """
        self.weights1 = Parameter(shape=(d_hidden, d_model))
        self.bias1 = np.zeros(shape = (1, d_hidden, d_model))
        self.weights2 = Parameter(shape=(d_hidden, d_model))
        self.bias2 = np.zeros(shape = (1, d_hidden, d_model))

        """

    def forward(self, inputs):
        # assert inputs.shape = (b, d, M)
        # b: batch_len, D: Dim model, M: Mem_length

        out = self.dropout(inputs)
        inputs = self.pre_acts_history[-self.M :]
        out = np.einsum("bdM,Mhd->bdh", inputs, self.weights1) + self.bias1
        out = out.squeeze(-1) / self.T
        return out
