import torch.nn as nn
import torch
import numpy as np


class Identity(nn.Module):
    """
    Identity Module.

    Returns the input tensor unchanged. Useful as a placeholder or a no-op layer
    in nn.Sequential containers or conditional network parts.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Squeeze(nn.Module):
    """
    Module to squeeze a dim size 1 from tensor
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SynapseUNet(nn.Module):
    """
    U-Net Style Synapse model (f_theta_1 in the paper)

    This module implements synaptic connections between neurons in the CTM's latent space.
    It concatenates previous post-activation state z^t with attention output o^t
    to produce pre-activations (a^t) for the next internal tick.
    Model stores pre and post-activation histories
    """

    def __init__(self, out_dims, depth, minimum_width=16, dropout=0.0):
        super().__init__()
        self.out_dims = out_dims
        self.depth = depth

        # Tracks the `width` of blocks from minimum_width to depth-1 blocks
        widths = np.linspace(out_dims, minimum_width, depth)

        # first layer
        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])), nn.LayerNorm(int(widths[0])), nn.SiLU()
        )

        # Establish a widths deep neural network
        num_blocks = len(widths) - 1
        self.up_projections = nn.ModuleList()
        self.down_projections = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(1, len(num_blocks)):
            self.down_projections.append(
                nn.Sequential(
                    nn.DropOut(dropout),
                    nn.Linear(int(widths[i]), int(widths[i + 1])),
                    nn.LayerNorm(int(widths[i + 1])),
                    nn.SiLU(),
                )
            )

            self.up_projections.append(
                (
                    nn.Sequential(
                        nn.DropOut(dropout),
                        nn.Linear(int(widths[i + 1]), int(widths[i])),
                        nn.LayerNorm(int(widths[i])),
                        nn.SiLU(),
                    )
                )
            )

            self.skip_connections.append(nn.LayerNorm(widths[i]))

    def forward(self, x):
        out = self.first_projection(x)

        down_outputs = [out]
        num_blocks = len(self.up_projections)
        for i in range(num_blocks):
            down_outputs.append(self.down_projections[i](down_outputs[-1]))

        up_output = down_outputs[-1]

        # Apply Up projection, then layernorm across skip connection and up projection
        for j in range(num_blocks):
            up_idx = num_blocks - 1 - j
            up_output = self.up_projections[up_idx](up_output)
            up_output = self.skip_connections[up_idx](up_output + down_outputs[up_idx])
        return up_output
