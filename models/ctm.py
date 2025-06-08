import torch.nn as nn
from neuronlevelmodels import nlm
from modules import Squeeze, SynapseUNet
from resnet18 import resnet18


class ctm(nn.Module):
    def __init__(
        self,
        iterations: int,
        model_dims: int,
        num_heads: int,
        in_dims: int,
        num_synch_out: int,
        num_synch_action: int,
        synapse_depth: int,
        memory_len: int,
        deep_nlms: bool,
        hidden_memory_dims: int,
        out_dims: int,
        dropout: float,
    ):
        self.iters = iterations
        self.model_dims = model_dims
        self.num_heads = num_heads
        self.in_dims = in_dims
        self.num_synch_out = num_synch_out
        self.num_synch_action = num_synch_action
        self.synapse_depth = synapse_depth
        self.memory_len = memory_len
        self.hidden_memory_dims = hidden_memory_dims
        self.out_dims = out_dims
        self.dropout = dropout
        self.backbone_type = "resnet-18"
        self.set_initial_rgb()

    def get_neuron_level_models(
        self,
        deep_nlms,
        do_layernorm_nlm,
        memory_length,
        memory_hidden_dims,
        dim_model,
        dropout,
    ):
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    nlm(
                        in_dims=memory_length,
                        out_dims=2 * memory_hidden_dims,
                        N=dim_model,
                        do_norm=do_layernorm_nlm,
                        dropout=dropout,
                    ),
                    nn.GLU(),
                    nlm(
                        in_dims=memory_length,
                        out_dims=2 * memory_hidden_dims,
                        N=dim_model,
                        do_norm=do_layernorm_nlm,
                        dropout=dropout,
                    ),
                    nn.GLU(),
                    Squeeze(-1),
                )
            )

        else:
            return nn.Sequential(
                nn.Sequential(
                    nlm(
                        in_dims=memory_length,
                        out_dims=2 * memory_hidden_dims,
                        N=dim_model,
                        do_norm=do_layernorm_nlm,
                        dropout=dropout,
                    ),
                    nn.GLU(),
                    Squeeze(-1),
                )
            )

    def compute_features(self, x):

        return

    def get_synapses(self, model_dimensions, synapse_depth, minimum_width, dropout):
        return SynapseUNet(
            model_dimensions,
            synapse_depth,
            minimum_width=minimum_width,
            dropout=dropout,
        )

    def set_initial_rgb(self):
        """Set the initial RGB processing module based on the backbone type."""
        if "resnet" in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1)  # Adapts input channels lazily
        else:
            self.initial_rgb = nn.Identity()
