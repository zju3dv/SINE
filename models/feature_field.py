import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from models.activation import trunc_exp


class FeatureField(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=128,
        in_dim=30,
        out_dim=384,
        enable_torch_network=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if enable_torch_network == False:
            self.feat_net = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=out_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )
        else:
            model_list = []
            for i in range(num_layers):
                if i == 0:
                    layer = nn.Linear(in_dim, hidden_dim)
                else:
                    layer = nn.Linear(hidden_dim, hidden_dim)
                layer = nn.Sequential(layer, nn.ReLU(True))
                model_list.append(layer)
            model_list.append(nn.Linear(hidden_dim, out_dim))
            self.feat_net = nn.Sequential(*model_list)

    def forward(self, input):
        h = self.feat_net(input)
        assert torch.any(torch.isnan(h)) == False
        return h
