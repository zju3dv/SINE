import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from models.activation import trunc_exp
from utils.model_util import posenc_window


class AffineMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine_rot = nn.Parameter(torch.eye(3, 3))
        self.affine_trans = nn.Parameter(torch.zeros(3, 1))

    def forward(self, x: torch.Tensor, mask=None):

        if x.dim() == 3:
            # [3, H, W]
            C, H, W = x.shape
            assert C == 3, "assert RGB channel"
            x_ = x.view(1, C, -1).permute(2, 1, 0)  # [H*W, 3, 1]
            out = self.affine_rot.view(1, 3, 3) @ x_
            out = out + self.affine_trans.view(1, 3, 1)
            out = out.view(H, W, 3).permute(2, 0, 1)
            if mask is not None:
                mask_val = mask.float().view(1, H, W)
                out = out * mask_val + (1 - mask_val)

        elif x.dim() == 4:
            B, C, H, W = x.shape
            assert C == 3, "assert RGB channel"
            x_ = x.permute(0, 2, 3, 1).view(B * H * W, 3, 1)
            out = self.affine_rot.view(1, 3, 3) @ x_
            out = out + self.affine_trans.view(1, 3, 1)
            out = out.view(B, H, W, 3).permute(0, 3, 1, 2)
            if mask is not None:
                mask_val = mask.float().view(B, 1, H, W)
                out = out * mask_val + (1 - mask_val)
        else:
            assert False, "Unkown input dimension, x.dim = {}".format(x.dim())
        out = out.clip(0, 1)
        return out


class ColorField(nn.Module):
    def __init__(
        self,
        num_layers=2,
        hidden_dim=128,
        num_levels=16,
        n_features_per_level=2,
        bound=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.bound = bound

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder_dim = num_levels * n_features_per_level

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=self.encoder_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    def forward(self, input, alpha=None):
        x = input[:, :3]

        # x: [N, 3], in [-bound, bound]

        # color
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        if alpha is not None:
            alpha = max(1.0, alpha)
            window = posenc_window(0, self.encoder_dim // 2, alpha).to(x.device)
            window = torch.repeat_interleave(window, 2)
            x = x * window

        h = self.rgb_net(x)

        outputs = h
        outputs = torch.sigmoid(outputs) * 2 - 1  # move to [-1, 1]

        return outputs
