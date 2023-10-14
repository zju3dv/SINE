import torch
from torch import nn
from models.nerf import Embedding


class DeformationField(nn.Module):
    def __init__(
        self,
        D=4,
        W=256,
        in_channels_xyz=3 + 3 * 2 * 4,
        out_channels_translation=3,
        padding_with_rgb=False,
    ):
        super(DeformationField, self).__init__()
        print("[Info] create DeformationField with {} depth, {} width".format(D, W))
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.out_channels_translation = out_channels_translation

        # xyz encoding layers
        for i in range(D - 1):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            nn.init.xavier_uniform_(layer.weight)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"layer_{i+1}", layer)

        self.translation = nn.Linear(W, out_channels_translation)
        nn.init.uniform_(self.translation.weight, a=-1e-4, b=1e-4)
        nn.init.uniform_(self.translation.bias, a=-1e-4, b=1e-4)
        if padding_with_rgb:
            self.rgb = nn.Sequential(
                nn.Linear(W, 3), nn.Sigmoid()
            )  # Note: the rgb layer is deprecated now
            nn.init.uniform_(self.rgb[0].weight, b=1e-4)

    def forward(self, x):
        xyz_ = x
        for i in range(self.D - 1):
            xyz_ = getattr(self, f"layer_{i+1}")(xyz_)
        out = {}
        trans = self.translation(xyz_)
        out["xyz"] = trans
        return out


def inference_deformation(
    model,
    embedding_xyz,
    sample_input,
    attr_embd=None,
    return_offset=False,
):
    xyz_embed = embedding_xyz(sample_input)
    if attr_embd is not None:
        input_embd = torch.cat([xyz_embed, attr_embd], dim=-1)
    else:
        input_embd = xyz_embed
    input_shape = input_embd.shape[:-1]
    input_embd = input_embd.reshape(-1, input_embd.shape[-1])
    offset_xyz = model(input_embd)["xyz"]
    offset_xyz = offset_xyz.reshape(*input_shape, offset_xyz.shape[-1])
    sample_deform = sample_input + offset_xyz

    if return_offset == False:
        return sample_deform
    else:
        return sample_deform, offset_xyz


def get_models(
    model_type,
    dim_emb,
    depth_mm,
    width_mm,
    with_embedding=False,
):
    input_dim = 3 + 3 * 2 * dim_emb
    if "attribute" in model_type:
        input_dim += 1 + 1 * 2 * 4
    if "inv_deform" in model_type:
        deformation_model = DeformationField(
            in_channels_xyz=input_dim,
            D=depth_mm,
            W=width_mm,
        )
    else:
        deformation_model = DeformationField(
            in_channels_xyz=input_dim,
            D=depth_mm,
            W=width_mm,
            padding_with_rgb=True,
        )

    if with_embedding:
        embeddings = {}
        embeddings["deform_xyz"] = Embedding(3, dim_emb)
        if "attribute" in model_type:
            embeddings["deform_attribute"] = Embedding(1, 4)

        return deformation_model, embeddings
    else:
        return deformation_model
