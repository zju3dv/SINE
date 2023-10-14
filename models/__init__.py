import os
from models.nerf import Embedding, NeRF
from models.unet.skip import define_G
from models.color_field_tcnn import AffineMap
from models.rendering import render_rays
from models.nerf_tcnn import NeRF_TCNN
from models.feature_field import FeatureField
from models.color_field_tcnn import ColorField
from models.deformation_field import get_models as get_deform_models

from utils import load_ckpt
from utils.train_util import find_substring_in_list


def load_model_ckpt(ckpt_path, model_list, is_model_loaded={}):
    verbose_str = ""
    for key in model_list.keys():
        if is_model_loaded.get(key, False):
            continue
        if key == "inv_deform":
            name = "inv_deformation_model"
        elif key == "deform":
            name = "deformation_model"
        else:
            name = key

        is_model_loaded[key] = load_ckpt(model_list[key], ckpt_path, model_name=name)
        if is_model_loaded[key]:
            verbose_str += f"[Info] {key} loaded\n"
        else:
            verbose_str += f"[Warning] {key} ckpt missing\n"

    print(verbose_str)
    return is_model_loaded


def load_model_ckpts(model_ckpt, model_list):
    is_model_loaded = {}
    model_ckpt_list = [model_ckpt] if isinstance(model_ckpt, str) else model_ckpt
    for i in range(len(model_ckpt_list)):
        model_ckpt = model_ckpt_list[i]
        assert os.path.exists(model_ckpt)
        is_model_loaded = load_model_ckpt(model_ckpt, model_list, is_model_loaded)


def construct_models(args):
    tcnn_bound = args.tcnn_bound
    depth_deformfield = args.depth_deformfield
    width_deformfield = args.width_deformfield
    dim_deformxyz_emb = args.dim_deformxyz_emb
    embeddings = {}
    model_list = {}
    if "nerf" in args.model_type:
        embeddings["xyz"] = Embedding(3, 10)
        embeddings["dir"] = Embedding(3, 4)
        nerf_coarse = NeRF()
        if args.N_importance > 0:
            nerf_fine = NeRF()
        model_list["nerf_coarse"] = nerf_coarse
        model_list["nerf_fine"] = nerf_fine
    elif "nerf_tcnn" in args.model_type:
        embeddings["xyz"] = Embedding(3, 0)
        embeddings["dir"] = Embedding(3, 0)
        nerf_coarse = NeRF_TCNN(bound=tcnn_bound)
        if args.N_importance > 0:
            nerf_fine = NeRF_TCNN(bound=tcnn_bound)
        model_list["nerf_coarse"] = nerf_coarse
        model_list["nerf_fine"] = nerf_fine

    if "feature_field" in args.model_type:
        if "nerf" in args.model_type:
            feature_field = FeatureField(in_dim=nerf_coarse.W)
        else:
            feature_field = FeatureField()
        model_list["feature_field"] = feature_field

    if "color_field" in args.model_type:
        embeddings["color_xyz"] = Embedding(3, 0)
        embeddings["color_dir"] = Embedding(3, 0)
        color_field = ColorField()
        model_list["color_field"] = color_field

    if "fusion_layer" in args.model_type:
        fusion_layer = define_G(
            model_receipt=args.unet_receipt,
            init_type="xavier",
            init_gain=0.02,
            num_input_channels=6 if args.fusion_split_channel else 3,
        )
        model_list["fusion_layer"] = fusion_layer

    if "affine_map" in args.model_type:
        affine_map = AffineMap()
        model_list["affine_map"] = affine_map

    # deform
    deform_model_name = find_substring_in_list("deform", args.model_type)
    if len(deform_model_name) > 0:
        deformation_model, deform_embedding = get_deform_models(
            deform_model_name[0],
            dim_deformxyz_emb,
            depth_deformfield,
            width_deformfield,
            True,
        )
        embeddings.update(deform_embedding)
        model_list["deform"] = deformation_model

    # inv_deform
    invdeform_model_name = find_substring_in_list("inv_deform", args.model_type)
    if len(invdeform_model_name) > 0:
        inv_deformation_model = get_deform_models(
            invdeform_model_name[0],
            dim_deformxyz_emb,
            depth_deformfield,
            width_deformfield,
        )
        model_list["inv_deform"] = inv_deformation_model

    return model_list, embeddings
