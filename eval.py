import os
import imageio
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import torch

import opt
from datasets import construct_dataset
from models import construct_models
from models.rendering import render_rays
from models import load_model_ckpts
from utils.feature_util import pca_transform, normalize_feat_color

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = opt.get_arg_parser()
    parser.add_argument(
        "--split", type=str, default="test_train", help="test or test_train"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="fine",
    )
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--out_type", type=str, nargs="+", default=["rgb"])
    return opt.load_config(parser)


@torch.no_grad()
def batched_inference(
    models,
    embeddings,
    rays,
    N_samples,
    N_importance,
    use_disp,
    chunk,
    white_back,
    attribute=None,
):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = render_rays(
            models,
            embeddings,
            rays[i : i + chunk],
            N_samples,
            use_disp,
            0,
            0,
            N_importance,
            chunk,
            white_back,
            test_time=True,
            attribute=attribute,
        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    if "fusion_layer" in models:
        orig_img = results[f"orig_rgb_fine"].view(h, w, 3)
        orig_img = orig_img.permute(2, 0, 1)  # (3, H, W)
        offset_img = results[f"offset_rgb_fine"].view(h, w, 3)
        offset_img = offset_img.permute(2, 0, 1)  # (3, H, W)
        img_pred = (
            model_list["fusion_layer"]
            .forward(torch.cat([orig_img.unsqueeze(0), offset_img.unsqueeze(0)], dim=1))
            .squeeze(0)
        )
        results["rgb_fine"] = img_pred.permute(1, 2, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    args.img_wh[0] = int(args.img_wh[0] / args.downscale)
    args.img_wh[1] = int(args.img_wh[1] / args.downscale)
    w, h = args.img_wh
    dataset = construct_dataset(args, split=args.split)

    model_list, embeddings = construct_models(args)
    load_model_ckpts(args.ckpt_path, model_list)

    for key in model_list.keys():
        model_list[key].cuda().eval()

    imgs = []
    dir_name = f"results/{args.dataset_name}/{args.exp_name}"
    dir_name += f"/{args.split}" if args.group is None else f"/{args.group}"
    if "feature" in args.out_type:
        feature_dir = os.path.join(dir_name, "feature")
        os.makedirs(feature_dir, exist_ok=True)

    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample["rays"].cuda()
        results = batched_inference(
            model_list,
            embeddings,
            rays,
            args.N_samples,
            args.N_importance,
            args.use_disp,
            args.chunk,
            dataset.white_back,
            attribute=args.editing_attribute,
        )
        img_pred = results["rgb_fine"].view(h, w, 3).cpu().numpy()

        pred_mask = (
            torch.clamp(results[f"opacity_{args.type}"], 1e-3, 1 - 1e-3)
            .view(h, w)
            .detach()
            .cpu()
        )

        if "feature_field" in results and "feature" in args.out_type:
            feat = results["feature_field"].detach().cpu().numpy()
            predfeat_rgb = normalize_feat_color(pca_transform(feat)[0]).reshape(h, w, 3)
            Image.fromarray(predfeat_rgb).save(
                os.path.join(feature_dir, f"feat_{i:03d}.png")
            )

        if "rgb" in args.out_type:
            img_pred_ = (img_pred * 255).astype(np.uint8)
            imgs += [img_pred_]
            imageio.imwrite(os.path.join(dir_name, f"{i:03d}.png"), img_pred_)

    if "rgb" in args.out_type:
        imageio.mimwrite(
            os.path.join(dir_name, f"{args.exp_name}.mp4"),
            imgs,
            fps=args.fps,
            quality=10,
        )
