import numpy as np
import torch
from einops import rearrange, reduce, repeat
import contextlib
from models.deformation_field import inference_deformation


__all__ = ["render_rays"]

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, "n1 n2 -> n1 1", "sum")  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(
        torch.stack([below, above], -1), "n1 n2 c -> n1 (n2 c)", c=2
    )
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def summary_out_chunk(out_chunk, N_rays, N_samples_, weights_only=False):
    if weights_only:
        patch_shape = (N_rays, N_samples_)
    else:
        patch_shape = (N_rays, N_samples_, 4)

    if "feature_field" in out_chunk:
        dim = out_chunk["feature_field"].shape[-1]
        out_chunk["feature_field"] = out_chunk["feature_field"].view(
            N_rays, N_samples_, dim
        )

    if "nerf" in out_chunk:
        res = out_chunk["nerf"].view(patch_shape)
    else:
        raise NotImplementedError
    return res


def render_rays(
    models,
    embeddings,
    rays,
    N_samples=64,
    use_disp=False,
    perturb=0,
    noise_std=1,
    N_importance=0,
    chunk=1024 * 32,
    white_back=False,
    test_time=False,
    alpha_embedding=None,
    **dummy_kwargs,
):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    # control gradient
    nerf_tcnn_no_grad = dummy_kwargs.get("nerf_tcnn_no_grad", False)
    feature_field_no_grad = dummy_kwargs.get("feature_field_no_grad", False)

    def point_inference(
        models, embeddings, xyz_chunk, dir_chunk, weights_only, out_chunk
    ):

        if "deform" in models:
            in_attr = dummy_kwargs.get("attribute", None)
            attributes_chunk = torch.ones(*xyz_chunk.shape[:-1], 1).to(
                xyz_chunk.device
            ) * (in_attr if in_attr is not None else 1.0)
            xyz_chunk, xyz_offset = inference_deformation(
                models["deform"],
                embeddings["deform_xyz"],
                xyz_chunk,
                attr_embd=embeddings["deform_attribute"](attributes_chunk)
                if attributes_chunk is not None
                else None,
                return_offset=True,
            )

        xyz_embedded = embeddings["xyz"](xyz_chunk)
        dir_embedded = embeddings["dir"](dir_chunk)
        if not weights_only:
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
        else:
            xyzdir_embedded = xyz_embedded

        if "feature_field" in models and "nerf" in models:
            with torch.no_grad() if nerf_tcnn_no_grad else contextlib.nullcontext():
                model_out, geo_feat = models["nerf"](
                    xyzdir_embedded, sigma_only=weights_only, return_feat=True
                )
            with torch.no_grad() if feature_field_no_grad else contextlib.nullcontext():
                feat_out = models["feature_field"](geo_feat)
            out_chunk["nerf"] += [model_out]
            out_chunk["feature_field"] += [feat_out]
        elif "nerf" in models:
            with torch.no_grad() if nerf_tcnn_no_grad else contextlib.nullcontext():
                out_chunk["nerf"] += [
                    models["nerf"](xyzdir_embedded, sigma_only=weights_only)
                ]

        if "color_field" in models:
            xyz_color_input = embeddings["color_xyz"](xyz_chunk)
            input_colorfield = xyz_color_input
            out_chunk["color_field"] += [
                models["color_field"](
                    input_colorfield,
                    alpha_embedding["color"] if alpha_embedding is not None else None,
                )
            ]

    def inference(
        models,
        embeddings,
        xyz_,
        dir_,
        z_vals,
        weights_only=False,
    ):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        embedding_xyz = embeddings["xyz"]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        dir_samples = torch.repeat_interleave(dir_, repeats=N_samples_, dim=0)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunk = {}
        for name in models.keys():
            out_chunk[name] = []

        for i in range(0, B, chunk):
            point_inference(
                models,
                embeddings,
                xyz_[i : i + chunk],
                dir_samples[i : i + chunk],
                weights_only,
                out_chunk,
            )

        for key in out_chunk.keys():
            if len(out_chunk[key]) > 0:
                out_chunk[key] = torch.cat(out_chunk[key], 0)

        network_out = summary_out_chunk(
            out_chunk, N_rays, N_samples_, weights_only=weights_only
        )
        if weights_only:
            sigmas = network_out
        else:
            rgbsigma = network_out
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            orig_rgbs = rgbs
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)
            if "color_field" in out_chunk:
                offset_rgb = out_chunk["color_field"].view(N_rays, N_samples_, -1)
                rgbs = torch.clamp(rgbs + offset_rgb, 0, 1)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(
            deltas[:, :1]
        )  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(
            -deltas * torch.relu(sigmas + noise)
        )  # (N_rays, N_samples_)
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # [1, a1, a2, ...]
        weights = (
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]
        )  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return {"weights": weights}

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        orig_rgb_final = torch.sum(weights.unsqueeze(-1) * orig_rgbs, -2)  # (N_rays, 3)
        if "color_field" in out_chunk:
            offset_rgb_final = torch.sum(
                weights.unsqueeze(-1) * offset_rgb, -2
            )  # (N_rays, 3)

        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
            orig_rgb_final = orig_rgb_final + 1 - weights_sum.unsqueeze(-1)

        ret = {
            "rgb_final": rgb_final,
            "orig_rgb_final": orig_rgb_final,
            "depth_final": depth_final,
            "weights": weights,
        }

        if "color_field" in out_chunk:
            ret["offset_rgb_final"] = offset_rgb_final

        if "feature_field" in out_chunk:
            feat_final = torch.sum(
                weights.unsqueeze(-1) * out_chunk["feature_field"], -2
            )
            ret["feature_field"] = feat_final

        if "color_field" in out_chunk:
            ret["offset_rgb"] = out_chunk["color_field"].view(N_rays, N_samples_, -1)

        return ret

    def prepare_models_dict(models, common_suffix=""):
        model_dict = {}
        for key in models.keys():
            if key.endswith(common_suffix):
                name = key[: -(len(common_suffix) + 1)]
                model_dict[name] = models[key]
            elif key == "feature_field":
                model_dict[key] = models[key]
            elif key == "color_field":
                model_dict[key] = models[key]
            elif key == "deform":
                model_dict[key] = models[key]
        return model_dict

    def collect_results(ret, common_suffix):
        result = {
            f"rgb_{common_suffix}": ret["rgb_final"],
            f"orig_rgb_{common_suffix}": ret["orig_rgb_final"],
            f"depth_{common_suffix}": ret["depth_final"],
            f"opacity_{common_suffix}": ret["weights"].sum(1),
        }
        if "offset_rgb_final" in ret:
            result[f"offset_rgb_{common_suffix}"] = ret["offset_rgb_final"]
        if common_suffix == "fine" and "feature_field" in ret:
            result["feature_field"] = ret["feature_field"]
        if common_suffix == "fine" and "offset_rgb" in ret:
            if (
                "b_apply_weight" in dummy_kwargs
                and dummy_kwargs["b_apply_weight"] == True
            ):
                result["offset_rgb"] = ret["weights"].unsqueeze(-1) * ret["offset_rgb"]
            else:
                result["offset_rgb"] = ret["offset_rgb"]
        return result

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
        2
    )  # (N_rays, N_samples, 3)

    coarse_models = prepare_models_dict(models, "coarse")

    if test_time:
        coarse_ret = inference(
            coarse_models,
            embeddings,
            xyz_coarse_sampled,
            rays_d,
            z_vals,
            weights_only=True,
        )
        result = {"opacity_coarse": coarse_ret["weights"].sum(1)}
    else:
        coarse_ret = inference(
            coarse_models,
            embeddings,
            xyz_coarse_sampled,
            rays_d,
            z_vals,
            weights_only=False,
        )
        result = collect_results(coarse_ret, "coarse")

    weights_coarse = coarse_ret["weights"]

    if N_importance > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(
            z_vals_mid, weights_coarse[:, 1:-1], N_importance, det=(perturb == 0)
        ).detach()
        # detach so that grad doesn't propogate to weights_coarse from here
        # pdf : [w1, w2, w3](dicard w0, w4), -> cdf [0, w1, w1+w2, w1+w2+w3],
        # interpolation: If a \in [w1, w1+w2], then return interpolation value between [z_mid2, z_mid3].
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
            2
        )
        # (N_rays, N_samples+N_importance, 3)
        fine_models = prepare_models_dict(models, "fine")

        fine_ret = inference(
            fine_models,
            embeddings,
            xyz_fine_sampled,
            rays_d,
            z_vals,
            weights_only=False,
        )
        fine_result = collect_results(fine_ret, "fine")
        result.update(fine_result)

    return result
