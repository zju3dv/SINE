import torch
import cv2
import numpy as np
from kornia import create_meshgrid
from math import sqrt, exp


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_z(th):
    return torch.Tensor(
        [
            [np.cos(th), -np.sin(th), 0, 0],
            [np.sin(th), np.cos(th), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rotate_3d(c2w, x, y, z):
    rot = (
        rot_phi(x / 180.0 * np.pi)
        @ rot_theta(y / 180.0 * np.pi)
        @ rot_z(z / 180.0 * np.pi)
    )
    return rot @ c2w

def get_select_inds(sample_wh, iterations, in_scale=-1, random_shift=True):
    w, h = torch.meshgrid(
        [
            torch.linspace(-1, 1, sample_wh[1]),
            torch.linspace(-1, 1, sample_wh[0]),
        ],
        indexing="ij",
    )
    h = h.unsqueeze(2)
    w = w.unsqueeze(2)

    scale = torch.as_tensor(in_scale)
    if scale <= 0:
        scale_anneal = 0.0025
        min_scale = 0.25
        max_scale = 1.0
        if scale_anneal > 0:
            k_iter = iterations // 1000 * 3
            min_scale = max(min_scale, max_scale * exp(-k_iter * scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = 0.25
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
    h = h * scale
    w = w * scale

    if random_shift:
        max_offset = 1 - scale.item()
        h_offset = (
            torch.Tensor(1).uniform_(0, max_offset)
            * (torch.randint(2, (1,)).float() - 0.5)
            * 2
        )
        w_offset = (
            torch.Tensor(1).uniform_(0, max_offset)
            * (torch.randint(2, (1,)).float() - 0.5)
            * 2
        )

        h += h_offset
        w += w_offset

    return torch.cat([h, w], dim=2), scale


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1
    )  # (H, W, 3), opencv model
    return torch.Tensor(directions)


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * ox_oz
    o1 = -1.0 / (H / (2.0 * focal)) * oy_oz
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def project_ij_to_direction(ij, H, W, focal):
    directions = torch.stack(
        [
            (ij[..., 1] - W / 2) / focal,
            -(ij[..., 0] - H / 2) / focal,
            -torch.ones_like(ij[..., 0:1]),
        ],
        -1,
    )  # (H, W, 3), opengl model
    return torch.Tensor(directions)


def project_uv_to_direction(uv, H, W, focal):
    directions = torch.stack(
        [
            (uv[..., 0] - W / 2) / focal,
            -(uv[..., 1] - H / 2) / focal,
            -torch.ones_like(uv[..., 0]),
        ],
        -1,
    )  # (H, W, 3), opengl model
    return directions.float()
