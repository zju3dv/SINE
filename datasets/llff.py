import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms as T
from utils.pose_util import center_poses

from .ray_utils import get_ndc_rays, get_ray_directions, get_rays


class LLFFDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(504, 378),
        spheric_poses=False,
        factor=-1,
        bound_clamp=[],
        pose_avg_path=None,
        white_back=True,
        llff_pose_avg=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.factor = factor
        self.bound_clamp = bound_clamp
        self.white_back = white_back
        self.llff_pose_avg = llff_pose_avg
        self.pose_avg_path = pose_avg_path
        self.define_transforms()

        self.read_meta()

    def collect_rays(self, directions, c2w):
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if not self.spheric_poses:
            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(
                self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d
            )
            # near plane is always at 1.0
            # near and far in NDC are always 0 and 1
            # See https://github.com/bmild/nerf/issues/34
        else:
            near = self.bounds.min()
            # focus on central object only
            far = min(8 * near, self.bounds.max())

        return torch.cat(
            [
                rays_o,
                rays_d,
                near * torch.ones_like(rays_o[:, :1]),
                far * torch.ones_like(rays_o[:, :1]),
            ],
            1,
        )

    def read_meta(self):
        poses_bounds = np.load(
            os.path.join(self.root_dir, "poses_bounds.npy")
        )  # (N_images, 17)

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)
        if len(self.bound_clamp) >= 2:
            self.bounds[:, 0] = self.bound_clamp[0]
            self.bounds[:, 1] = self.bound_clamp[1]

        # Step 1: rescale focal length according to training resolution
        # original intrinsics, same for all images
        H, W, self.focal = poses[0, :, -1]

        self.focal *= self.img_wh[0] / W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        if self.llff_pose_avg:
            if self.pose_avg_path is not None:
                pose_avg_homo_inv = np.load(self.pose_avg_path)
                self.poses, self.pose_avg = center_poses(poses, pose_avg_homo_inv)
            else:
                self.poses, self.pose_avg = center_poses(poses)
                np.save(os.path.join(self.root_dir, "pose_avg.npy"), self.pose_avg)
        else:
            self.poses = poses

        # choose val image as the closest to
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        if self.factor <= 0:
            scale_factor = near_original * 0.75  # 0.75 is the default parameter
            # the nearest depth is at 1/0.75=1.33
        else:
            scale_factor = self.factor
        print(f"[Info] scale_factor: {scale_factor} ")
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor
        print(f"[Info] max pos abs value: {np.max(np.abs(self.poses[...,3]))} ")
        print(f"[Info] max bound abs value: {np.max(np.abs(self.bounds))} ")
        print(f"[Info] min pos abs value: {np.min(np.abs(self.poses[...,3]))} ")
        print(f"[Info] min bound abs value: {np.min(np.abs(self.bounds))} ")

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            self.img_wh[1], self.img_wh[0], self.focal
        )  # (H, W, 3)

        if self.split == "train":  # create buffer of all rays and rgb data
            raise NotImplementedError
        elif self.split == "val":
            raise NotImplementedError
        else:  # for testing, create a parametric rendering path
            if self.split.endswith("train"):  # test on training set
                self.poses_test = self.poses

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            raise NotImplementedError
        elif self.split == "val":
            raise NotImplementedError
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            raise NotImplementedError
        elif self.split == "val":
            raise NotImplementedError
        else:
            c2w = torch.FloatTensor(self.poses_test[idx])
            rays = self.collect_rays(self.directions, c2w)

            sample = {"rays": rays, "c2w": c2w}

        return sample
