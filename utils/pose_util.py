import os
import numpy as np
import torch


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def read_poses_bound(path, return_focal=False):
    poses_bounds = np.load(path)  # (N_images, 17)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    poses_gl = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    if return_focal == False:
        return poses_gl, bounds
    else:
        H, W, focal = poses[0, :, -1]
        return poses_gl, bounds, H, W, focal


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, pose_avg_homo_inv=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    if pose_avg_homo_inv is None:
        pose_avg = average_poses(poses)  # (3, 4)
        pose_avg_homo = np.eye(4)
        # convert to homogeneous coordinate for faster computation
        pose_avg_homo[:3] = pose_avg
        pose_avg_homo_inv = np.linalg.inv(pose_avg_homo)

    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_homo_inv @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo_inv


def create_nerf_pose(path, pose_avg=None, scale_factor=-1, return_focal=False):
    out = read_poses_bound(path, return_focal)
    poses = out[0]
    poses, _ = center_poses(poses, pose_avg)
    poses[..., 3] /= scale_factor
    return (poses, *out[1:])


def save_poses_bound(path, poses, bounds, H, W, focal, b_gl_poses=True):
    # poses: opencv poses, (N, 3, 4)
    # bounds: near, far, (N, 2)
    num_poses = len(poses)
    if b_gl_poses == True:
        # poses @ gl2cv
        poses[:, :, 1] *= -1
        poses[:, :, 2] *= -1

    poses = poses[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate(
        [
            poses,
            np.tile(np.array([H, W, focal]).reshape(3, 1, 1), [1, 1, poses.shape[-1]]),
        ],
        1,
    )  # (3, 5, N_poses)
    poses = np.concatenate(
        [
            poses[:, 1:2, :],
            poses[:, 0:1, :],
            -poses[:, 2:3, :],
            poses[:, 3:4, :],
            poses[:, 4:5, :],
        ],
        1,
    )

    save_arr = []
    for i in range(num_poses):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds[i]], 0))
    save_arr = np.array(save_arr)

    np.save(path, save_arr)
    pass


def project_world_onto_img(pts, w2c, H, W, focal, return_depth=False):
    """
    pts : (N, 3)
    w2c: (3, 4) from world to opencv camera
    """
    pts_cam = (torch.mm(w2c[:3, :3], pts.permute(1, 0)) + w2c[:3, 3:4]).permute(
        1, 0
    )  # (N, 3)
    pts_img = pts_cam / (pts_cam[..., 2:3] + 1e-9)
    pts_img[..., 0] = pts_img[..., 0] * focal + W / 2
    pts_img[..., 1] = pts_img[..., 1] * focal + H / 2
    if return_depth == False:
        return pts_img[..., :2]
    else:
        return (pts_img[..., :2], pts_cam[..., 2])


def view_matrix(forward: np.ndarray, up: np.ndarray, cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def look_at(
    cam_location: np.ndarray,
    point: np.ndarray,
    up=np.array([0.0, -1.0, 0.0])  # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)  # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)


def c2w_track_spiral(
    c2w,
    up_vec,
    rads,
    focus: float,
    zrate: float,
    rots: int,
    N: int,
    zdelta: float = 0.0,
):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]

    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.0])

    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.0])  # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4],
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.0])
            * rads,  # openCV convention
        )
        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks
