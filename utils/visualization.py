import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch


def visualize_depth(depth, mi=None, ma=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    is_tensor_type = torch.is_tensor(depth)
    if is_tensor_type:
        x = depth.cpu().numpy()
    else:
        x = depth
    x = np.nan_to_num(x)  # change nan to 0
    if mi is None:
        mi = np.min(x)  # get minimum depth
    if ma is None:
        ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    if is_tensor_type:
        x_ = T.ToTensor()(x_)  # (3, H, W)
    else:
        x_ = np.array(x_)
    return x_


def normalize_image(image):
    # image -= image.min()
    # image /= image.max()
    # return image
    min_value, max_value = image.min(), image.max()
    return (image - min_value) / (max_value - min_value)


import matplotlib.pyplot as plt


def map_to_color(x, cmap="viridis", vmin=None, vmax=None):
    if vmin == None:
        vmin = x.min()
    if vmax == None:
        vmax = x.max()
    data = np.clip(x, vmin, vmax)
    colors = plt.cm.get_cmap(cmap)((data - vmin) / (vmax - vmin))
    return colors


def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(palette)
    return np.array(new_mask.convert("RGB"))


def dump_proj_uvs(proj_uv, ref_uv, img_wh, save_path):
    def batch_paint_pixels(img, i, j, color):
        mask = ((i >= 0) & (i < img.shape[0])) & ((j >= 0) & (j < img.shape[1]))
        img[i[mask], j[mask]] = color

    def draw_img(img, uv, color):
        batch_paint_pixels(img, uv[..., 1], uv[..., 0], color)
        batch_paint_pixels(img, uv[..., 1] - 1, uv[..., 0] - 1, color)
        batch_paint_pixels(img, uv[..., 1] + 1, uv[..., 0] + 1, color)
        batch_paint_pixels(img, uv[..., 1] - 1, uv[..., 0] + 1, color)
        batch_paint_pixels(img, uv[..., 1] + 1, uv[..., 0] - 1, color)
        batch_paint_pixels(img, uv[..., 1], uv[..., 0] + 1, color)
        batch_paint_pixels(img, uv[..., 1], uv[..., 0] - 1, color)
        batch_paint_pixels(img, uv[..., 1] - 1, uv[..., 0], color)
        batch_paint_pixels(img, uv[..., 1] + 1, uv[..., 0], color)

    img = np.zeros((img_wh[1], img_wh[0], 4))
    proj_uv_tmp = proj_uv.int().detach().cpu().numpy()
    ref_uv_tmp = ref_uv.int().detach().cpu().numpy()
    for i in range(len(proj_uv_tmp)):
        color = np.random.rand(4) * 255
        color[-1] = 255
        draw_img(img, proj_uv_tmp[i], color.astype(np.uint8))
        color = color * 0.8
        draw_img(img, ref_uv_tmp[i], color.astype(np.uint8))
    cv2.imwrite(
        save_path,
        img,
    )


def draw_optical(out_img, mkpts0, mkpts1, skip_prob=0.0, colors=None):
    """
    mkpts0 : [x, y], x: width, y: height
    """
    for i in range(len(mkpts0)):
        if np.random.random() < skip_prob:
            continue
        # if colors is None:
        #     color = np.random.rand(3) * 255
        # else:
        #     color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        if out_img.shape[-1] == 4:
            color1 = (0, 0, 255, 255)
            color2 = (255, 0, 0, 255)
            color3 = (0, 255, 0, 255)
        else:
            color1 = (0, 0, 255)
            color2 = (255, 0, 0)
            color3 = (0, 255, 0)
        cv2.circle(out_img, (mkpts0[i][0], mkpts0[i][1]), 2, color1, 2)
        cv2.circle(out_img, (mkpts1[i][0], mkpts1[i][1]), 2, color2, 2)
        cv2.line(
            out_img,
            (mkpts0[i][0], mkpts0[i][1]),
            (mkpts1[i][0], mkpts1[i][1]),
            color3,
            2,
        )
    return out_img
