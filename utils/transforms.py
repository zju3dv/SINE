import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch


class Global_crops(nn.Module):
    def __init__(self, n_crops, min_cover, last_transform, flip=False):
        super().__init__()
        self.n_crops = n_crops
        self.min_cover = min_cover

        transforms_lst = [last_transform]
        if flip:
            transforms_lst += [transforms.RandomHorizontalFlip()]

        self.last_transform = transforms.Compose(transforms_lst)

    def forward(self, imgs):
        """
        imgs: [N, 3, h, w]
        """
        crops = []
        if not torch.is_tensor(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            tensor_ims = []
            for img in imgs:
                tesnor_im = transforms.ToTensor()(img)
                tensor_ims.append(tesnor_im)
            imgs = torch.stack(tensor_ims, dim=0)
        h = imgs.shape[2]
        w = imgs.shape[-1]
        size = int(round(np.random.uniform(self.min_cover * h, h)))
        t = transforms.Compose([transforms.RandomCrop(min(size, w))])
        for _ in range(self.n_crops):
            crop = t(imgs)
            crops.append(crop)
        return torch.stack(crops)


dino_structure_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.5,
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    ]
)

dino_texture_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])


if __name__ == "__main__":
    import os
    from PIL import Image
    import torchvision.transforms as T

    # input_path1 = "/home/baochong/Projects/Nerf/Archive/nerf_pl/logs/toyota-eps_goodpose_trainsplit_nerftcnn_texturegrey_deferredbatchbugfullresolutionwithmask_severalviews_tensorarfbp_1011/version_0/vit/0000000270_A.png"
    # input_path2 = "/home/baochong/Projects/Nerf/Archive/nerf_pl/logs/toyota-eps_goodpose_trainsplit_nerftcnn_texturegrey_deferredbatchbugfullresolutionwithmask_severalviews_tensorarfbp_1011/version_0/vit/0000000270_x.png"
    # image1 = Image.open(input_path1)
    # image2 = Image.open(input_path2)
    # image1 = T.ToTensor()(image1)
    # image2 = T.ToTensor()(image2)
    input_path1 = "/mnt/nas_8/group/baochong/Projects/SemanticNerfEdit/Archive/nerf_pl/logs/original_img_0.pt"
    input_path2 = "/mnt/nas_8/group/baochong/Projects/SemanticNerfEdit/Archive/nerf_pl/logs/fine_rgb_0.pt"
    image1 = torch.load(input_path1).squeeze(0)
    image2 = torch.clamp(torch.load(input_path2).squeeze(0), 0, 1)
    aug = torch.stack([image1, image2], dim=0)
    output_path = "/home/baochong/Projects/Nerf/Archive/nerf_pl/logs/toyota-eps_goodpose_trainsplit_nerftcnn_texturegrey_deferredbatchbugfullresolutionwithmask_severalviews_tensorarfbp_1011/version_0/vit/debug"
    os.makedirs(output_path, exist_ok=True)
    for i in range(50):
        global_A_patches = transforms.Compose(
            [
                dino_structure_transforms,
                Global_crops(
                    n_crops=1,
                    min_cover=0.95,
                    last_transform=transforms.ToTensor(),
                ),
            ]
        )
        aug_images = global_A_patches(aug)
        aug_image1 = aug_images[0, 0]

        aug_image2 = aug_images[0, 1]
        T.ToPILImage()(aug_image1).save(os.path.join(output_path, f"{i}_image1.png"))
        T.ToPILImage()(aug_image2).save(os.path.join(output_path, f"{i}_image2.png"))
