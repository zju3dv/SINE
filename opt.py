import argparse
from utils.io_util import ForceKeyErrorDict
from omegaconf import OmegaConf


def get_opts_args(do_parse_args=True):

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego",
        help="root directory of dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="blender",
        choices=["blender", "llff"],
        help="which dataset to train/val",
    )
    parser.add_argument(
        "--img_wh",
        nargs="+",
        type=int,
        default=[800, 800],
        help="resolution (img_w, img_h) of the image",
    )
    parser.add_argument(
        "--spheric_poses",
        default=False,
        action="store_true",
        help="whether images are taken in spheric poses (for llff)",
    )

    parser.add_argument("--factor", type=float, default=-1.0, help="scale pose")
    parser.add_argument("--bound_clamp", nargs="+", type=float, default=[])
    parser.add_argument("--pose_avg_path", type=str, default=None)
    parser.add_argument("--black_back", action="store_true", default=False)

    # rendering
    parser.add_argument(
        "--chunk",
        type=int,
        default=32 * 1024,
        help="chunk size to split the input to avoid OOM",
    )
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=128,
        help="number of additional fine samples",
    )
    parser.add_argument(
        "--use_disp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="factor to perturb depth sampling points",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="std dev of noise added to regularize sigma",
    )
    parser.add_argument(
        "--editing_attribute",
        type=float,
        default=1.0,
        help="a indicator to determine the extent of editing",
    )

    # models
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=[],
        nargs="+",
        help="pretrained checkpoint path to load",
    )
    parser.add_argument(
        "--prefixes_to_ignore",
        nargs="+",
        type=str,
        default=["loss"],
        help="the prefixes to ignore in the checkpoint state dict",
    )
    parser.add_argument("--model_type", default=[], nargs="+", type=str)
    parser.add_argument("--tcnn_bound", type=int, default=16)
    parser.add_argument("--depth_deformfield", type=int, default=2)
    parser.add_argument("--width_deformfield", type=int, default=128)
    parser.add_argument("--dim_deformxyz_emb", type=int, default=4)

    # train/eval
    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")

    if do_parse_args == True:
        return parser.parse_args()
    else:
        return parser


def get_arg_parser():
    parser = get_opts_args(False)
    parser.add_argument("--config", type=str, required=True)
    return parser


def load_config(parser, parser_unkonwn_args=False):
    if parser_unkonwn_args == True:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    tot_dict = vars(args)
    tot_dict.pop("config")
    tot_dict.update(conf)
    configs = ForceKeyErrorDict(**tot_dict)
    return configs
