from .llff import LLFFDataset

dataset_dict = {"llff": LLFFDataset}


def construct_dataset(args, split):
    dataset = dataset_dict[args.dataset_name]
    kwargs = {
        "root_dir": args.root_dir,
        "img_wh": tuple(args.img_wh),
        "factor": args.factor,
        "bound_clamp": args.bound_clamp,
        "pose_avg_path": args.pose_avg_path,
        "white_back": not args.black_back,
        "llff_pose_avg": args.get("llff_pose_avg", True),
    }
    if "llff" in args.dataset_name:
        kwargs["spheric_poses"] = args.spheric_poses
    train_dataset = dataset(split=split, **kwargs)
    return train_dataset
