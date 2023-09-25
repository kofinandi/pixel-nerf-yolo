import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import train_util
from render import render_util
from model import make_model, loss
from data import get_split_dataset
import util
import torch
import imageio
import numpy as np


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="0",
        help="Source view(s) in image, in increasing order.",
    )
    parser.add_argument(
        "--dest", type=int, default=0, help="Destination view to use"
    )
    parser.add_argument(
        "--nmst", type=float, default=None, help="NMS threshold to use"
    )
    parser.add_argument(
        "--nmsiou", type=float, default=None, help="NMS IoU threshold to use"
    )

    return parser


if __name__ == '__main__':
    args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
    device = util.get_cuda(args.gpu_id[0])

    # override conf with args for nms if specified
    if args.nmst is not None:
        conf["yolo.nms_threshold"] = args.nmst
    if args.nmsiou is not None:
        conf["yolo.nms_iou_threshold"] = args.nmsiou

    dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, conf=conf)
    print(
        "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp if hasattr(dset, "lindisp") else "N/A")
    )

    net = make_model(conf["model"]).to(device=device)
    net.load_weights(args)

    renderer = render_util.make_renderer(conf, lindisp=dset.lindisp if hasattr(dset, "lindisp") else None,).to(device=device)

    # Parallelize
    render_par = renderer.bind_parallel(net, args.gpu_id).eval()

    nviews = list(map(int, args.nviews.split()))

    trainer = train_util.make_trainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device)

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
    )

    data = next(iter(data_loader)) # TODO: get the right data (args.subset)
    source = np.array(list(map(int, args.source.split())), dtype=np.int32)
    dest = args.dest

    with torch.no_grad():
        vis, _ = trainer.vis_step(data, False, 0, source, dest)

    # create the directory if it does not exist
    os.makedirs(os.path.join(args.visual_path, "yolo_vis"), exist_ok=True)

    vis_u8 = (vis * 255).astype(np.uint8)
    imageio.imwrite(
        os.path.join(
            args.visual_path,
            "yolo_vis",
            "{:04}_{:04}_vis.png".format(args.subset, dest),
        ),
        vis_u8,
    )