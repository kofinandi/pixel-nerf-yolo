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

    return parser


if __name__ == '__main__':
    args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
    device = util.get_cuda(args.gpu_id[0])

    dset, val_dset, test_dset = get_split_dataset(args.dataset_format, args.datadir, conf=conf)
    print(
        "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far,
                                                      dset.lindisp if hasattr(dset, "lindisp") else "N/A")
    )

    early_restart = conf.get_bool("yolo.early_restart", False)

    net = make_model(conf["model"]).to(device=device)

    print("Number of model parameters:", util.count_parameters(net))

    renderer = render_util.make_renderer(conf, lindisp=dset.lindisp if hasattr(dset, "lindisp") else None, ).to(
        device=device)

    render_par = renderer.bind_parallel(net, args.gpu_id)

    nviews = list(map(int, args.nviews.split()))

    trainer = train_util.make_trainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device)

    print("\n------------ Generating images ------------")

    test_data_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    data = next(iter(test_data_loader))
    source = np.array(args.source.split(), dtype='int')
    dest = args.dest

    while True:
        nmst = float(input("Enter nmst: "))
        nmsiou = float(input("Enter nmsiou: "))

        trainer.nms_threshold = nmst
        trainer.nms_iou_threshold = nmsiou

        vis, _ = trainer.vis_step(data, idx=0, srcs=source, dest=dest)

        os.makedirs(os.path.join(args.visual_path, "yolo_vis"), exist_ok=True)

        vis_u8 = (vis * 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(
                args.visual_path,
                "yolo_vis",
                "{:04}_{:04}_vis_{}_{}.png".format(args.subset, dest, nmsiou, nmst),
            ),
            vis_u8,
        )
