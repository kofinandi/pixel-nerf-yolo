# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

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
    return parser


if __name__ == '__main__':
    args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
    device = util.get_cuda(args.gpu_id[0])

    dset, val_dset, test_dset = get_split_dataset(args.dataset_format, args.datadir, conf=conf)
    print(
        "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp if hasattr(dset, "lindisp") else "N/A")
    )

    early_restart = conf.get_bool("yolo.early_restart", False)

    net = make_model(conf["model"]).to(device=device)

    print("Number of model parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    renderer = render_util.make_renderer(conf, lindisp=dset.lindisp if hasattr(dset, "lindisp") else None,).to(device=device)

    render_par = renderer.bind_parallel(net, args.gpu_id)

    nviews = list(map(int, args.nviews.split()))

    trainer = train_util.make_trainer(args, conf, dset, val_dset, net, renderer, render_par, nviews, device)

    print("\n------------ Eval ------------")

    test_data_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    precision, recall, f1 = trainer.metric_step(test_data_loader, print_hc=True)

    print("Precision\tRecall\tF1")
    print("{}\t{}\t{}".format(precision, recall, f1))
