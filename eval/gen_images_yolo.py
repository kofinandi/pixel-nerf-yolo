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

    all_images = data["images"][0].to(device=device)  # (NV, 3, H, W)
    all_poses = data["poses"][0].to(device=device)  # (NV, 4, 4)
    all_bboxes = data["bboxes"]  # NV long list, num_scales long tuple, (1, anchors_per_scale, H_scaled, W_scaled, 6)
    focal = data["focal"][0:1].to(device=device)  # (2)
    c = data["c"][0:1].to(device=device)  # (2)

    anchors = torch.Tensor(conf["yolo.anchors"][:1]).to(device=device)

    NV, _, H, W = all_images.shape

    views_src = source
    view_dest = dest
    views_src = torch.from_numpy(views_src)

    H_scaled = H // conf["yolo.cell_sizes"][0]
    W_scaled = W // conf["yolo.cell_sizes"][0]
    # scale the focal and c by the cell size
    focal_scaled = focal / conf["yolo.cell_sizes"][0]
    c_scaled = c / conf["yolo.cell_sizes"][0]

    cam_rays = util.gen_rays(
        all_poses, W_scaled, H_scaled, focal_scaled, 0.1, 30, c=c_scaled
    )  # (NV, H, W, 8)

    renderer.eval()

    with torch.no_grad():
        test_rays = cam_rays[view_dest]  # (H_scaled, W_scaled, 8)
        test_images = all_images[views_src]  # (NS, 3, H, W)
        net.encode(
            test_images.unsqueeze(0),
            all_poses[views_src].unsqueeze(0),
            focal.to(device=device),
            c=c.to(device=device),
        )

        test_rays = test_rays.reshape(1, H_scaled * W_scaled, -1)  # (1, H_scaled*W_scaled, 8)
        render = render_par(test_rays)  # (H_scaled*W_scaled, num_anchors_per_scale, 7)

        # reshape the render to be (1, num_anchors_per_scale, H_scaled, W_scaled, 7)
        render = render.reshape(1, 3, H_scaled, W_scaled, 7)

    dest_img = all_images[view_dest].permute(1, 2, 0).to("cpu")
    dest_img = dest_img * 0.5 + 0.5

    while True:
        nmst = float(input("Enter nmst: "))
        nmsiou = float(input("Enter nmsiou: "))

        boxes_gt = \
            util.convert_cells_to_bboxes(all_bboxes[view_dest][0], anchors, H_scaled, W_scaled, is_predictions=False)[0]
        boxes_gt = util.nms(boxes_gt, nmsiou, nmst)
        boxes_gt_visual = util.draw_bounding_boxes(dest_img, boxes_gt)

        boxes_predicted = util.convert_cells_to_bboxes(render, anchors, H_scaled, W_scaled, is_predictions=True)[0]
        boxes_predicted = util.nms(boxes_predicted, nmsiou, nmst)

        print("boxes_predicted", len(boxes_predicted))

        boxes_predicted_visual = util.draw_bounding_boxes(dest_img, boxes_predicted)

        source_views = (
            (all_images[views_src] * 0.5 + 0.5)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        vis_list = [
            *source_views,
            dest_img.cpu().numpy(),
            boxes_gt_visual,
            boxes_predicted_visual,
        ]

        vis = np.hstack(vis_list)

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
