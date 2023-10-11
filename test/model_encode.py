import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import unittest
import src.model as model
import src.util as util
import src.data as data
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

class MyTestCase(unittest.TestCase):
    def test_encode_index(self):
        args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128, default_conf="../conf/exp/yolo.conf")
        args.dataset_format = "yolo"
        args.datadir = "../data/yolo"
        dset, val_dset, test_dset = data.get_split_dataset(args.dataset_format, args.datadir, conf=conf)
        net = model.make_model(conf["model"])

        test_data_loader = torch.utils.data.DataLoader(
            test_dset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

        all_images = test_data_loader.dataset[0]["images"]
        all_poses = test_data_loader.dataset[0]["poses"]
        all_focals = test_data_loader.dataset[0]["focal"]
        all_c = test_data_loader.dataset[0]["c"]

        test_image = all_images[[2, 5]].unsqueeze(0)
        test_pose = all_poses[[2, 5]].unsqueeze(0)
        test_focal = all_focals.unsqueeze(0)
        test_c = all_c.unsqueeze(0)

        net.encode(
            test_image,
            test_pose,
            test_focal,
            c=test_c,
        )

        test_point1 = torch.tensor([5.26, -0.83, -0.18]).unsqueeze(0).unsqueeze(0)
        test_viewdir = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0)
        net(test_point1, viewdirs=test_viewdir)


if __name__ == '__main__':
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    )

    unittest.main()
