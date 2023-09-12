import os
import trainlib
from model import make_model, loss
import util
import numpy as np
import torch


class YoloTrainer(trainlib.Trainer):
    def __init__(self, args, conf, dset, val_dset, net, renderer, render_par, nviews, device):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)

        self.renderer = renderer
        self.net = net
        self.dset = dset
        self.val_dset = val_dset
        self.device = device
        self.nviews = nviews
        self.render_par = render_par

        self.renderer_state_path = "%s/%s/_renderer" % (
            args.checkpoints_path,
            args.name,
        )

        self.yolo_loss = loss.YoloLoss().to(device=device)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                self.renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.cell_width = conf["yolo.cell_width"]
        self.cell_height = conf["yolo.cell_height"]

    def extra_save_state(self):
        torch.save(self.renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True):
        assert "images" in data

        all_images = data["images"].to(device=self.device)
        all_poses = data["poses"].to(device=self.device)
        all_objects = data["objects"].to(device=self.device)
        focal = torch.tensor([data["focal"], data["focal"]], device=self.device)
        c = torch.tensor([data["c"], data["c"]], device=self.device)

        SB, NV, _, H, W = all_images.shape

        # scale the height and width of the images by the cell height and width
        H = H // self.cell_height
        W = W // self.cell_width
        # scale the focal and c by the cell height and width
        focal = focal / [self.cell_width, self.cell_height]
        c = c / [self.cell_width, self.cell_height]

        # TODO: get the number of views from a different array

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            if curr_nviews > 1:
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )

            # TODO: get the ground truth bounding boxes from the dataset

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)

    def train_step(self, data):
        print("train_step")

    def eval_step(self, data):
        print("eval_step")

    def vis_step(self, data):
        print("vis_step")