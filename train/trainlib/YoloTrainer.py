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

        self.num_scales = conf["yolo.num_scales"]
        self.num_anchors_per_scale = conf["yolo.num_anchors_per_scale"]
        self.cell_sizes = conf["yolo.cell_sizes"][:self.num_scales]
        self.anchors = conf["yolo.anchors"][:self.num_scales]

        self.ray_batch_size = conf["yolo.ray_batch_size"]

    def extra_save_state(self):
        torch.save(self.renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True):
        assert "images" in data

        all_images = data["images"].to(device=self.device)  # (SB, NV, 3, H, W)
        all_poses = data["poses"].to(device=self.device)  # (SB, NV, 4, 4)
        all_bboxes = data["bboxes"].to(device=self.device)  # (SB, num_scales, NV, H_scaled, W_scaled, anchors, 7)
        all_focals = data["focal"].to(device=self.device)  # (SB, 2)
        all_c = data["c"].to(device=self.device)  # (SB, 2)

        # TODO: get the number of views from a different array
        SB, NV, _, H, W = all_images.shape

        all_bboxes_gt = []
        all_rays = []

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)

        # loop through the objects in the batch
        for scene_idx in range(SB):
            poses = all_poses[scene_idx]  # (NV, 4, 4)
            bboxes = all_bboxes[scene_idx]  # (NV, H_scaled, W_scaled, anchors, 7)
            focal = all_focals[scene_idx]  # (2)
            c = all_c[scene_idx]  # (2)

            # randomly select the views to use (encode) for this object
            if curr_nviews > 1:
                image_ord[scene_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )

            for scale_idx in range(self.num_scales):
                # scale the height and width of the images by cell size
                H_scaled = H // self.cell_sizes[scale_idx]
                W_scaled = W // self.cell_sizes[scale_idx]
                # scale the focal and c by the cell size
                focal_scaled = focal / self.cell_sizes[scale_idx]
                c_scaled = c / self.cell_sizes[scale_idx]

                # generate all the rays for all the views
                cam_rays = util.gen_rays(
                    poses, W_scaled, H_scaled, focal_scaled, self.z_near, self.z_far, c=c_scaled
                )  # (NV, H_scaled, W_scaled, 8)

                assert cam_rays.shape == (NV, H_scaled, W_scaled, 8)

                # reshape the rays
                cam_rays = cam_rays.reshape(-1, 8)  # (NV*H_scaled*W_scaled, 8)

                # reshape the bbox ground truth
                bbox_gt_all = bboxes.reshape(-1, self.num_anchors_per_scale, 7)  # (NV*H_scaled*W_scaled, num_anchors_per_scale, 7)

                # select random rays to render
                pix_inds = torch.randint(0, NV * H_scaled * W_scaled, (self.ray_batch_size,))

                rays = cam_rays[pix_inds]  # (ray_batch_size, 8)
                bbox_gt = bbox_gt_all[pix_inds]  # (ray_batch_size, num_anchors_per_scale, 7)

                # append the rays and bbox ground truth to the list
                all_rays.append(rays)
                all_bboxes_gt.append(bbox_gt)

        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)
        all_bboxes_gt = torch.stack(all_bboxes_gt)  # (SB, ray_batch_size, num_anchors_per_scale, 7)

        image_ord = image_ord.to(self.device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        self.net.encode(
            src_images,
            src_poses,
            all_focals,
            c=all_c,
        )

        render = self.render_par(all_rays)

        loss = self.yolo_loss(render, all_bboxes_gt, self.anchors)

        if is_train:
            loss.backward()

        return loss.item()

    def train_step(self, data):
        return self.calc_losses(data, is_train=True)

    def eval_step(self, data):
        self.renderer.eval()
        losses = self.calc_losses(data, is_train=False)
        self.renderer.train()
        return losses

    def vis_step(self, data):
        print("vis_step")
