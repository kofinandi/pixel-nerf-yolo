import os
import trainlib
from model import make_model, loss
import util
import numpy as np
import torch
import matplotlib.pyplot as plt


class YOLOTrainer(trainlib.Trainer):
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


        if args.resume:
            if os.path.exists(self.renderer_state_path):
                self.renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.num_scales = conf["model.mlp_coarse.num_scales"]
        self.num_anchors_per_scale = conf["model.mlp_coarse.num_anchors_per_scale"]
        self.cell_sizes = conf["yolo.cell_sizes"][:self.num_scales]
        self.anchors = torch.Tensor(conf["yolo.anchors"][:self.num_scales]).to(device=device)

        self.ray_batch_size = conf["yolo.ray_batch_size"]

        self.yolo_loss = loss.YoloLoss.from_conf(conf, self.num_anchors_per_scale).to(device=device)

    def extra_save_state(self):
        torch.save(self.renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True):
        assert "images" in data

        all_images = data["images"].to(device=self.device)  # (SB, NV, 3, H, W)
        all_poses = data["poses"].to(device=self.device)  # (SB, NV, 4, 4)
        all_bboxes = data["bboxes"]  # NV long list, num_scales long tuple, (1, anchors_per_scale, H_scaled, W_scaled, 6)
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
            bboxes = all_bboxes  # TODO: with batch size > 1 get the bboxes for this scene
            focal = all_focals[scene_idx]  # (2)
            c = all_c[scene_idx]  # (2)

            # randomly select the views to use (encode) for this object
            if curr_nviews > 1:
                image_ord[scene_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )

            for scale_idx in range(self.num_scales):
                # convert to a tensor of size (NV, 1, anchors_per_scale, H_scaled, W_scaled, 6)
                # from a list of length NV of tuples of length num_scales of tensors of size (1, anchors_per_scale, H_scaled, W_scaled, 6)
                # and get only the bboxes for this scale
                bboxes_at_scale = []
                for i in range(len(bboxes)):
                    bboxes_at_scale.append(bboxes[i][scale_idx].to(device=self.device))

                bboxes_at_scale = torch.stack(bboxes_at_scale)  # (NV, 1, anchors_per_scale, H_scaled, W_scaled, 6)

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

                # TODO: check if this H_scaled, W_scaled is the same as above
                # reshape the bbox ground truth
                bbox_gt_all = bboxes_at_scale.reshape(-1, self.num_anchors_per_scale, 6)  # (NV*H_scaled*W_scaled, num_anchors_per_scale, 6)

                # select random rays to render
                pix_inds = torch.randint(0, NV * H_scaled * W_scaled, (self.ray_batch_size,))

                rays = cam_rays[pix_inds]  # (ray_batch_size, 8)
                bbox_gt = bbox_gt_all[pix_inds]  # (ray_batch_size, num_anchors_per_scale, 6)

                # append the rays and bbox ground truth to the list
                all_rays.append(rays)
                all_bboxes_gt.append(bbox_gt)

        all_rays = torch.stack(all_rays)  # (SB * num_scales, ray_batch_size, 8)
        all_bboxes_gt = torch.stack(all_bboxes_gt)  # (SB * num_scales, ray_batch_size, num_anchors_per_scale, 6)

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

        # TODO: check sizes
        # TODO: do a loop here to do bigger batches in splits
        render = self.render_par(all_rays)  # (SB * num_scales * ray_batch_size, num_anchors_per_scale, 8)

        # reshape the render to be (SB * num_scales, ray_batch_size, num_anchors_per_scale, 7)
        render = render.reshape(SB * self.num_scales, self.ray_batch_size, self.num_anchors_per_scale, 7)

        loss, box_loss, object_loss, no_object_loss, class_loss = self.yolo_loss(render, all_bboxes_gt, self.anchors)

        if is_train:
            loss.backward()

        loss_dict = {"t": loss.item(), "box_loss": box_loss.item(), "object_loss": object_loss.item(),
                     "no_object_loss": no_object_loss.item(), "class_loss": class_loss.item()}

        return loss_dict

    def train_step(self, data, global_step=None):
        return self.calc_losses(data, is_train=True)

    def eval_step(self, data, global_step=None):
        self.renderer.eval()
        losses = self.calc_losses(data, is_train=False)
        self.renderer.train()
        return losses

    def vis_step(self, data, global_step=None, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx

        all_images = data["images"][batch_idx].to(device=self.device)  # (NV, 3, H, W)
        all_poses = data["poses"][batch_idx].to(device=self.device)  # (NV, 4, 4)
        all_bboxes = data["bboxes"]  # NV long list, num_scales long tuple, (1, anchors_per_scale, H_scaled, W_scaled, 6)
        focal = data["focal"][batch_idx: batch_idx + 1].to(device=self.device)  # (2)
        c = data["c"][batch_idx: batch_idx + 1].to(device=self.device)  # (2)

        NV, _, H, W = all_images.shape

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.choice(views_src)
        views_src = torch.from_numpy(views_src)

        H_scaled = H // self.cell_sizes[0]
        W_scaled = W // self.cell_sizes[0]
        # scale the focal and c by the cell size
        focal_scaled = focal / self.cell_sizes[0]
        c_scaled = c / self.cell_sizes[0]

        cam_rays = util.gen_rays(
            all_poses, W_scaled, H_scaled, focal_scaled, self.z_near, self.z_far, c=c_scaled
        )  # (NV, H, W, 8)

        self.renderer.eval()

        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H_scaled, W_scaled, 8)
            test_images = all_images[views_src]  # (NS, 3, H, W)
            self.net.encode(
                test_images.unsqueeze(0),
                all_poses[views_src].unsqueeze(0),
                focal.to(device=self.device),
                c=c.to(device=self.device),
            )

            test_rays = test_rays.reshape(1, H_scaled * W_scaled, -1)  # (1, H_scaled*W_scaled, 8)
            render = self.render_par(test_rays)  # (H_scaled*W_scaled, num_anchors_per_scale, 7)

            # reshape the render to be (1, num_anchors_per_scale, H_scaled, W_scaled, 7)
            render = render.reshape(1, self.num_anchors_per_scale, H_scaled, W_scaled, 7)

        dest_img = all_images[view_dest].permute(1, 2, 0).to("cpu")
        dest_img = dest_img * 0.5 + 0.5

        boxes_gt = util.convert_cells_to_bboxes(all_bboxes[view_dest][0], self.anchors, H_scaled, W_scaled, is_predictions=False)[0]
        boxes_gt = util.nms(boxes_gt, 1, 0.8)
        boxes_gt_visual = util.draw_bounding_boxes(dest_img, boxes_gt)

        boxes_predicted = util.convert_cells_to_bboxes(render, self.anchors, H_scaled, W_scaled, is_predictions=True)[0]
        boxes_predicted = util.nms(boxes_predicted, 1, 0.8)

        if len(boxes_predicted) == 0:
            print("no boxes predicted")
            return None, None

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

        self.renderer.train()

        return vis, None
