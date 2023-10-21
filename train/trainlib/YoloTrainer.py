import os
import trainlib
from model import make_model, loss
import util
import numpy as np
import torch


class YOLOTrainer(trainlib.Trainer):
    def __init__(self, args, conf, dset, val_dset, net, renderer, render_par, nviews, device):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)

        self.renderer = renderer
        self.net = net
        self.dset = dset
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

        self.early_restart = conf["yolo.early_restart"]

        self.nms_iou_threshold = conf["yolo.nms_iou_threshold"]
        self.nms_threshold = conf["yolo.nms_threshold"]

        self.metric_views = conf["yolo.metric_views"]
        self.match_iou_threshold = conf["yolo.match_iou_threshold"]

        print("n_coarse", conf["renderer.n_coarse"])
        print("nms_iou_threshold", self.nms_iou_threshold)
        print("nms_threshold", self.nms_threshold)
        print("match_iou_threshold", self.match_iou_threshold)

    def extra_save_state(self):
        torch.save(self.renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True):
        assert "images" in data

        all_images = data["images"]  # (SB, NV, 3, H, W)
        all_poses = data["poses"]  # (SB, NV, 4, 4)
        all_bboxes = data["bboxes"]  # NV long list, num_scales long tuple, (SB, H_scaled, W_scaled, anchors_per_scale, 6)
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

                bboxes_at_scale = torch.stack(bboxes_at_scale).squeeze(1)  # (NV, H_scaled, W_scaled, anchors_per_scale, 6)

                # scale the height and width of the images by cell size
                H_scaled = H // self.cell_sizes[scale_idx]
                W_scaled = W // self.cell_sizes[scale_idx]
                # scale the focal and c by the cell size
                focal_scaled = focal / self.cell_sizes[scale_idx]
                c_scaled = c / self.cell_sizes[scale_idx]

                target_poses = poses[image_ord[scene_idx]]  # (curr_nviews, 4, 4)

                # generate all the rays for all the views
                cam_rays = util.gen_rays_yolo(
                    target_poses, W_scaled, H_scaled, focal_scaled, c_scaled, self.z_near, self.z_far
                )  # (curr_nviews, H_scaled, W_scaled, 8)

                assert cam_rays.shape == (curr_nviews, H_scaled, W_scaled, 8)

                # reshape the rays
                cam_rays = cam_rays.reshape(-1, 8)  # (curr_nviews*H_scaled*W_scaled, 8)

                target_bbox = bboxes_at_scale[image_ord[scene_idx]]  # (curr_nviews, H_scaled, W_scaled, anchors_per_scale, 6)

                # reshape the bbox ground truth
                bbox_gt_all = target_bbox.reshape(-1, self.num_anchors_per_scale, 6)  # (curr_nviews*H_scaled*W_scaled, num_anchors_per_scale, 6)

                # append the rays and bbox ground truth to the list
                all_rays.append(cam_rays)
                all_bboxes_gt.append(bbox_gt_all)

        src_images = util.batched_index_select_nd(all_images, image_ord).to(self.device)  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord).to(self.device)  # (SB, NS, 4, 4)

        self.net.encode(
            src_images,
            src_poses,
            all_focals,
            c=all_c,
        )

        total_loss = 0
        total_box_loss = 0
        total_object_loss = 0
        total_no_object_loss = 0
        total_class_loss = 0

        scale = 0
        mini_batch = 0
        for rays_on_scale, bboxes_on_scale in zip(all_rays, all_bboxes_gt):
            rays_on_scale = rays_on_scale.unsqueeze(0)
            bboxes_on_scale = bboxes_on_scale.unsqueeze(0)

            # split the rays into batches
            rays_on_scale = torch.split(rays_on_scale, self.ray_batch_size, dim=1)  # (SB, ray_batch_size, 8)
            bboxes_on_scale = torch.split(bboxes_on_scale, self.ray_batch_size, dim=1)  # (SB, ray_batch_size, num_anchors_per_scale, 6)

            for rays, bboxes_gt in zip(rays_on_scale, bboxes_on_scale):
                mini_batch += 1
                current_ray_batch_size = rays.shape[1]
                render = self.render_par(rays.to(self.device))  # (SB * current_ray_batch_size, num_anchors_per_scale, 7)

                # print if any of the values are nan
                if torch.isnan(render).any():
                    print("render contains nan")
                    print(render)

                if torch.isnan(bboxes_gt).any():
                    print("all_bboxes_gt contains nan")
                    print(bboxes_gt)

                # print if any of the values are inf
                if torch.isinf(render).any():
                    print("render contains inf")
                    print(render)

                if torch.isinf(bboxes_gt).any():
                    print("all_bboxes_gt contains inf")
                    print(bboxes_gt)

                # reshape the render to be (SB, current_ray_batch_size, num_anchors_per_scale, 7)
                render = render.reshape(SB, current_ray_batch_size, self.num_anchors_per_scale, 7)

                loss, box_loss, object_loss, no_object_loss, class_loss = self.yolo_loss(render, bboxes_gt, self.anchors[scale])

                if is_train:
                    loss.backward(retain_graph=True)

                    # print if any of the gradients are nan
                    if any(torch.isnan(p.grad).any() if p.grad is not None else False for p in self.net.parameters()):
                        print("model gradients contain nan")

                    # print if any of the gradients are inf
                    if any(torch.isinf(p.grad).any() if p.grad is not None else False for p in self.net.parameters()):
                        print("model gradients contain inf")

                total_loss += loss.item()
                total_box_loss += box_loss.item()
                total_object_loss += object_loss.item()
                total_no_object_loss += no_object_loss.item()
                total_class_loss += class_loss.item()

            scale += 1

        total_loss /= mini_batch
        total_box_loss /= mini_batch
        total_object_loss /= mini_batch
        total_no_object_loss /= mini_batch
        total_class_loss /= mini_batch

        loss_dict = {"t": total_loss, "box_loss": total_box_loss, "object_loss": total_object_loss,
                     "no_object_loss": total_no_object_loss, "class_loss": total_class_loss}

        return loss_dict

    def train_step(self, data, global_step=None):
        return self.calc_losses(data, is_train=True)

    def eval_step(self, data, global_step=None):
        self.renderer.eval()
        losses = self.calc_losses(data, is_train=False)
        self.renderer.train()
        return losses

    def vis_step(self, data, global_step=None, idx=None, srcs=None, dest=None, only_bbox=False):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            batch_idx = idx

        all_images = data["images"][batch_idx]  # (NV, 3, H, W)
        all_poses = data["poses"][batch_idx]  # (NV, 4, 4)
        all_bboxes = data["bboxes"]  # NV long list, num_scales long tuple, (1, anchors_per_scale, H_scaled, W_scaled, 6)
        focal = data["focal"][batch_idx: batch_idx + 1]  # (2)
        c = data["c"][batch_idx: batch_idx + 1]  # (2)

        NV, _, H, W = all_images.shape

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False)) if srcs is None else srcs
        view_dest = np.random.choice(views_src) if dest is None else dest
        views_src = torch.from_numpy(views_src)

        self.renderer.eval()

        boxes_gt = []
        boxes_predicted = []

        with torch.no_grad():
            test_images = all_images[views_src]  # (NS, 3, H, W)
            self.net.encode(
                test_images.unsqueeze(0).to(device=self.device),
                all_poses[views_src].unsqueeze(0).to(device=self.device),
                focal.to(device=self.device),
                c=c.to(device=self.device),
            )

            for scale_idx in range(self.num_scales):
                # scale the height and width of the images by cell size
                H_scaled = H // self.cell_sizes[scale_idx]
                W_scaled = W // self.cell_sizes[scale_idx]
                # scale the focal and c by the cell size
                focal_scaled = focal[0] / self.cell_sizes[scale_idx]
                c_scaled = c[0] / self.cell_sizes[scale_idx]

                cam_rays = util.gen_rays_yolo(
                    all_poses, W_scaled, H_scaled, focal_scaled, c_scaled, self.z_near, self.z_far
                )  # (NV, H_scaled, W_scaled, 8)

                test_rays = cam_rays[view_dest]  # (H_scaled, W_scaled, 8)

                test_rays = test_rays.reshape(1, H_scaled * W_scaled, -1)  # (1, H_scaled*W_scaled, 8)

                test_rays = test_rays.split(self.ray_batch_size, dim=1)  # (1, ray_batch_size, 8)

                render = []
                for rays in test_rays:
                    render.append(self.render_par(rays.to(self.device)).to("cpu"))  # (H_scaled*W_scaled, num_anchors_per_scale, 7)

                render = torch.cat(render, dim=0)  # (H_scaled*W_scaled, num_anchors_per_scale, 7)

                # reshape the render to be (1, H_scaled, W_scaled, num_anchors_per_scale, 7)
                render = render.reshape(1, H_scaled, W_scaled, self.num_anchors_per_scale, 7)

                boxes_gt.append(util.convert_cells_to_bboxes(all_bboxes[view_dest][scale_idx], self.anchors[scale_idx].to("cpu"),
                                                             H_scaled, W_scaled, is_predictions=False)[0])
                boxes_predicted.append(util.convert_cells_to_bboxes(render, self.anchors[scale_idx].to("cpu"),
                                                                    H_scaled, W_scaled, is_predictions=True)[0])

        # flatten the list of bboxes
        boxes_gt = [item for sublist in boxes_gt for item in sublist]
        boxes_predicted = [item for sublist in boxes_predicted for item in sublist]

        if only_bbox:
            return boxes_gt, boxes_predicted

        boxes_gt, hc, bat = util.nms(boxes_gt, self.nms_iou_threshold, self.nms_threshold)
        print("highest confidence:", hc)
        print("bboxes above threshold", self.nms_threshold, ":", bat)

        boxes_predicted, hc, bat = util.nms(boxes_predicted, self.nms_iou_threshold, self.nms_threshold)
        print("highest confidence:", hc)
        print("bboxes above threshold", self.nms_threshold, ":", bat)
        print("boxes predicted:", len(boxes_predicted))

        if self.early_restart and len(boxes_predicted) == 0 and len(boxes_gt) > 0:
            print("no boxes predicted")
            return None, None

        dest_img = all_images[view_dest].permute(1, 2, 0).to("cpu")
        dest_img = dest_img * 0.5 + 0.5

        boxes_gt_visual = util.draw_bounding_boxes(dest_img, boxes_gt)
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

    def metric_step(self, data_loader, print_hc=False):
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for data in data_loader:
            for views in self.metric_views:
                views = np.array(views)
                for dest in views:
                    bbox_gt, bbox_pred = self.vis_step(data, idx=0, srcs=views, dest=dest, only_bbox=True)
                    tp, fp, fn = util.calculate_tp_fp_fn(bbox_gt, bbox_pred, self.nms_iou_threshold, self.nms_threshold, self.match_iou_threshold, print_hc=print_hc)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        print("total_tp", total_tp, "total_fp", total_fp, "total_fn", total_fn)
        return util.calculate_precision_recall_f1(total_tp, total_fp, total_fn)
