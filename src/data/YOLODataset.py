import os
import torch
import util
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class YOLODataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            stage="train",
            z_near=1.2,
            z_far=4.0,
            conf=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()

        self.base_path = path
        assert os.path.exists(self.base_path)

        if stage == "train":
            file_list = os.path.join(self.base_path, "train.lst")
        elif stage == "val":
            file_list = os.path.join(self.base_path, "val.lst")
        elif stage == "test":
            file_list = os.path.join(self.base_path, "test.lst")

        with open(file_list, "r") as f:
            all_objs = [x.strip() for x in f.readlines()]

        self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading YOLO dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
        )

        self.image_scale = conf["image_scale"]
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
        )

        self.z_near = z_near
        self.z_far = z_far

        self.num_scales = conf["yolo.num_scales"]
        self.num_anchors_per_scale = conf["yolo.num_anchors_per_scale"]
        self.cell_sizes = conf["yolo.cell_sizes"][:self.num_scales]
        self.anchors = conf["yolo.anchors"][:self.num_scales]

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        # read all the images from the folder indicated by the index
        # the needed images are named "image_XXXX.png" with XXXX being the index
        # the needed poses can be read from extrinsic npy files corresponding to the images
        # the intrinsic and extrinsic npy files are named "intrinsic_XXXX.npy" and "extrinsic_XXXX.npy" with the
        # same XXXX as the images

        root_dir = self.all_objs[index]
        root_dir = os.path.join(self.base_path, root_dir)
        all_imgs = []
        all_poses = []
        all_bboxes = []

        # read all the images
        img_count = 0
        while True:
            try:
                img_path = os.path.join(root_dir, "image_{:04d}.png".format(img_count))
                img = imageio.imread(img_path)[..., :3]

                # scale the image
                img = cv2.resize(img, (0, 0), fx=self.image_scale, fy=self.image_scale)

                # turn the image into tensor
                img_tensor = self.image_to_tensor(img)
                all_imgs.append(img_tensor)

                img_count += 1
            except:
                break

        # read all the poses
        for i in range(img_count):
            pose_path = os.path.join(root_dir, "extrinsic_{:04d}.npy".format(i))
            pose = np.load(pose_path)
            pose = (
                    self._coord_trans_world
                    @ torch.tensor(pose, dtype=torch.float32)
                    @ self._coord_trans_cam
            )
            all_poses.append(pose)

        # read all the bounding boxes
        for i in range(img_count):
            bboxes = np.roll(
                np.loadtxt(fname=os.path.join(root_dir, "projected_bboxes_{:04d}.txt"), delimiter=" ", ndmin=2), 4,
                axis=1).tolist()
            all_bboxes.append(self._get_all_bboxes(bboxes, all_imgs[i].shape[0], all_imgs[i].shape[1]))

        intrinsic_path = os.path.join(root_dir, "intrinsic_0000.npy")
        intrinsic = np.load(intrinsic_path)

        # the focal can be read from the intrinsic npy file which are the same for all images
        focal = intrinsic[0, 0] * self.image_scale
        focal = torch.tensor((focal, focal), dtype=torch.float32)

        # the camera center can be read from the intrinsic npy file which are the same for all images
        c = intrinsic[:2, 2] * self.image_scale
        c = torch.tensor(c, dtype=torch.float32)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "bboxes": all_bboxes,
            "poses": all_poses,
            "c": c,
        }

        return result

    def _get_all_bboxes(self, bboxes, height, width):
        # get the grid sizes from the height, width and cell sizes
        grid_sizes = [(height // cs, width // cs) for cs in self.cell_sizes]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, class_label]
        targets = [torch.zeros((self.num_anchors_per_scale, s_h, s_w, 6))
                   for (s_h, s_w) in grid_sizes]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = util.iou(torch.tensor(box[2:4]),
                                   self.anchors,
                                   is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            # At each scale, assigning the bounding box to the
            # best matching anchor box
            has_anchor = [False] * self.num_scales
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Identifying the grid size for the scale
                (s_h, s_w) = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding box belongs
                i, j = int(s_h * y), int(s_w * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative
                    # to the cell
                    x_cell, y_cell = s_w * x - j, s_h * y - i

                    # Calculating the width and height of the bounding box
                    # relative to the cell
                    width_cell, height_cell = (width * s_w, height * s_h)

                    # Indentify the box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell,
                         height_cell]
                    )

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the
                # IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target
        return tuple(targets)
