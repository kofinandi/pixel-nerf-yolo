import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class YOLODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        stage="train",
        image_size=0.25,
        z_near=1.2,
        z_far=4.0,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
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

        self.image_size = image_size
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
        )

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

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

        # read all the images
        img_count = 6
        for i in range(img_count):
            img_path = os.path.join(root_dir, "image_{:04d}.png".format(i))
            img = imageio.imread(img_path)[..., :3]

            # scale the image
            img = cv2.resize(img, (0, 0), fx=self.image_size, fy=self.image_size)

            # turn the image into tensor
            img_tensor = self.image_to_tensor(img)
            all_imgs.append(img_tensor)

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

        intrinsic_path = os.path.join(root_dir, "intrinsic_0000.npy")
        intrinsic = np.load(intrinsic_path)

        # the focal can be read from the intrinsic npy file which are the same for all images
        focal = intrinsic[0, 0] * self.image_size
        focal = torch.tensor((focal, focal), dtype=torch.float32)

        # the camera center can be read from the intrinsic npy file which are the same for all images
        c = intrinsic[:2, 2] * self.image_size
        c = torch.tensor(c, dtype=torch.float32)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "c": c,
        }

        return result
