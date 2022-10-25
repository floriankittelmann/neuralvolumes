# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

import torch.utils.data
from data.blender2_utils import CameraInSetup


class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, period=128):
        self.camera = CameraInSetup(1)
        self.length = length
        self.period = period
        self.width, self.height = self.camera.get_img_width(), self.camera.get_img_height() + 1

        self.focal = np.array([self.camera.get_focal_length(), self.camera.get_focal_length()], dtype=np.float32)
        self.princpt = np.array(self.camera.get_principt(), dtype=np.float32)

    def __len__(self):
        return self.length

    def get_allcameras(self):
        return ["rotate"]

    def get_krt(self):
        return {"rotate": {
            "focal": self.focal,
            "princpt": self.princpt,
            "size": np.array([self.height, self.width])}}

    def __getitem__(self, idx):
        # t = (np.cos(idx * 2. * np.pi / self.period) * 0.5 + 0.5)
        campos = self.camera.get_cam_pos_training().astype(np.float32) / 3.5

        """lookat = np.array([0., 0., 0.], dtype=np.float32)
        up = np.array([0., 0., 1.], dtype=np.float32)
        forward = lookat - campos
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)"""

        camrot = self.camera.get_cam_rot_matrix_training().astype(np.float32)

        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1)

        return {"campos": campos,
                "camrot": camrot,
                "focal": self.focal,
                "princpt": self.princpt,
                "pixelcoords": pixelcoords}
