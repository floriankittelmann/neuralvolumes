# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np

import torch.utils.data
from data.CameraSetups.CameraSetupInBlender2 import CameraSetupInBlender2
from scipy.spatial.transform import Rotation as R


class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, period=128):
        self.camera = CameraSetupInBlender2(1)
        self.length = length
        self.period = period
        self.width, self.height = self.camera.get_img_width(), self.camera.get_img_height() + 1

        self.focal = np.array([self.camera.get_focal_length(), self.camera.get_focal_length()], dtype=np.float32)
        self.princpt = np.array([self.camera.get_principt_width(), self.camera.get_principt_height()], dtype=np.float32)

    def __len__(self):
        return self.length

    def get_allcameras(self):
        return ["rotate"]

    def get_krt(self):
        return {"rotate": {
            "focal": self.focal,
            "princpt": self.princpt,
            "size": np.array([self.width, self.height])}}

    def __getitem__(self, idx):
        # t = (np.cos(idx * 2. * np.pi / self.period) * 0.5 + 0.5)

        nof_frames = 500.0
        radius = self.camera.get_radius()
        step = 2.0 * math.pi / nof_frames
        alpha = idx * step
        x = math.cos(alpha) * radius
        y = math.sin(alpha) * radius
        z = 0.0

        z_rot = alpha + 90
        xyz_rot = [0.0, 0.0, z_rot]
        xyz_rot = R.from_euler('xyz', xyz_rot, degrees=True)
        extrinsic_matrix = np.array(xyz_rot.as_matrix()).astype(np.float32)
        rad_rot = 180.0 / 360.0 * 2 * math.pi
        y_rot_matrix = np.asarray([
            [math.cos(rad_rot), 0, math.sin(rad_rot)],
            [0, 1, 0],
            [-math.sin(rad_rot), 0, math.cos(rad_rot)],
        ])
        camrot = extrinsic_matrix.dot(y_rot_matrix).T.astype(np.float32)
        xyz_pos = [x, y, z]
        campos = np.array(xyz_pos).astype(np.float32)

        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1)

        return {"campos": campos,
                "camrot": camrot,
                "focal": self.focal,
                "princpt": self.princpt,
                "pixelcoords": pixelcoords}
