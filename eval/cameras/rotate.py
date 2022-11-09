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
        campos, camrot = self.camera.get_render_rot(idx)

        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1)

        return {"campos": campos,
                "camrot": camrot,
                "focal": self.focal,
                "princpt": self.princpt,
                "pixelcoords": pixelcoords}
