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
from data.Datasets.Blender2Dataset import Blender2Dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, img_res_mode: int, period=128):
        self.camera = CameraSetupInBlender2(1)
        self.length = length
        self.period = period
        if img_res_mode == Blender2Dataset.MODE_512x334_LOSSIMG_INPUT_RES:
            self.width, self.height = 512, 334
        elif img_res_mode == Blender2Dataset.MODE_128x84:
            self.width, self.height = 128, 84
        else:
            self.width, self.height = 1024, 668

        self.focal = np.array([
            self.camera.get_focal_length(self.height, self.width),
            self.camera.get_focal_length(self.height, self.width)
        ], dtype=np.float32)
        self.princpt = np.array([
            self.camera.get_principt_width(self.width),
            self.camera.get_principt_height(self.height)
        ], dtype=np.float32)

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
