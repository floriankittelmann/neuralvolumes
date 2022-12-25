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
    def __init__(self, length: int, dataset_to_render: Blender2Dataset):
        self.length = length
        self.dataset_to_render = dataset_to_render

    def __len__(self) -> int:
        return self.length

    def get_allcameras(self):
        return self.dataset_to_render.get_allcameras()

    def get_krt(self):
        return self.dataset_to_render.get_krt()

    def known_background(self):
        return True

    def get_background(self, bg) -> None:
        self.dataset_to_render.get_background(bg)

    def __getitem__(self, idx):
        originDataset = self.dataset_to_render[idx]
        camera = CameraSetupInBlender2(0)
        campos, camrot = camera.get_render_rot(idx)
        originDataset['campos'] = campos
        originDataset['camrot'] = camrot
        return originDataset
