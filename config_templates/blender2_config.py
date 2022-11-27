# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from data.Datasets.Blender2Dataset import Blender2Dataset
from data.Profiles.Blender2Profiles import TrainBlender2
from data.Profiles.Blender2Profiles import Progress
from data.Profiles.Blender2Profiles import ProgressWriter
from data.Profiles.Blender2Profiles import Render
from typing import Callable


def get_dataset_config_func(
        camerafilter: Callable[[str], bool] = lambda x: True,
        maxframes: int = -1,
        subsampletype=None
):
    return Blender2Dataset(
        camerafilter=camerafilter,
        framelist=[i for i in range(1, 502, 3)][:maxframes],
        keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],
        imagemean=100.,
        imagestd=25.,
        subsampletype=subsampletype,
        subsamplesize=128,
        scale_focal=3.0,
        scale_factor=2.0
    )


def get_autoencoder_config_func(dataset):
    import models.neurvol1 as aemodel
    import models.encoders.mvconv1 as encoderlib
    import models.decoders.voxel1 as decoderlib
    import models.volsamplers.warpvoxel as volsamplerlib
    import models.colorcals.colorcal1 as colorcalib
    return aemodel.Autoencoder(
        dataset,
        encoderlib.Encoder(3),
        decoderlib.Decoder(globalwarp=False, warptype=None, viewconditioned=False),
        volsamplerlib.VolSampler(),
        colorcalib.Colorcal(dataset.get_allcameras()),
        4. / 256)

def get_progress():
    return Progress(
        get_dataset_func=get_dataset_config_func,
        batchsize=32
    )

def get_train_profile() -> TrainBlender2:
    return TrainBlender2(
        get_autencoder_func=get_autoencoder_config_func,
        get_dataset_func=get_dataset_config_func,
        batchsize=32,
        maxiter=500000,
        lr=0.0001
    )

