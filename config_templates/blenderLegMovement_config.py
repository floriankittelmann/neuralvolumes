# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from data.Datasets.Blender2Dataset import Blender2Dataset
from data.Datasets.BlenderLegMovementTrainDataset import BlenderLegMovementTrainDataset
from data.Datasets.BlenderLegMovementTestDataset import BlenderLegMovementTestDataset
from data.Profiles.Blender2Profiles import TrainBlender2
from data.Profiles.Blender2Profiles import Progress
from data.Profiles.Blender2Profiles import ProgressWriter
from data.Profiles.Blender2Profiles import Render
from typing import Callable
import models.encoders.mvconv1 as encoderlib
import models.decoders.voxel1 as decoderlib
import models.volsamplers.warpvoxel as volsamplerlib
import models.colorcals.colorcal1 as colorcalib
from models.neurvol1 import Autoencoder


class DatasetConfig:

    def __init__(self):
        self.loss_mode_res = Blender2Dataset.MODE_128x84
        self.encoder_mode_res = Blender2Dataset.MODE_128x84
        # use all: self.loss_cameras = ['028', '034' '004', '010', '016', '022']
        # use default: self.encoder_input_cameras = ['028', '004', '016']
        self.ground_truth_resolution = 48
        self.subsamplesize = 48
        self.templatesize = 48

    def get_train_dataset_config_func(
            self,
            camerafilter: Callable[[str], bool] = lambda x: True,
            maxframes: int = -1,
            subsampletype=None
    ) -> BlenderLegMovementTrainDataset:
        return BlenderLegMovementTrainDataset(
            camerafilter=camerafilter,
            framelist=[i for i in range(0, 3200, 1)][:maxframes],
            encoder_input_imgsize=self.encoder_mode_res,
            loss_imgsize_mode=self.loss_mode_res,
            keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],

            imagemean=100.,
            imagestd=25.,
            subsampletype=subsampletype,
            subsamplesize=self.subsamplesize,
            scale_focal=1.0,
            scale_factor=1.0,
            ground_truth_resolution=self.ground_truth_resolution
        )

    def get_test_dataset_config_func(
            self,
            camerafilter: Callable[[str], bool] = lambda x: True,
            maxframes: int = -1,
            subsampletype=None
    ) -> BlenderLegMovementTestDataset:
        return BlenderLegMovementTestDataset(
            camerafilter=camerafilter,
            framelist=[i for i in range(0, 800, 1)][:maxframes],
            encoder_input_imgsize=self.encoder_mode_res,
            loss_imgsize_mode=self.loss_mode_res,
            keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],
            imagemean=100.,
            imagestd=25.,
            subsampletype=subsampletype,
            subsamplesize=self.subsamplesize,
            scale_focal=1.0,
            scale_factor=1.0,
            ground_truth_resolution=self.ground_truth_resolution
        )

    def get_autoencoder_config_func(self, dataset) -> Autoencoder:
        template_size = self.templatesize
        raymarching_dt = 2. / float(template_size - 4)
        print("template resolution: ({}, {}, {})".format(template_size, template_size, template_size))
        return Autoencoder(
            dataset=dataset,
            encoder=encoderlib.Encoder(3, encoder_input_mod=self.encoder_mode_res),
            decoder=decoderlib.Decoder(
                globalwarp=True,
                warptype="affinemix",
                templatetype="affinemix",
                viewconditioned=True,
                templateres=template_size),
            volsampler=volsamplerlib.VolSampler(),
            colorcal=colorcalib.Colorcal(dataset.get_allcameras()),
            dt=raymarching_dt,
            ground_truth_resolution=self.ground_truth_resolution
        )

    def get_train_profile(self) -> TrainBlender2:
        return TrainBlender2(
            get_autoencoder_func=self.get_autoencoder_config_func,
            get_dataset_func=self.get_train_dataset_config_func,
            batchsize=32,
            maxiter=500000,
            lr=0.0001,
            should_train_with_ground_truth=True
        )

    def get_progress(self) -> Progress:
        return Progress(
            get_dataset_func=self.get_test_dataset_config_func,
            batchsize=32
        )

    def get_progresswriter(self) -> ProgressWriter:
        return ProgressWriter()

    def get_render_profile(self) -> Render:
        return Render(
            get_autoencoder_func=self.get_autoencoder_config_func,
            get_dataset_func=self.get_test_dataset_config_func,
            batchsize=16,
            resolution_mode=self.loss_mode_res
        )
