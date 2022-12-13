# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from PIL import Image
import torch.utils.data
from data.CameraSetups.CameraSetupInBlender2 import CameraSetupInBlender2
from typing import List
from typing import Callable


class Blender2Dataset(torch.utils.data.Dataset):

    MODE_512x334_ENCODER_INPUT_RES = 1
    MODE_256x167_ENCODER_INPUT_RES = 2

    MODE_1024x668_LOSSIMG_INPUT_RES = 3
    MODE_512x334_LOSSIMG_INPUT_RES = 4

    MODE_128x84 = 5

    def __init__(
            self,
            camerafilter: Callable[[str], bool],
            keyfilter: List[str],
            framelist: List[int],
            encoder_input_imgsize: int,
            loss_imgsize_mode: int,
            fixedcameras: List[str] = ['028', '001', '019'],
            subsampletype=None,
            subsamplesize: int = 0,
            imagemean: float = 100.,
            imagestd: float = 25.,
            scale_factor: float = 1.0,
            scale_focal: float = 1.0
    ):
        # get options
        self.allcameras = []
        self.campos = {}
        self.camrot = {}
        self.focal = {}
        self.princpt = {}
        self.encoder_input_imgsize = encoder_input_imgsize
        self.loss_imgsize_mode = loss_imgsize_mode

        self.width = 1024
        self.height = 667
        if self.loss_imgsize_mode == self.MODE_512x334_LOSSIMG_INPUT_RES:
            self.width = 512
            self.height = 334
        elif self.loss_imgsize_mode == self.MODE_128x84:
            self.width = 128
            self.height = 84

        for camera_nr in range(36):
            camera_str = "{:03d}".format(camera_nr)
            camera = CameraSetupInBlender2(camera_nr)
            self.allcameras.append(camera_str)

            # cameras need to be scaled because the volume is normalized to the space of [1,-1]^3
            self.campos[camera_str] = camera.get_cam_pos_training() * scale_factor
            self.camrot[camera_str] = camera.get_cam_rot_matrix_training()

            # the focal length does not needed to normalize because it is given in px
            self.focal[camera_str] = np.array([
                camera.get_focal_length(self.height, self.width) * scale_focal,
                camera.get_focal_length(self.height, self.width) * scale_focal
            ])
            self.princpt[camera_str] = np.array([
                camera.get_principt_height(self.height),
                camera.get_principt_width(self.width)
            ])

        self.cameras = list(filter(camerafilter, self.allcameras))
        self.framelist = framelist
        self.framecamlist = [(x, cam)
                             for x in self.framelist
                             for cam in (self.cameras if len(self.cameras) > 0 else [None])]
        self.fixedcameras = fixedcameras
        self.keyfilter = keyfilter
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.imagemean = imagemean
        self.imagestd = imagestd

        # load background images for each camera
        if "bg" in self.keyfilter:
            self.bg = {}
            for i, cam in enumerate(self.cameras):
                try:
                    imagepath = self.get_bg_img_path()
                    resize_param_loss_imgs = 1
                    if self.loss_imgsize_mode == self.MODE_512x334_LOSSIMG_INPUT_RES:
                        resize_param_loss_imgs = 2
                    elif self.loss_imgsize_mode == self.MODE_128x84:
                        resize_param_loss_imgs = 8
                    image = np.asarray(Image.open(imagepath), dtype=np.uint8)
                    image = image[::resize_param_loss_imgs, ::resize_param_loss_imgs, :].transpose((2, 0, 1)) \
                        .astype(np.float32)
                    self.bg[cam] = image
                except:
                    pass

    def get_bg_img_path(self):
        return "experiments/blender2/data/bg.jpg"

    def get_krt(self) -> dict:
        return {k: {
            "pos": self.campos[k],
            "rot": self.camrot[k],
            "focal": self.focal[k],
            "princpt": self.princpt[k],
            "size": np.array([self.height, self.width])}
            for k in self.cameras}

    def get_allcameras(self) -> list:
        return self.allcameras

    def get_nof_frames(self) -> int:
        return len(self.framelist)

    def known_background(self) -> bool:
        return "bg" in self.keyfilter

    def get_background(self, bg) -> None:
        if "bg" in self.keyfilter:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to("cuda")

    def __len__(self) -> int:
        return len(self.framecamlist)

    def get_images_path(self):
        return "experiments/blender2/data"

    def __getitem__(self, idx: int) -> dict:
        frame, cam = self.framecamlist[idx]
        result = {}
        validinput = True
        if "fixedcamimage" in self.keyfilter:

            ninput = len(self.fixedcameras)
            fixedimg_inputshape = (512, 334)
            resize_param = 2
            if self.encoder_input_imgsize == self.MODE_256x167_ENCODER_INPUT_RES:
                fixedimg_inputshape = (256, 167)
                resize_param = 4
            elif self.encoder_input_imgsize == self.MODE_128x84:
                fixedimg_inputshape = (128, 84)
                resize_param = 8
            fixedcamimage = np.zeros((3 * ninput, fixedimg_inputshape[0], fixedimg_inputshape[1]), dtype=np.float32)

            for i in range(ninput):
                imagepath = (
                    "{}/{}/cam{}_frame{:04}.jpg"
                        .format(self.get_images_path(), self.fixedcameras[i], self.fixedcameras[i], int(frame)))
                image = np.asarray(Image.open(imagepath), dtype=np.uint8)[::resize_param, ::resize_param, :]\
                    .transpose((2, 0, 1)).astype(np.float32)

                if np.sum(image) == 0:
                    validinput = False
                fixedcamimage[i * 3:(i + 1) * 3, :, :] = image

            fixedcamimage[:] -= self.imagemean
            fixedcamimage[:] /= self.imagestd
            result["fixedcamimage"] = fixedcamimage

        result["validinput"] = np.float32(1.0 if validinput else 0.0)

        # image data
        if cam is not None:
            if "camera" in self.keyfilter:
                # camera data
                result["camrot"] = self.camrot[cam]
                result["campos"] = self.campos[cam]
                result["focal"] = self.focal[cam]
                result["princpt"] = self.princpt[cam]
                result["camindex"] = self.allcameras.index(cam)

            if "image" in self.keyfilter:
                # image
                imagepath = (
                    "{}/{}/cam{}_frame{:04}.jpg"
                        .format(self.get_images_path(), cam, cam, int(frame)))
                resize_param_loss_imgs = 1
                if self.loss_imgsize_mode == self.MODE_512x334_LOSSIMG_INPUT_RES:
                    resize_param_loss_imgs = 2
                elif self.loss_imgsize_mode == self.MODE_128x84:
                    resize_param_loss_imgs = 8
                image = np.asarray(Image.open(imagepath), dtype=np.uint8)
                image = image[::resize_param_loss_imgs, ::resize_param_loss_imgs, :].transpose((2, 0, 1))\
                    .astype(np.float32)
                height, width = image.shape[1:3]
                valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
                result["image"] = image
                result["imagevalid"] = valid

                if "pixelcoords" in self.keyfilter:
                    if self.subsampletype == "patch":
                        indx = np.random.randint(0, width - self.subsamplesize + 1)
                        indy = np.random.randint(0, height - self.subsamplesize + 1)

                        px, py = np.meshgrid(
                            np.arange(indx, indx + self.subsamplesize).astype(np.float32),
                            np.arange(indy, indy + self.subsamplesize).astype(np.float32))
                    elif self.subsampletype == "random":
                        px = np.random.randint(0, width, size=(self.subsamplesize, self.subsamplesize)).astype(
                            np.float32)
                        py = np.random.randint(0, height, size=(self.subsamplesize, self.subsamplesize)).astype(
                            np.float32)
                    elif self.subsampletype == "random2":
                        px = np.random.uniform(0, width - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(
                            np.float32)
                        py = np.random.uniform(0, height - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(
                            np.float32)
                    else:
                        px, py = np.meshgrid(np.arange(width).astype(np.float32), np.arange(height).astype(np.float32))
                    result["pixelcoords"] = np.stack((px, py), axis=-1)

        return result
