# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from PIL import Image
import torch.utils.data
from scipy.spatial.transform import Rotation as R
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            camerafilter,
            keyfilter,
            framelist,
            subsampletype=None,
            subsamplesize=0,
            imagemean=100.,
            imagestd=25.,
    ):
        # get options
        self.allcameras = ['001', '002', '003', '004', '005', '006']
        self.cameras = list(filter(camerafilter, self.allcameras))
        self.framelist = framelist
        self.framecamlist = [(x, cam)
                for x in self.framelist
                for cam in self.cameras]

        self.fixedcameras = ['005', '003', '004']
        self.keyfilter = keyfilter
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.imagemean = imagemean
        self.imagestd = imagestd

        # compute camera positions
        self.campos = {
            '001': np.array([2.0, 3.464, 0.8]).astype(np.float32),
            '002': np.array([-2.0, 3.464, 0.8]).astype(np.float32),
            '003': np.array([-4.0, 0.0, 0.8]).astype(np.float32),
            '004': np.array([4.0, 0.0, 0.8]).astype(np.float32),
            '005': np.array([-2.0, -3.464, 0.8]).astype(np.float32),
            '006': np.array([2.0, -3.464, 0.8]).astype(np.float32)
        }
        self.camrot = {
            # from_quat([x, y, z, w])
            '001': R.from_quat([-0.353553, 0.612372, 0.353553, 0.612372]).as_matrix().astype(np.float32),
            '002': R.from_quat([-0.612372, 0.353553, 0.612372, 0.353553]).as_matrix().astype(np.float32),
            '003': R.from_quat([-0.707107, 0.000000, 0.707107, 0.000000]).as_matrix().astype(np.float32),
            '004': R.from_quat([0.000000, 0.707107, 0.000000, 0.707107]).as_matrix().astype(np.float32),
            '005': R.from_quat([-0.612372, -0.353553, 0.612372, -0.353553]).as_matrix().astype(np.float32),
            '006': R.from_quat([-0.353553, -0.612372, 0.353553, -0.612372]).as_matrix().astype(np.float32)
        }
        self.focal = {
            '001': np.array([40.0, 40.0]),
            '002': np.array([40.0, 40.0]),
            '003': np.array([40.0, 40.0]),
            '004': np.array([40.0, 40.0]),
            '005': np.array([40.0, 40.0]),
            '006': np.array([40.0, 40.0])
        }

        # what is that??
        self.princpt = {
            '001': np.array([389.7428, 472.3040]),
            '002': np.array([389.7428, 472.3040]),
            '003': np.array([389.7428, 472.3040]),
            '004': np.array([389.7428, 472.3040]),
            '005': np.array([389.7428, 472.3040]),
            '006': np.array([389.7428, 472.3040])
        }

        # load background images for each camera
        if "bg" in self.keyfilter:
            self.bg = {}
            for i, cam in enumerate(self.cameras):
                try:
                    imagepath = "experiments/blender1/data/{}/bg.jpg".format(cam)
                    image = np.asarray(Image.open(imagepath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
                    self.bg[cam] = image
                except:
                    pass

    def get_krt(self):
        return {k: {
                "pos": self.campos[k],
                "rot": self.camrot[k],
                "focal": self.focal[k],
                "princpt": self.princpt[k],
                "size": np.array([667, 1024])}
                for k in self.cameras}

    def get_allcameras(self):
        return self.allcameras

    def known_background(self):
        return "bg" in self.keyfilter

    def get_background(self, bg):
        if "bg" in self.keyfilter:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to("cuda")

    def __len__(self):
        return len(self.framecamlist)

    def __getitem__(self, idx):
        frame, cam = self.framecamlist[idx]
        result = {}
        validinput = True

        if "fixedcamimage" in self.keyfilter:

            ninput = len(self.fixedcameras)
            fixedcamimage = np.zeros((3 * ninput, 512, 334), dtype=np.float32)
            for i in range(ninput):
                imagepath = (
                    "experiments/blender1/data/{}/cam{}_frame{:04}.jpg"
                        .format(self.fixedcameras[i], self.fixedcameras[i], int(frame)))
                image = np.asarray(Image.open(imagepath), dtype=np.uint8)[::2, ::2, :].transpose((2, 0, 1)).astype(np.float32)

                if np.sum(image) == 0:
                    validinput = False
                fixedcamimage[i*3:(i+1)*3, :, :] = image

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
                        "experiments/blender1/data/{}/cam{}_frame{:04}.jpg"
                        .format(cam, cam, int(frame)))
                image = np.asarray(Image.open(imagepath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
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
                        px = np.random.randint(0, width, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                        py = np.random.randint(0, height, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    elif self.subsampletype == "random2":
                        px = np.random.uniform(0, width - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                        py = np.random.uniform(0, height - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    else:
                        px, py = np.meshgrid(np.arange(width).astype(np.float32), np.arange(height).astype(np.float32))
                    result["pixelcoords"] = np.stack((px, py), axis=-1)

        return result

