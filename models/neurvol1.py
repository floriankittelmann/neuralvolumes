# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RayMarchingHelper import init_with_camera_position, init_section_view
from models.colorcals.colorcal1 import Colorcal
from models.decoders.voxel1 import Decoder
from models.encoders.mvconv1 import Encoder
from models.volsamplers.warpvoxel import VolSampler


class Autoencoder(nn.Module):
    def __init__(
            self,
            dataset,
            encoder: Encoder,
            decoder: Decoder,
            volsampler: VolSampler,
            colorcal: Colorcal,
            dt: float,
            stepjitter: float = 0.01,
            estimatebg: bool = False
    ):
        super(Autoencoder, self).__init__()

        self.estimatebg = estimatebg
        self.allcameras = dataset.get_allcameras()

        self.encoder = encoder
        self.decoder = decoder
        self.volsampler = volsampler
        self.bg = nn.ParameterDict({
            k: nn.Parameter(torch.ones(3, v["size"][1], v["size"][0]), requires_grad=estimatebg)
            for k, v in dataset.get_krt().items()})
        self.colorcal = colorcal
        self.dt = dt
        self.stepjitter = stepjitter

        self.imagemean = dataset.imagemean
        self.imagestd = dataset.imagestd

        if dataset.known_background():
            dataset.get_background(self.bg)

    # omit background from state_dict if it's not being estimated
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(Autoencoder, self).state_dict(destination, prefix, keep_vars)
        if not self.estimatebg:
            for k in self.bg.keys():
                del ret[prefix + "bg." + k]
        return ret

    def forward(self, iternum, losslist, camrot, campos, focal, princpt, pixelcoords, validinput,
                fixedcamimage=None, encoding=None, keypoints=None, camindex=None,
                image=None, imagevalid=None, viewtemplate=False,
                outputlist=[]):
        result = {"losses": {}}
        # encode input or get encoding

        if encoding is None:
            encout = self.encoder(fixedcamimage, losslist)
            encoding = encout["encoding"]
            result["losses"].update(encout["losses"])

        # decode
        decout = self.decoder(encoding, campos, losslist)
        result["losses"].update(decout["losses"])
        result["decout"] = decout

        raymarching = init_with_camera_position(pixelcoords, princpt, focal, camrot, campos, self.dt)
        rayrgb, rayalpha = raymarching.do_raymarching(self.volsampler, decout, viewtemplate, self.stepjitter)

        if image is not None:
            imagesize = torch.tensor(image.size()[3:1:-1], dtype=torch.float32, device=pixelcoords.device)
            samplecoords = pixelcoords * 2. / (imagesize[None, None, None, :] - 1.) - 1.

        # color correction / bg
        if camindex is not None:
            rayrgb = self.colorcal(rayrgb, camindex)
            if pixelcoords.size()[1:3] != image.size()[2:4]:
                bg = F.grid_sample(
                    torch.stack([self.bg[self.allcameras[camindex[i].item()]] for i in range(campos.size(0))], dim=0),
                    samplecoords)
            else:
                bg = torch.stack([self.bg[self.allcameras[camindex[i].item()]] for i in range(campos.size(0))], dim=0)

            rayrgb = rayrgb + (1. - rayalpha) * bg.clamp(min=0.)

        if "irgbrec" in outputlist:
            result["irgbrec"] = rayrgb
        if "ialpharec" in outputlist:
            result["ialpharec"] = rayalpha

        # opacity prior
        if "alphapr" in losslist:
            alphaprior = torch.mean(
                torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1)) +
                torch.log(0.1 + 1. - rayalpha.view(rayalpha.size(0), -1)) - -2.20727, dim=-1)
            result["losses"]["alphapr"] = alphaprior

        # irgb loss
        if image is not None:
            if pixelcoords.size()[1:3] != image.size()[2:4]:
                image = F.grid_sample(image, samplecoords)

            # standardize
            rayrgb = (rayrgb - self.imagemean) / self.imagestd
            image = (image - self.imagemean) / self.imagestd

            # compute reconstruction loss weighting
            if imagevalid is not None:
                weight = imagevalid[:, None, None, None].expand_as(image) * validinput[:, None, None, None]
            else:
                weight = torch.ones_like(image) * validinput[:, None, None, None]

            irgbsqerr = weight * (image - rayrgb) ** 2

            if "irgbsqerr" in outputlist:
                result["irgbsqerr"] = irgbsqerr

            if "irgbmse" in losslist:
                irgbmse = torch.sum(irgbsqerr.view(irgbsqerr.size(0), -1), dim=-1)
                irgbmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                result["losses"]["irgbmse"] = (irgbmse, irgbmse_weight)
        return result
