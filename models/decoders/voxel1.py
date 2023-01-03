# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from models.decoders.AffineMixWarp import AffineMixWarp
from models.decoders.ConvTemplate import ConvTemplate
from models.decoders.LinearTemplate import LinearTemplate
from models.decoders.ConvWarp import ConvWarp
import models.utils


def gettemplate(templatetype, **kwargs):
    if templatetype == "conv":
        return ConvTemplate(**kwargs)
    elif templatetype == "affinemix":
        return LinearTemplate(**kwargs)
    else:
        return None


def getwarp(warptype, **kwargs):
    if warptype == "conv":
        return ConvWarp(**kwargs)
    elif warptype == "affinemix":
        return AffineMixWarp(**kwargs)
    else:
        return None


class Decoder(nn.Module):
    def __init__(
            self,
            templatetype="conv",
            templateres=128,
            viewconditioned=False,
            globalwarp=True,
            warptype="affinemix",
            displacementwarp=False,
            frameindexinfo=False
    ):
        super(Decoder, self).__init__()
        self.templatetype = templatetype
        self.templateres = templateres
        self.viewconditioned = viewconditioned
        self.globalwarp = globalwarp
        self.warptype = warptype
        self.displacementwarp = displacementwarp
        self.frameindexinfo = frameindexinfo

        encodingsize = 256
        if self.frameindexinfo:
            encodingsize = encodingsize + 1

        if self.viewconditioned:
            self.template = gettemplate(self.templatetype, encodingsize=encodingsize + 3,
                                        outchannels=3, templateres=self.templateres)
            self.templatealpha = gettemplate(self.templatetype, encodingsize=256,
                                             outchannels=1, templateres=self.templateres)
        else:
            self.template = gettemplate(self.templatetype, encodingsize=encodingsize, templateres=self.templateres)

        self.warp = getwarp(self.warptype, displacementwarp=self.displacementwarp)

        if self.globalwarp:
            self.quat = models.utils.Quaternion()

            self.gwarps = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3))
            self.gwarpr = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 4))
            self.gwarpt = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3))

            initseq = models.utils.initseq
            for m in [self.gwarps, self.gwarpr, self.gwarpt]:
                initseq(m)

    def forward(self, encoding, viewpos, frameindex, losslist=[]):
        # run template branch
        viewdir = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=-1, keepdim=True))
        templatein = torch.cat([encoding, viewdir], dim=1) if self.viewconditioned else encoding
        frameindex = torch.reshape(frameindex, (frameindex.size()[0], 1))
        templatein = torch.cat([templatein, frameindex], dim=1) if self.frameindexinfo else templatein
        template = self.template(templatein)
        if self.viewconditioned:
            # run alpha branch without viewpoint information
            template = torch.cat([template, self.templatealpha(encoding)], dim=1)

        template = torch.clip(template, min=-1.0, max=1.0)

        # scale up to 0-255
        scaleTwo = torch.tensor([255., 255., 255., 1.], device=encoding.device)[None, :, None, None, None]
        scaleOne = torch.tensor([2., 2., 2., 2.], device=encoding.device)[None, :, None, None, None]
        bias = torch.tensor([1., 1., 1., 1.], device=encoding.device)[None, :, None, None, None]
        template = (template + bias) / scaleOne * scaleTwo

        # compute warp voxel field
        warp = self.warp(encoding) if self.warp is not None else None

        if self.globalwarp:
            # compute single affine transformation
            gwarps = 1.0 * torch.exp(0.05 * self.gwarps(encoding).view(encoding.size(0), 3))
            gwarpr = self.gwarpr(encoding).view(encoding.size(0), 4) * 0.1
            gwarpt = self.gwarpt(encoding).view(encoding.size(0), 3) * 0.025
            gwarprot = self.quat(gwarpr.view(-1, 4)).view(encoding.size(0), 3, 3)

        losses = {}

        # tv-L1 prior
        if "tvl1" in losslist:
            logalpha = torch.log(1e-5 + template[:, -1, :, :, :])
            losses["tvl1"] = torch.mean(torch.sqrt(1e-5 +
                                                   (logalpha[:, :-1, :-1, 1:] - logalpha[:, :-1, :-1, :-1]) ** 2 +
                                                   (logalpha[:, :-1, 1:, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2 +
                                                   (logalpha[:, 1:, :-1, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2))

        return {"template": template, "warp": warp,
                **({"gwarps": gwarps, "gwarprot": gwarprot, "gwarpt": gwarpt} if self.globalwarp else {}),
                "losses": losses}
