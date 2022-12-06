import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.utils


class AffineMixWarp(nn.Module):
    def __init__(self, **kwargs):
        super(AffineMixWarp, self).__init__()

        self.quat = models.utils.Quaternion()

        self.warps = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 3 * 16))
        self.warpr = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 4 * 16))
        self.warpt = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 3 * 16))
        self.weightbranch = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 16 * 32 * 32 * 32))
        for m in [self.warps, self.warpr, self.warpt, self.weightbranch]:
            models.utils.initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

    def forward(self, encoding):
        warps = self.warps(encoding).view(encoding.size(0), 16, 3)
        warpr = self.warpr(encoding).view(encoding.size(0), 16, 4)
        warpt = self.warpt(encoding).view(encoding.size(0), 16, 3) * 0.1
        warprot = self.quat(warpr.view(-1, 4)).view(encoding.size(0), 16, 3, 3)

        weight = torch.exp(self.weightbranch(encoding).view(encoding.size(0), 16, 32, 32, 32))

        warpedweight = torch.cat([
            F.grid_sample(weight[:, i:i + 1, :, :, :],
                          torch.sum(((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :] *
                                     warprot[:, None, None, None, i, :, :]), dim=5) *
                          warps[:, None, None, None, i, :], padding_mode='border')
            for i in range(weight.size(1))], dim=1)

        warp = torch.sum(torch.stack([
            warpedweight[:, i, :, :, :, None] *
            (torch.sum(((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :] *
                        warprot[:, None, None, None, i, :, :]), dim=5) *
             warps[:, None, None, None, i, :])
            for i in range(weight.size(1))], dim=1), dim=1) / torch.sum(warpedweight, dim=1).clamp(min=0.001)[:, :, :,
                                                              :, None]

        return warp.permute(0, 4, 1, 2, 3)
