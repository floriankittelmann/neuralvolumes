import numpy as np
import torch
import torch.nn as nn
import models.utils


class ConvWarp(nn.Module):
    def __init__(self, displacementwarp=False, **kwargs):
        super(ConvWarp, self).__init__()

        self.displacementwarp = displacementwarp

        self.warp1 = nn.Sequential(
            nn.Linear(256, 1024), nn.LeakyReLU(0.2))
        self.warp2 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(512, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 3, 4, 2, 1))
        for m in [self.warp1, self.warp2]:
            models.utils.initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=0)[None].astype(np.float32)))

    def forward(self, encoding):
        finalwarp = self.warp2(self.warp1(encoding).view(-1, 1024, 1, 1, 1)) * (2. / 1024)
        if not self.displacementwarp:
            finalwarp = finalwarp + self.grid
        return finalwarp
