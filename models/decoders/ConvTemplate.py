import numpy as np
import torch.nn as nn
import models.utils


class ConvTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(ConvTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        # build template convolution stack
        self.template1 = nn.Sequential(nn.Linear(self.encodingsize, 1024), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 1024, 512
        for i in range(int(np.log2(self.templateres)) - 1):
            template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        template2.append(nn.ConvTranspose3d(inchannels, 4, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            models.utils.initseq(m)

    def forward(self, encoding):
        return self.template2(self.template1(encoding).view(-1, 1024, 1, 1, 1))
