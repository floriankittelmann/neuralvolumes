import torch.nn as nn
import models.utils


class LinearTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(LinearTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(
            nn.Linear(self.encodingsize, 8), nn.LeakyReLU(0.2),
            nn.Linear(8, self.templateres ** 3 * self.outchannels))

        for m in [self.template1]:
            models.utils.initseq(m)

    def forward(self, encoding):
        return self.template1(encoding).view(-1, self.outchannels, self.templateres, self.templateres, self.templateres)
