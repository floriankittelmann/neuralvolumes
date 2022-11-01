# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import data.blender2 as datamodel


def get_dataset(camerafilter=lambda x: True, maxframes=-1, subsampletype=None):
    return datamodel.Dataset(
        camerafilter=camerafilter,
        framelist=[i for i in range(1, 502, 3)][:maxframes],
        keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],
        imagemean=100.,
        imagestd=25.,
        subsampletype=subsampletype,
        subsamplesize=128,
        scale_focal=1.5
    )


def get_autoencoder(dataset):
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


### profiles
# A profile is instantiated by the training or evaluation scripts
# and controls how the dataset and autoencoder is created
class Train():
    batchsize = 16
    maxiter = 500000
    lr = 0.0001
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_dataset(self): return get_dataset(subsampletype="random2")
    def get_optimizer(self, ae):
        import itertools
        import torch.optim
        aeparams = itertools.chain(
            [{"params": x} for x in ae.encoder.parameters()],
            [{"params": x} for x in ae.decoder.parameters()],
            [{"params": x} for x in ae.colorcal.parameters()])
        return torch.optim.Adam(aeparams, lr=self.lr, betas=(0.9, 0.999))

    def get_loss_weights(self):
        return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}


class ProgressWriter():
    def batch(self, iternum, itemnum, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []
        if len(rows) == 0:
            rows.append(np.concatenate(row, axis=1))
        imgout = np.concatenate(rows, axis=0)
        outpath = os.path.dirname(__file__)
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(
            os.path.join(outpath, "prog_{:06}.jpg".format(iternum)))


class Progress():
    """Write out diagnostic images during training."""
    batchsize = 16
    def get_ae_args(self): return dict(outputlist=["irgbrec"])
    def get_dataset(self): return get_dataset(maxframes=1)
    def get_writer(self): return ProgressWriter()


class Render():
    """Render model with training camera or from novel viewpoints.
    
    e.g., python render.py {configpath} Render --maxframes 128"""

    def __init__(self, cam="rotate", maxframes=-1, showtarget=False, viewtemplate=False):
        self.cam = cam
        self.maxframes = maxframes
        self.showtarget = showtarget
        self.viewtemplate = viewtemplate
        self.batchsize = 16

    def get_autoencoder(self, dataset):
        return get_autoencoder(dataset)

    def get_ae_args(self):
        return dict(outputlist=["irgbrec"], viewtemplate=self.viewtemplate)

    def get_dataset(self):
        import data.utils
        import eval.cameras.rotate as cameralib
        dataset = get_dataset(camerafilter=lambda x: x == self.cam, maxframes=self.maxframes)
        if self.cam == "rotate":
            camdataset = cameralib.Dataset(len(dataset))
            return data.utils.JoinDataset(camdataset, dataset)
        else:
            return dataset

    def get_writer(self):
        import eval.writers.videowriter as writerlib
        return writerlib.Writer(
            os.path.dirname(__file__),
            "render_{}{}.mp4".format("rotate" if self.cam is None else self.cam,
                                     "_template" if self.viewtemplate else ""),
            showtarget=self.showtarget
        )
