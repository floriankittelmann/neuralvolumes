from typing import Callable
import os

""" 
profiles
A profile is instantiated by the training or evaluation scripts
and controls how the dataset and autoencoder is created
"""


class TrainBlender2:
    def __init__(
            self,
            get_autoencoder_func: Callable,
            get_dataset_func: Callable,
            batchsize: int,
            maxiter: int,
            lr: float
    ):
        self.get_autoencoder_func = get_autoencoder_func
        self.get_dataset_func = get_dataset_func
        self.batchsize = batchsize
        self.maxiter = maxiter
        self.lr = lr

    def get_batchsize(self) -> int: return self.batchsize

    def get_maxiter(self) -> int: return self.maxiter

    def get_lr(self) -> float: return self.lr

    def get_autoencoder(self, dataset): return self.get_autoencoder_func(dataset)

    def get_dataset(self): return self.get_dataset_func(subsampletype="random2")

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


class ProgressWriter:
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


class Progress:
    """Write out diagnostic images during training."""

    def __init__(
            self,
            get_dataset_func: Callable,
            batchsize: int
    ):
        self.batchsize = batchsize
        self.get_dataset_func = get_dataset_func

    def get_batchsize(self): return self.batchsize

    def get_ae_args(self): return dict(outputlist=["irgbrec"])

    def get_dataset(self): return self.get_dataset_func(maxframes=1)

    def get_writer(self): return ProgressWriter()


class Render:
    """
        Render model with training camera or from novel viewpoints.
        e.g., python render.py {configpath} Render --maxframes 128
    """

    def __init__(
            self,
            get_autoencoder_func: Callable,
            get_dataset_func: Callable,
            batchsize: int = 16,
            cam: str = "rotate",
            maxframes: int = -1,
            showtarget: bool = False,
            viewtemplate: bool = False,
    ):
        self.get_autoencoder_func = get_autoencoder_func
        self.get_dataset_func = get_dataset_func
        self.cam = cam
        self.maxframes = maxframes
        self.showtarget = showtarget
        self.viewtemplate = viewtemplate
        self.batchsize = batchsize

    def get_autoencoder(self, dataset):
        return self.get_autoencoder_func(dataset)

    def get_ae_args(self):
        return dict(outputlist=["irgbrec"], viewtemplate=self.viewtemplate)

    def get_dataset(self):
        import data.utils
        import eval.cameras.rotate as cameralib
        dataset = self.get_dataset_func(camerafilter=lambda x: x == self.cam, maxframes=self.maxframes)
        if self.cam == "rotate":
            camdataset = cameralib.Dataset(len(dataset))
            return data.utils.JoinDataset(camdataset, dataset)
        else:
            return dataset

    def get_writer(self, nthreads=16):
        import eval.writers.videowriter as writerlib
        return writerlib.Writer(
            os.path.dirname(__file__),
            "render_{}{}.mp4".format("rotate" if self.cam is None else self.cam,
                                     "_template" if self.viewtemplate else ""),
            showtarget=self.showtarget,
            nthreads=nthreads
        )
