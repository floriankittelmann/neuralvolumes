from typing import Callable
import os

from data.Datasets.Blender2Dataset import Blender2Dataset

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
            lr: float,
            should_train_with_ground_truth: bool
    ):
        self.get_autoencoder_func = get_autoencoder_func
        self.get_dataset_func = get_dataset_func
        self.batchsize = batchsize
        self.maxiter = maxiter
        self.lr = lr
        self.should_train_with_ground_truth = should_train_with_ground_truth

    def get_should_train_with_ground_truth(self):
        return self.should_train_with_ground_truth

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
        #return {"irgbmse": 1000.0, "kldiv": 0.01, "alphapr": 0.01, "tvl1": 0.01}
        return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}


class ProgressWriter:
    def batch(self, iternum, itemnum, outpath, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]
                ), axis=1))
            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []
        if len(rows) == 0:
            rows.append(np.concatenate(row, axis=1))
        imgout = np.concatenate(rows, axis=0)
        path_of_img = os.path.join(outpath, "prog_{:06}.jpg".format(iternum))
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(path_of_img)
        return imgout


class Progress:
    """Write out diagnostic images during training."""

    def __init__(
            self,
            get_dataset_func: Callable,
            batchsize: int,
    ):
        self.batchsize = batchsize
        self.get_dataset_func = get_dataset_func

    def get_batchsize(self): return self.batchsize

    def get_ae_args(self): return dict(outputlist=["irgbrec"])

    def get_dataset(self): return self.get_dataset_func()

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
            resolution_mode: int = Blender2Dataset.MODE_1024x668_LOSSIMG_INPUT_RES,
    ):
        self.get_autoencoder_func = get_autoencoder_func
        self.get_dataset_func = get_dataset_func
        self.cam = cam
        self.maxframes = maxframes
        self.showtarget = showtarget
        self.viewtemplate = viewtemplate
        self.batchsize = batchsize
        self.resolution_mode = resolution_mode

    def get_autoencoder(self, dataset):
        return self.get_autoencoder_func(dataset)

    def get_ae_args(self):
        return dict(outputlist=["irgbrec"], viewtemplate=self.viewtemplate)

    def get_dataset(self):
        import data.utils
        import eval.cameras.rotate as cameralib

        if self.cam == "rotate":
            dataset = self.get_dataset_func(camerafilter=lambda x: x == "000", maxframes=self.maxframes)
            dataset = cameralib.Dataset(len(dataset), dataset)
            #dataset = data.utils.JoinDataset(camdataset, dataset)
        else:
            dataset = self.get_dataset_func(camerafilter=lambda x: x == self.cam, maxframes=self.maxframes)
        return dataset

    def get_writer(self, outpath, nthreads=16, is_plot_batch=False):
        import eval.writers.videowriter as writerlib
        return writerlib.Writer(outpath,
            "render_{}{}.mp4".format("rotate" if self.cam is None else self.cam,
                                     "_template" if self.viewtemplate else ""),
            showtarget=self.showtarget,
            nthreads=nthreads,
            is_plot_batch=is_plot_batch
        )
