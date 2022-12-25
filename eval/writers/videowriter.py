# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import multiprocessing
import os
import shutil
import subprocess
import numpy as np
import matplotlib.cm as cm
from PIL import Image


def writeimage(x):
    itemnum, imgout, outpath = x
    gamma_correction_value = (2. / 1.)
    imgout = np.clip(np.clip(imgout / 255., 0., 255.) ** gamma_correction_value * 255., 0., 255).astype(np.uint8)

    if imgout.shape[1] % 2 != 0:
        imgout = imgout[:, :-1]

    Image.fromarray(imgout).save(os.path.join(outpath, "{:06}.jpg".format(itemnum)))


class Writer:
    def __init__(
            self,
            outpath,
            filename,
            showtarget=False,
            showdiff=False,
            bgcolor=[64., 64., 64.],
            colcorrect=[1., 1., 1.],
            nthreads=16,
            is_plot_batch=False
    ):
        self.showtarget = showtarget
        self.showdiff = showdiff
        self.bgcolor = np.array(bgcolor, dtype=np.float32)
        self.colcorrect = np.array(colcorrect, dtype=np.float32)

        # set up temporary output
        self.randid = ''.join([str(x) for x in np.random.randint(0, 9, size=10)])
        if is_plot_batch:
            outpath_img_folder = os.path.join(outpath, "cache_plots")
        else:
            outpath_img_folder = os.path.join(outpath, "{}".format(self.randid))
        self.outpath_img_folder = outpath_img_folder
        self.outpath_video = os.path.join(outpath, filename)

        try:
            os.makedirs(self.outpath_img_folder)
        except OSError:
            pass

        self.writepool = multiprocessing.Pool(nthreads)
        self.nitems = 0

    def batch(
            self,
            iternum,
            itemnum,
            irgbrec,
            ialpharec=None,
            image=None,
            irgbsqerr=None,
            **kwargs
    ):
        irgbrec = irgbrec.data.to("cpu").numpy().transpose((0, 2, 3, 1))
        if ialpharec is not None:
            ialpharec = ialpharec.data.to("cpu").numpy()[:, 0, :, :, None]
        else:
            ialpharec = 1.0

        # color correction
        imgout = irgbrec * self.colcorrect[None, None, None, :]

        # composite background color
        imgout = imgout + (1. - ialpharec) * self.bgcolor[None, None, None, :]

        # concatenate ground truth image
        if self.showtarget and image is not None:
            image = image.data.to("cpu").numpy().transpose((0, 2, 3, 1))
            image = image * self.colcorrect[None, None, None, :]
            imgout = np.concatenate((imgout, image), axis=2)

        # concatenate difference image
        if self.showdiff and imagediff is not None:
            irgbsqerr = np.mean(irgbsqerr.data.to("cpu").numpy(), axis=1)
            irgbsqerr = (cm.magma(4. * irgbsqerr / 255.)[:, :, :, :3] * 255.)
            imgout = np.concatenate((imgout, irgbsqerr), axis=2)

        outpath_img_folder = self.outpath_img_folder
        self.writepool.map(writeimage,
                           zip(itemnum.data.to("cpu").numpy(),
                               imgout,
                               [outpath_img_folder for i in range(itemnum.size(0))]))
        self.nitems += itemnum.size(0)

    def finalize(self):
        # make video file
        command = (
            "ffmpeg -y -r 30 -i {}/%06d.jpg "
            "-vframes {} "
            "-vcodec libx264 -crf 18 "
            "-pix_fmt yuv420p "
            "{}".format(self.outpath_img_folder, self.nitems, self.outpath_video)
        ).split()
        subprocess.call(command)
        shutil.rmtree(self.outpath_img_folder)
