# worth to have a look: https://github.com/NVIDIAGameWorks/kaolin
import os
from typing import Callable
import pyvista as pv
import torch
from pyvista import Plotter
from models.volsamplers.warpvoxel import VolSampler
import numpy as np


class NeuralVolumePlotter:

    def __init__(self):
        self.plotter = pv.Plotter()

    def __prepare_template_np_plot(self, np_filename: str, density: float):
        template = None
        with open(np_filename, 'rb') as f:
            template = np.load(f)
        if template is None:
            raise Exception("should load file")

        min = -1.0
        max = 1.0

        distribution = np.arange(min, max, (2.0 / density))
        x, y, z = np.meshgrid(distribution, distribution, distribution)
        pos = np.stack((x, y, z), axis=3)
        dimension = int(density ** 3)
        batchsize = template.shape[0]
        pos = np.array([pos for i in range(batchsize)])
        pos = pos.reshape((batchsize, 1, dimension, 1, 3))

        torch.cuda.set_device("cuda:0")
        cur_device = torch.cuda.current_device()

        pos = torch.from_numpy(pos)
        pos = pos.to(cur_device)

        template = torch.from_numpy(template)
        template = template.to(cur_device)

        volsampler = VolSampler()
        sample_rgb, sample_alpha = volsampler(pos=pos, template=template)

        sample_rgb = sample_rgb.cpu().numpy().reshape((batchsize, dimension, 3))
        sample_alpha = sample_alpha.cpu().numpy().reshape((batchsize, dimension, 1))
        pos = pos.cpu().numpy().reshape((batchsize, dimension, 3))

        sample_rgba = np.zeros((batchsize, dimension, 4))
        sample_rgba[:, :, 0:3] = sample_rgb
        sample_rgba[:, :, 3] = sample_alpha[:, :, 0]
        sample_rgba = sample_rgba.reshape((batchsize, int(density), int(density), int(density), 4))
        return pos, sample_rgba.astype(float), dimension

    def pyvista_3d_from_template_np(
            self,
            outputfolder: str,
            overwrite_color_to_black: bool = False,
            nof_frames: int = 500
    ):
        plotter_test: Plotter = pv.Plotter()
        prepare_template: Callable = self.__prepare_template_np_plot
        density: float = 128.0

        def create_points(value):
            plotter_test.clear_actors()
            res: int = int(value)
            np_filename: str = os.path.join(outputfolder, "frame{}.npy".format(res))
            pos, sample_rgba, dimension = prepare_template(np_filename, density)
            min, max = -1.0, 1.0
            x = np.arange(min, max, (2.0 / density))
            y = np.arange(min, max, (2.0 / density))
            z = np.arange(min, max, (2.0 / density))
            x, y, z = np.meshgrid(x, y, z)

            if overwrite_color_to_black:
                sample_rgba[:, :, :, :, 0:3] = np.zeros(sample_rgba[:, :, :, :, 0:3].shape)

            sample_rgba[:, :, :, :, 3] = sample_rgba[:, :, :, :, 3] / 255.
            grid = pv.StructuredGrid(x, y, z)

            #batchsize = sample_rgba.shape[0]
            #for i in range(batchsize):
            sample_rgb = sample_rgba[0, :, :, :, 0:4].reshape((dimension, 4))
            actor = plotter_test.add_points(grid.points, scalars=sample_rgb, rgb=True)
            return

        numpy_files = [f for f in os.listdir(outputfolder) if os.path.isfile(os.path.join(outputfolder, f)) and f.endswith(".npy")]
        plotter_test.add_slider_widget(callback=create_points, value=0, rng=[0, len(numpy_files)], title='Time')
        plotter_test.show_axes_all()
        plotter_test.show()


if __name__ == "__main__":

    plotter = NeuralVolumePlotter()
    filename = "experiments/blender2/20221206_123045_train_faster_smaller_res/templates"
    plotter.pyvista_3d_from_template_np(filename, False)
    #path_stl = "C:\\Users\\Flori\\Desktop\\Test-Export-Blender\\test0.stl"
    #plotter.plot_stl_pyvista(path_stl)
    #plotter.show_plot()
