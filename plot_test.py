import matplotlib.pyplot as plt
from models.RayMarchingHelper import RayMarchingHelper
from utils.RenderUtils import get_distributed_coords
# worth to have a look: https://github.com/NVIDIAGameWorks/kaolin
import os
from typing import Callable
import pyvista as pv
import torch
from pyvista import Plotter
from models.volsamplers.warpvoxel import VolSampler
import numpy as np


class TestPlot:

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

        print(np.max(sample_rgb))
        print(np.min(sample_rgb))

        print(np.max(sample_alpha))
        print(np.min(sample_alpha))

        sample_rgba = np.zeros((batchsize, dimension, 4))
        sample_rgba[:, :, 0:3] = sample_rgb
        sample_rgba[:, :, 3] = sample_alpha[:, :, 0]
        sample_rgba = sample_rgba.reshape((batchsize, int(density), int(density), int(density), 4))
        return pos, sample_rgba.astype(float), dimension


    def pyvista_3d_from_template_np(
            self,
            outputfolder: str,
            overwrite_color_to_black: bool = False,
            add_ground_truth: bool = False,
            nof_frames: int = 500):
        plotter_test: Plotter = pv.Plotter()
        prepare_template: Callable = self.__prepare_template_np_plot
        density: float = 16.0
        add_ground_truth_to_plotter: Callable = self.plot_stl_pyvista

        def slider_callback_create_points(value):
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
            # else:
            # sample_rgba[:, :, :, :, 0:3] = sample_rgba[:, :, :, :, 0:3] / 255.

            # sample_rgba[:, :, :, :, 3] = np.ones(sample_rgba[:, :, :, :, 3].shape) * 0.5

            grid = pv.StructuredGrid(x, y, z)
            # batchsize = sample_rgba.shape[0]
            # for i in range(batchsize):
            sample_rgba = sample_rgba[0, :, :, :, 0:4].reshape((dimension, 4))
            actor = plotter_test.add_points(grid.points, scalars=sample_rgba, rgb=True)
            if add_ground_truth:
                grid, mask = add_ground_truth_to_plotter(
                    "experiments/blenderLegMovement/data/groundtruth_test/frame{:04d}.stl".format(res))
                plotter_test.add_points(grid.points, cmap=["#00000000", "#ff00004D"], scalars=mask)
            return

        numpy_files = [f for f in os.listdir(outputfolder) if
                       os.path.isfile(os.path.join(outputfolder, f)) and f.endswith(".npy")]
        plotter_test.add_slider_widget(callback=slider_callback_create_points, value=0, rng=[0, len(numpy_files)],
                                       title='Time')
        plotter_test.show_axes_all()
        plotter_test.show()

    def matplotlib_2d_from_template_np(self, np_filename: str):
        density = 50.0
        pos, sample_rgba, dimension = self.__prepare_template_np_plot(np_filename, density)
        plt.imshow(sample_rgba[:, :, int(density / 2)].reshape((int(density), int(density), 4)), vmin=0, vmax=255)
        plt.show()

    def matplotlib_3d_from_template_np(self, np_filename: str):
        density = 20.0
        pos, sample_rgba, dimension = self.__prepare_template_np_plot(np_filename, density)

        shape_plot = (int(density) + 1, int(density) + 1, int(density) + 1)
        x, y, z = (np.indices(shape_plot) / density) * 2.0 - 1.0
        all_test = np.full((int(density), int(density), int(density)), True)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(x, y, z, all_test,
                  facecolors=sample_rgba / 255.0,
                  edgecolors=[0.0, 0.0, 0.0, 0.0],
                  linewidth=0.0)
        plt.show()

    def plot_nv_from_decout(self, decout: dict):
        nof_points = 10
        batchsize = 4
        start_coords = get_distributed_coords(batchsize=batchsize, fixed_value=-1.0, nof_points=nof_points,
                                              fixed_axis=2)
        direction_coords = np.full((batchsize, nof_points, nof_points, 3), (0.0, 0.0, 1.0))
        dt = 2.0 / float(nof_points)
        t = np.ones((batchsize, nof_points, nof_points)) * -1
        end = np.ones((batchsize, nof_points, nof_points))
        raymarching = RayMarchingHelper(
            torch.from_numpy(start_coords).to("cuda"),
            torch.from_numpy(direction_coords).to("cuda"),
            dt,
            torch.from_numpy(t).to("cuda"),
            torch.from_numpy(end).to("cuda"),
            RayMarchingHelper.OUTPUT_VOLUME
        )
        rgb, alpha = raymarching.do_raymarching(
            VolSampler(),
            decout,
            False,
            0.0
        )
        print(rgb.size())
        print(alpha.size())
        print(rgb[0, 0, 0, 0, 0])
        print(alpha[0, 0, 0, 0, 0])

    def show_plot(self):
        self.plotter.show_axes_all()
        self.plotter.show()
