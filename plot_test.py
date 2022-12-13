from models.volsamplers.warpvoxel import VolSampler
from plot_neuralvolumes import NeuralVolumePlotter
import matplotlib.pyplot as plt
from models.RayMarchingHelper import RayMarchingHelper
from utils.RenderUtils import get_distributed_coords
import numpy as np
import torch


class TestPlot(NeuralVolumePlotter):


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
