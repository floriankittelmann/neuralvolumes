# worth to have a look: https://github.com/NVIDIAGameWorks/kaolin

from matplotlib.colors import ListedColormap
import pyvista as pv
import matplotlib.pyplot as plt
import torch
from models.volsamplers.warpvoxel import VolSampler
import numpy as np
from models.RayMarchingHelper import RayMarchingHelper
from utils.RenderUtils import get_distributed_coords


class NeuralVolumePlotter:

    def __init__(self):
        self.plotter = pv.Plotter()

    def plot_nv_from_decout(self, decout: dict):
        nof_points = 10
        batchsize = 4
        start_coords = get_distributed_coords(batchsize=batchsize, fixed_value=-1.0, nof_points=nof_points, fixed_axis=2)
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

        pos = pos.reshape((1, 1, dimension, 1, 3))
        template_shape = template.shape
        template = template.reshape((1, template_shape[0], template_shape[1], template_shape[2], template_shape[3]))

        torch.cuda.set_device("cuda:0")
        cur_device = torch.cuda.current_device()

        pos = torch.from_numpy(pos)
        pos = pos.to(cur_device)

        template = torch.from_numpy(template)
        template = template.to(cur_device)

        volsampler = VolSampler()
        sample_rgb, sample_alpha = volsampler(pos=pos, template=template)

        sample_rgb = sample_rgb.cpu().numpy().reshape((dimension, 3))
        sample_alpha = sample_alpha.cpu().numpy().reshape((dimension, 1))
        pos = pos.cpu().numpy().reshape((dimension, 3))

        sample_rgba = np.zeros((dimension, 4))
        sample_rgba[:, 0:3] = sample_rgb
        sample_rgba[:, 3] = sample_alpha[:, 0]
        sample_rgba = sample_rgba.reshape((int(density), int(density), int(density), 4))
        return pos, sample_rgba.astype(float), dimension

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

    def pyvista_3d_from_template_np(self, np_filename: str, overwrite_color_to_black: bool = False):
        density = 128.0
        pos, sample_rgba, dimension = self.__prepare_template_np_plot(np_filename, density)
        min, max = -1.0, 1.0
        x = np.arange(min, max, (2.0 / density))
        y = np.arange(min, max, (2.0 / density))
        z = np.arange(min, max, (2.0 / density))
        x, y, z = np.meshgrid(x, y, z)

        if overwrite_color_to_black:
            sample_rgba[:, :, :, 0:3] = np.zeros(sample_rgba[:, :, :, 0:3].shape)

        sample_rgba[:, :, :, 3] = sample_rgba[:, :, :, 3] / 255.
        grid = pv.StructuredGrid(x, y, z)
        self.plotter.add_points(grid.points,
                                scalars=sample_rgba[:, :, :, 0:4].reshape((sample_rgba.shape[0] ** 3, 4)),
                                rgb=True)

    def plot_stl_pyvista(self, filenameMesh: str):
        mesh = pv.read(filenameMesh)

        density = mesh.length / 100
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        x = np.arange(x_min, x_max, density)
        y = np.arange(y_min, y_max, density)
        z = np.arange(z_min, z_max, density)
        x, y, z = np.meshgrid(x, y, z)

        # Create unstructured grid from the structured grid
        grid = pv.StructuredGrid(x, y, z)
        ugrid = pv.UnstructuredGrid(grid)

        grid = pv.StructuredGrid(x / 100., y / 100., z / 100.)
        # get part of the mesh within the mesh's bounding surface.
        selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)
        mask = selection.point_data['SelectedPoints'].view(bool)
        mask = mask.reshape(x.shape)
        self.plotter.add_points(grid.points, cmap=["#00000000", "#ff00004D"], scalars=mask)

    def show_plot(self):
        self.plotter.show_axes_all()
        self.plotter.show()


if __name__ == "__main__":
    plotter = NeuralVolumePlotter()
    filename = "test.npy"
    plotter.pyvista_3d_from_template_np(filename, True)
    path_stl = "C:\\Users\\Flori\\Desktop\\Test-Export-Blender\\test0.stl"
    #plotter.plot_stl_pyvista(path_stl)
    plotter.show_plot()
