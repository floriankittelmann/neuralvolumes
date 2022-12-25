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

        print(np.max(sample_rgb))
        print(np.min(sample_rgb))

        print(np.max(sample_alpha))
        print(np.min(sample_alpha))

        sample_rgba = np.zeros((batchsize, dimension, 4))
        sample_rgba[:, :, 0:3] = sample_rgb
        sample_rgba[:, :, 3] = sample_alpha[:, :, 0]
        sample_rgba = sample_rgba.reshape((batchsize, int(density), int(density), int(density), 4))
        return pos, sample_rgba.astype(float), dimension

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
        return grid, mask

    def pyvista_3d_from_template_np(
            self,
            outputfolder: str,
            overwrite_color_to_black: bool = False,
            add_ground_truth: bool = False,
            nof_frames: int = 500):
        plotter_test: Plotter = pv.Plotter()
        prepare_template: Callable = self.__prepare_template_np_plot
        density: float = 32.0
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
            else:
                sample_rgba[:, :, :, :, 0:3] = sample_rgba[:, :, :, :, 0:3] / 255.
                gamma_correction_value = (2. / 1.)
                sample_rgba[:, :, :, :, 0:3] = sample_rgba[:, :, :, :, 0:3] ** gamma_correction_value

            #sample_rgba[:, :, :, :, 3] = np.ones(sample_rgba[:, :, :, :, 3].shape) * 0.5

            grid = pv.StructuredGrid(x, y, z)
            # batchsize = sample_rgba.shape[0]
            # for i in range(batchsize):
            sample_rgba = sample_rgba[0, :, :, :, 0:4].reshape((dimension, 4))
            actor = plotter_test.add_points(grid.points, scalars=sample_rgba, rgb=True)
            if add_ground_truth:
                grid, mask = add_ground_truth_to_plotter("experiments/blenderLegMovement/data/groundtruth_test/frame{:04d}.stl".format(res))
                plotter_test.add_points(grid.points, cmap=["#00000000", "#ff00004D"], scalars=mask)
            return

        numpy_files = [f for f in os.listdir(outputfolder) if os.path.isfile(os.path.join(outputfolder, f)) and f.endswith(".npy")]
        plotter_test.add_slider_widget(callback=slider_callback_create_points, value=0, rng=[0, len(numpy_files)], title='Time')
        plotter_test.show_axes_all()
        plotter_test.show()


if __name__ == "__main__":
    plotter = NeuralVolumePlotter()
    filename = "experiments/blenderLegMovement/20221224_144646_blenderLegNormLoss/templates"
    plotter.pyvista_3d_from_template_np(filename,
        overwrite_color_to_black=False,
        add_ground_truth=False)

    #path_stl = "C:\\Users\\Flori\\Desktop\\Test-Export-Blender\\test0.stl"
    #plotter.plot_stl_pyvista(path_stl)
    #plotter.show_plot()
