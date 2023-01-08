import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
import pyvista as pv
from models.volsamplers.warpvoxel import VolSampler
from models.RayMarchingHelper import init_with_camera_position, RayMarchingHelper
from eval.neural_volumes_plotter_helper import set_background_values_transparent, exclude_by_hsv, plot_hists


class NeuralVolumePlotter:

    def __init__(self, output_path: str):
        self.resolution: int = 64
        self.output_path: str = output_path
        self.should_plot_hist: bool = False
        self.set_background_transparent: bool = False
        self.exclude_gray_colors: bool = True

    def get_uniform_positions(self, decout: dict) -> torch.Tensor:
        template: torch.Tensor = decout['template']
        batchsize = template.size()[0]
        density: float = float(self.resolution)
        min: float = -1.0
        max: float = 1.0
        distribution: np.ndarray = np.arange(min, max, (2.0 / density))
        x, y, z = np.meshgrid(distribution, distribution, distribution)
        pos = np.stack((x, y, z), axis=3)
        dimension = int(density ** 3)
        pos = np.array([pos for i in range(batchsize)])
        pos = pos.reshape((batchsize, 1, dimension, 1, 3))
        torch.cuda.set_device("cuda:0")
        cur_device = torch.cuda.current_device()
        pos = torch.from_numpy(pos)
        return pos.to(cur_device)

    def get_dist_volume(
            self,
            pos: torch.Tensor,
            decout: dict
    ):
        volsampler: VolSampler = VolSampler()
        sample_rgb, sample_alpha = volsampler(pos=pos, **decout)
        sample_rgb: np.ndarray = sample_rgb.cpu().numpy()
        sample_alpha: np.ndarray = sample_alpha.cpu().numpy()
        pos: np.ndarray = pos.cpu().numpy()
        shape: tuple = sample_rgb.shape
        nof_data_points = shape[3]

        batchsize = shape[0]
        sample_rgba: np.ndarray = np.zeros((batchsize, nof_data_points, 4))

        sample_rgb = sample_rgb.reshape((batchsize, nof_data_points, 3))
        sample_alpha = sample_alpha.reshape((batchsize, nof_data_points))
        sample_rgba[:, :, 0:3] = sample_rgb
        sample_rgba[:, :, 3] = sample_alpha
        pos = pos.reshape((batchsize, nof_data_points, 3))
        return pos, sample_rgba

    def save_uniform_dist_volume(self, decout: dict, frameidx: int):
        pos: torch.Tensor = self.get_uniform_positions(decout)
        self.save_volume_and_pos(
            pos=pos,
            decout=decout,
            frameidx=frameidx)

    def save_volume_from_camera(
            self,
            data: dict,
            decout: dict,
            dt: float,
            frameidx: int
    ):
        pixelcoords = data['pixelcoords'].to('cuda')
        princpt = data['princpt'].to('cuda')
        camrot = data['camrot'].to('cuda')
        focal = data['focal'].to('cuda')
        campos = data['campos'].to('cuda')
        raymarchHelper: RayMarchingHelper = init_with_camera_position(
            pixelcoords=pixelcoords,
            princpt=princpt,
            camrot=camrot,
            focal=focal,
            campos=campos,
            dt=dt)
        raypos_appended = None
        for raypos in raymarchHelper.iterate_raypos():
            raypos = raypos.cpu().numpy()
            shape = raypos.shape
            batchsize = shape[0]
            dimension = shape[1] * shape[2]
            raypos = raypos.reshape((batchsize, 1, dimension, 1, 3))
            if raypos_appended is None:
                raypos_appended = raypos
            else:
                raypos_appended = np.append(raypos_appended, raypos, axis=2)
        pos: torch.Tensor = torch.from_numpy(raypos_appended)
        self.save_volume_and_pos(pos, decout, frameidx)

    def save_volume_and_pos(
            self,
            pos: torch.Tensor,
            decout: dict,
            frameidx: int
    ):
        pos, sample_rgba = self.get_dist_volume(pos=pos, decout=decout)
        name = "_{}_{}.npy".format(self.resolution, frameidx)

        path_volume = os.path.join(self.output_path, "volume{}".format(name))
        with open(path_volume, 'wb') as f:
            np.save(f, sample_rgba)

        path_pos = os.path.join(self.output_path, "pos{}".format(name))
        with open(path_pos, 'wb') as f:
            np.save(f, pos)

    def __plot_one_frame(self, idx, plot_axis, list_templates: list, list_pos: list):
        volume = list_templates[idx]
        positions = list_pos[idx]
        nof_data_points = positions.shape[1]
        volume = volume.reshape((nof_data_points, 4))
        positions = positions.reshape((nof_data_points, 3))

        volume[:, 0:3] = volume[:, 0:3] / 255.
        volume = volume.clip(min=0., max=1.)

        print(volume.shape)
        if self.exclude_gray_colors:
            mask = np.full((volume.shape[0], 1), True)
            temp_data = np.concatenate((volume, mask), axis=1)
            temp_data = np.apply_along_axis(exclude_by_hsv, 1, temp_data)
            mask = temp_data[:, 4].astype(bool)
            volume = volume[mask, :]
            positions = positions[mask, :]
        print(volume.shape)

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        if self.should_plot_hist:
            plot_hists(volume)

        if self.set_background_transparent:
            volume = np.apply_along_axis(set_background_values_transparent, 1, volume)

        return plot_axis.scatter3D(x, y, z, c=volume)

    def plot_stl_pyvista(self, filenameMesh: str):
        mesh = pv.read(filenameMesh)

        min, max = -100, 100
        density = 200. / float(self.resolution)
        x = np.arange(min, max, density)
        y = np.arange(min, max, density)
        z = np.arange(min, max, density)
        x, y, z = np.meshgrid(x, y, z)

        # Create unstructured grid from the structured grid
        grid = pv.StructuredGrid(x, y, z)
        ugrid = pv.UnstructuredGrid(grid)

        # get part of the mesh within the mesh's bounding surface.
        selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)

        grid = pv.StructuredGrid(x / 100., y / 100., z / 100.)
        mask = selection.point_data['SelectedPoints'].view(bool).reshape((self.resolution**3))
        print(mask.shape)

        alpha = np.zeros((mask.shape[0], 1))
        alpha[mask, 0] = 1.

        colors = np.ones((mask.shape[0], 3))
        colors[mask, 1] = 0.
        colors[mask, 2] = 0.

        print(colors.shape)
        print(alpha.shape)

        volume = np.concatenate((colors, alpha), axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        x = grid.points[:, 0]
        y = grid.points[:, 1]
        z = grid.points[:, 2]
        ax.scatter3D(x, y, z, c=volume)
        plt.show()

    def plot_frames(self):
        stl_plot = False
        if stl_plot:
            path = "experiments/blenderLegMovement/data/groundtruth_train/frame0000.stl"
            self.plot_stl_pyvista(path)
        else:
            list_templates = []
            list_positions = []
            print("load templates")
            for idx in range(99):
                name = "_{}_{}.npy".format(self.resolution, idx)
                path_volume = os.path.join(self.output_path, "volume{}".format(name))
                with open(path_volume, 'rb') as f:
                    volume = np.load(f)
                path_pos = os.path.join(self.output_path, "pos{}".format(name))
                with open(path_pos, 'rb') as f:
                    positions = np.load(f)
                list_templates.append(volume)
                list_positions.append(positions)
            print("finish load templates")

            is_implemented_animation = False
            if not is_implemented_animation:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                artist = self.__plot_one_frame(0, ax, list_templates, list_positions)
                plt.show()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                artist = None
                for frameidx in range(100):
                    if artist:
                        artist.remove()
                    artist = self.__plot_one_frame(frameidx, ax, list_templates, list_positions)
                    plt.pause(1.0)
