import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt

from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder
from eval.NeuralVolumePlotter.NeuralVolumeFormatter import NeuralVolumeFormatter


class NeuralVolumePlotter:

    # in case the model still outputs a lot of voxel in the color of the background
    EXCLUDE_GRAY_POINTS_FROM_MODEL_NV = True

    # this speeds up creating of the plot, because less points are in the plot
    # transparent voxels can not be seen anyway with the our eyes
    EXCLUDE_TRANSPARENT_POINTS = True

    def __init__(self, output_path: str, resolution: int):
        self.output_path: str = output_path
        self.resolution: int = resolution

    def save_volume_and_pos(self, decout: dict, frameidx: int):
        nv_builder = NeuralVolumeBuilder(self.resolution, frameidx, NeuralVolumeBuilder.MODE_TRAIN_DATASET)
        pos, volume = nv_builder.get_nv_from_model_output(decout)
        name = "_{}_{}.npy".format(self.resolution, frameidx)

        path_volume = os.path.join(self.output_path, "volume{}".format(name))
        with open(path_volume, 'wb') as f:
            np.save(f, volume)

        path_pos = os.path.join(self.output_path, "pos{}".format(name))
        with open(path_pos, 'wb') as f:
            np.save(f, pos)

    def __plot_nv_output_model(self, frameidx: int, plot_axis, list_templates: list, list_pos: list):
        volume = list_templates[frameidx]
        positions = list_pos[frameidx]
        nof_data_points = positions.shape[1]
        volume = volume.reshape((nof_data_points, 4))
        positions = positions.reshape((nof_data_points, 3))

        formatter = NeuralVolumeFormatter()
        if self.EXCLUDE_GRAY_POINTS_FROM_MODEL_NV:
            positions, volume = formatter.exclude_gray_from_volume(positions=positions, volume=volume)

        if self.EXCLUDE_TRANSPARENT_POINTS:
            positions, volume = formatter.exclude_transparent_points(positions=positions, volume=volume)

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return plot_axis.scatter3D(x, y, z, c=volume)

    def __plot_nv_ground_truth(self, frameidx: int, ax):
        nv_builder = NeuralVolumeBuilder(self.resolution, frameidx, NeuralVolumeBuilder.MODE_TRAIN_DATASET)
        positions, volume = nv_builder.get_nv_ground_truth()

        if self.EXCLUDE_TRANSPARENT_POINTS:
            formatter = NeuralVolumeFormatter()
            positions, volume = formatter.exclude_transparent_points(positions=positions, volume=volume)

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return ax.scatter3D(x, y, z, c=volume)

    def plot_frames(self):
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
            builder = NeuralVolumeBuilder(self.resolution, 0, NeuralVolumeBuilder.MODE_TRAIN_DATASET)
            builder.calculate_mse_loss_from_cached_data(list_templates, list_positions)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            self.__plot_nv_output_model(0, ax, list_templates, list_positions)
            self.__plot_nv_ground_truth(0, ax)
            limit = [-1.0, 1.0]
            ax.set_xlim(limit)
            ax.set_ylim(limit)
            ax.set_zlim(limit)
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            artist = None
            for frameidx in range(100):
                if artist:
                    artist.remove()
                artist = self.__plot_nv_output_model(frameidx, ax, list_templates, list_positions)
                plt.pause(1.0)
