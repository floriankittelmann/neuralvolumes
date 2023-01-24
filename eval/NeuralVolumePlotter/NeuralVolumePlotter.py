import numpy as np
import torch
from matplotlib import pyplot as plt

from eval.NeuralVolumePlotter.GroundTruthLoss import GroundTruthLoss
from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder
from eval.NeuralVolumePlotter.NeuralVolumeFormatter import NeuralVolumeFormatter


class NeuralVolumePlotter:

    # in case the model still outputs a lot of voxel in the color of the background
    __EXCLUDE_GRAY_POINTS_FROM_MODEL_NV = False

    # this speeds up creating of the plot, because less points are in the plot
    # transparent voxels can not be seen anyway with the our eyes
    __EXCLUDE_TRANSPARENT_POINTS = True

    def __init__(self, resolution: int):
        self.resolution: int = resolution

    def __plot_nv_output_model(self, decout: dict, ax):
        nv_builder = NeuralVolumeBuilder(self.resolution)
        positions, volume = nv_builder.get_nv_from_model_output(decout)

        positions: np.ndarray = positions.cpu().detach().numpy()
        volume: np.ndarray = volume.cpu().detach().numpy()

        volume[:, :, 0:3] = volume[:, :, 0:3] / 255.
        volume = volume.clip(min=0., max=1.)

        positions, volume = self.__prepare_pos_nv_np_arrays_for_plot(positions, volume)

        formatter = NeuralVolumeFormatter()
        if self.__EXCLUDE_GRAY_POINTS_FROM_MODEL_NV:
            positions, volume = formatter.exclude_gray_from_volume(positions=positions, volume=volume)

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return ax.scatter3D(x, y, z, c=volume)

    def __prepare_pos_nv_np_arrays_for_plot(self, positions: np.ndarray, volume: np.ndarray):
        batchsize = positions.shape[0]
        if batchsize != 1:
            raise Exception("when plotting a NeuralVolume the batch size should be 1. Can just plot one NeuralVolume "
                            "at once")
        nof_data_points = positions.shape[1]
        volume = volume.reshape((nof_data_points, 4))
        positions = positions.reshape((nof_data_points, 3))
        if self.__EXCLUDE_TRANSPARENT_POINTS:
            formatter = NeuralVolumeFormatter()
            positions, volume = formatter.exclude_transparent_points(positions=positions, volume=volume)
        return positions, volume

    def __plot_nv_ground_truth(self, positions: np.ndarray, volume: np.ndarray, frame: int, ax):
        nv_builder = NeuralVolumeBuilder(self.resolution)
        frame_path = "experiments/blenderLegMovement/data/groundtruth_test/frame{:04d}.stl".format(frame)
        positions, volume = nv_builder.get_nv_ground_truth(frame_path)
        volume = volume.reshape((1, volume.shape[0], volume.shape[1]))
        positions = positions.reshape((1, positions.shape[0], positions.shape[1]))

        volume[:, :, 0:3] = volume[:, :, 0:3] / 255.
        positions, volume = self.__prepare_pos_nv_np_arrays_for_plot(positions, volume)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return ax.scatter3D(x, y, z, c=volume)

    def plot_one_frame(self, decout: dict, input: dict):
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        self.__plot_nv_output_model(decout, ax)
        limit = [-1.0, 1.0]
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.set_zlim(limit)

        ax = fig.add_subplot(122, projection='3d')
        ground_positions = input["gt_positions"]
        ground_volume = input["gt_volume"]
        frame: torch.Tensor = input["frame"]
        if frame.shape[0] != 1:
            raise Exception("when plotting a NeuralVolume the batch size should be 1. Can just plot one NeuralVolume "
                            "at once")
        frame: int = frame.item()
        self.__plot_nv_ground_truth(ground_positions, ground_volume, frame, ax)
        limit = [-1.0, 1.0]
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.set_zlim(limit)
        plt.show()


