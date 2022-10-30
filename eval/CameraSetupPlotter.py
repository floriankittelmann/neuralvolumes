from collections.abc import Iterator
from eval.CoordinateSystem import CoordinateSystem
from eval.CubePlotter import CubePlotter
from config_templates.blender2_config import get_dataset as get_dataset_blender
from config_templates.dryice1_config import get_dataset as get_dataset_dryice
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from models.RayMarchingHelper import RayMarchingHelper
from torch.utils.data import Sampler
import torch
import numpy as np
import copy

class CustomBatchSampler(Sampler[int]):

    def __init__(self, nof_cameras: int, nof_frames: int):
        self.nof_cameras = nof_cameras
        self.nof_frames = nof_frames

    def __iter__(self) -> Iterator[int]:
        indices = []
        for i in range(self.nof_cameras):
            indices.append(i * self.nof_frames)
        return iter(indices)

    def __len__(self) -> int:
        return self.nof_cameras


class CameraSetupPlotter:
    MODE_BLENDER2_DATASET = 1
    MODE_DRYICE_DATASET = 2

    def __init__(self, mode: int):
        if mode == self.MODE_BLENDER2_DATASET:
            self.ds = get_dataset_blender()
            self.arrow_length = 0.8
        elif mode == self.MODE_DRYICE_DATASET:
            self.ds = get_dataset_dryice()
            self.arrow_length = 1.0
        else:
            raise Exception("mode not known")
        self.krt = self.ds.get_krt()
        self.mode = mode
        fig = plt.figure()
        self.ax = fig.add_subplot(projection='3d')
        self.nof_cameras = len(self.ds.get_allcameras())
        self.nof_frames = self.ds.get_nof_frames()

    def __plot_location_neural_volumes(self):
        cube = CubePlotter()
        cube.draw(self.ax)

    def __plot_coordinate_system_from_cam_index(self, i):
        cam_key = list(self.krt.keys())[i]
        values_krt = self.krt[cam_key]

        if self.mode == self.MODE_DRYICE_DATASET:
            rot_krt = self.ds.get_rot_of_cam(cam_key)
            pos_krt = self.ds.get_pos_of_cam(cam_key)
        else:
            rot_krt = values_krt["rot"]
            pos_krt = values_krt["pos"]

        rot_cam = Rotation.from_matrix(rot_krt)
        cs_cam = CoordinateSystem(pos_krt[0], pos_krt[1], pos_krt[2], rot_cam, arrow_length=self.arrow_length)
        cs_cam.draw(self.ax)

    def __plot_ray_marching_positions_from_cam_index(self, i):
        datasetindex = i * self.nof_frames
        dataset_of_camera = self.ds[datasetindex]
        pixelcoords = dataset_of_camera['pixelcoords']
        princpt = dataset_of_camera['princpt']
        focal = dataset_of_camera['focal']
        camrot = dataset_of_camera['camrot']
        campos = dataset_of_camera['campos']
        pixelcoords = torch.from_numpy(pixelcoords.reshape((1, 1024, 667, 2)))
        princpt = torch.from_numpy(princpt.reshape((1, 2)))
        focal = torch.from_numpy(focal.reshape((1, 2)))
        camrot = torch.from_numpy(camrot.reshape((1, 3, 3)))
        campos = torch.from_numpy(campos.reshape((1, 3)))
        dt = 2.0
        ray_helper = RayMarchingHelper(pixelcoords, princpt, focal, camrot, campos, dt)
        for raypos in ray_helper.iterate_raypos(1):
            self.__plot_raypos(raypos)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-5, 5)
        plt.show()
        exit()

    def __plot_raypos(self, raypos: torch.Tensor):
        raypos_np = copy.deepcopy(raypos.numpy())
        img_size_1 = raypos_np.shape[1]
        img_size_2 = raypos_np.shape[2]
        reshape_size = (img_size_1, img_size_2)
        X = raypos_np[:, :, :, 0].reshape(reshape_size)
        Y = raypos_np[:, :, :, 1].reshape(reshape_size)
        Z = raypos_np[:, :, :, 2].reshape(reshape_size)
        self.ax.plot_surface(X, Y, Z, color="red")

    def plot_camera_setup(self):
        self.__plot_location_neural_volumes()
        for i in range(self.nof_cameras):
            self.__plot_coordinate_system_from_cam_index(i)
            self.__plot_ray_marching_positions_from_cam_index(i)


