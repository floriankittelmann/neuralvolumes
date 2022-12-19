import math

import numpy as np
from eval.CoordinateSystem import CoordinateSystem
from eval.GlobalCoordinateSystem import GlobalCoordinateSystem
from eval.CubePlotter import CubePlotter
from config_templates.blender2_config import DatasetConfig as DatasetConfigBlender2
from config_templates.blenderLegMovement_config import DatasetConfig as DatasetConfigLegM
from config_templates.dryice1_config import get_dataset as get_dataset_dryice
from config_templates.blender2_config import Render as BlenderRender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from models.RayMarchingHelper import init_with_camera_position
import torch
import copy
from data.CameraSetups.CameraSetupInBlender2 import CameraSetupInBlender2


class CameraSetupPlotter:
    MODE_BLENDER2_DATASET = 1
    MODE_DRYICE_DATASET = 2
    MODE_ROT_RENDER = 3

    MODE_LEG_MOVEMENT_TRAIN_DATASET = 4
    MODE_LEG_MOVEMENT_TEST_DATASET = 5

    def __init__(self, mode: int):
        if mode == self.MODE_BLENDER2_DATASET:
            dataset_config = DatasetConfigBlender2()
            self.ds = dataset_config.get_dataset_config_func()
        if mode == self.MODE_BLENDER2_DATASET:
            dataset_config = DatasetConfigBlender2()
            self.ds = dataset_config.get_dataset_config_func()
        elif mode == self.MODE_LEG_MOVEMENT_TRAIN_DATASET:
            dataset_config = DatasetConfigLegM()
            self.ds = dataset_config.get_train_dataset_config_func()
        elif mode == self.MODE_LEG_MOVEMENT_TEST_DATASET:
            dataset_config = DatasetConfigLegM()
            self.ds = dataset_config.get_test_dataset_config_func()
        elif mode == self.MODE_DRYICE_DATASET:
            self.ds = get_dataset_dryice()
        elif mode == self.MODE_ROT_RENDER:
            blender = BlenderRender()
            self.ds = blender.get_dataset()
        else:
            raise Exception("mode not known")
        self.krt = self.ds.get_krt()
        self.mode = mode
        self.nof_cameras = len(self.ds.get_allcameras())
        self.nof_frames = self.ds.get_nof_frames()
        self.nof_plots = 8
        self.nof_rows = 2
        self.nof_cols = 4
        self.list_ax = []
        self.current_index_plot = 0
        self.camera_setup = CameraSetupInBlender2(1)

    def __init_plot(self):
        self.list_ax = []
        fig = plt.figure()
        for i in range(self.nof_plots):
            ax = fig.add_subplot(self.nof_rows, self.nof_cols, i + 1, projection='3d')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)
            self.list_ax.append(ax)
        self.current_index_plot = 0

    def __plot_location_neural_volumes(self):
        cube = CubePlotter()
        cube.draw(self.__get_current_ax())

    def __plot_coordinate_system_from_cam_index(self, i):
        cam_key = list(self.krt.keys())[i]
        values_krt = self.krt[cam_key]
        if self.mode == self.MODE_DRYICE_DATASET:
            rot_krt = self.ds.get_rot_of_cam(cam_key)
            pos_krt = self.ds.get_pos_of_cam(cam_key)
            rot_krt = rot_krt.T
        else:
            rot_krt = values_krt["rot"]
            pos_krt = values_krt["pos"]
        rot_cam = Rotation.from_matrix(rot_krt)
        cs_cam = CoordinateSystem(pos_krt[0], pos_krt[1], pos_krt[2], rot_cam)
        cs_cam.draw(self.__get_current_ax())

    def __plot_ray_marching_positions_from_cam_index(self, i):
        datasetindex = i
        dataset_of_camera = self.ds[datasetindex]
        pixelcoords = dataset_of_camera['pixelcoords']
        princpt = dataset_of_camera['princpt']
        focal = dataset_of_camera['focal']
        camrot = dataset_of_camera['camrot']
        campos = dataset_of_camera['campos']

        cam_key = list(self.krt.keys())[i]
        values_krt = self.krt[cam_key]
        x_size = values_krt["size"][1]
        y_size = values_krt["size"][0]
        pixelcoords = torch.from_numpy(pixelcoords.reshape((1, x_size, y_size, 2)))
        princpt = torch.from_numpy(princpt.reshape((1, 2)))
        focal = torch.from_numpy(focal.reshape((1, 2)))
        camrot = torch.from_numpy(camrot.reshape((1, 3, 3)))
        campos = torch.from_numpy(campos.reshape((1, 3)))
        dt = 0.1
        ray_helper = init_with_camera_position(pixelcoords, princpt, focal, camrot, campos, dt)
        list_points_to_plot = []
        for x in range(0, x_size, 50):
            for y in range(0, y_size, 50):
                list_points_to_plot.append({'x': x, 'y': y, 'plotX': None, 'plotY': None, 'plotZ': None})
        nof_iterations = 0
        for raypos in ray_helper.iterate_raypos():
            raypos_np = copy.deepcopy(raypos.numpy())
            for point in list_points_to_plot:
                x_point = raypos_np[0, point['x'], point['y'], 0]
                y_point = raypos_np[0, point['x'], point['y'], 1]
                z_point = raypos_np[0, point['x'], point['y'], 2]
                if point['plotX'] is None:
                    point['plotX'] = np.array([x_point])
                    point['plotY'] = np.array([y_point])
                    point['plotZ'] = np.array([z_point])
                else:
                    point['plotX'] = np.append(point['plotX'], x_point)
                    point['plotY'] = np.append(point['plotY'], y_point)
                    point['plotZ'] = np.append(point['plotZ'], z_point)
            nof_iterations = nof_iterations + 1

        for point in list_points_to_plot:
            ax = self.__get_current_ax()
            ax.plot(point['plotX'], point['plotY'], point['plotZ'], color="red")

    def __get_current_ax(self):
        return self.list_ax[self.current_index_plot]

    def __plot_coordinate_system_rotrender(self, idx: int):
        campos, camrot = self.camera_setup.get_render_rot(idx)
        rot_cam = Rotation.from_matrix(camrot)
        cs_cam = CoordinateSystem(campos[0], campos[1], campos[2], rot_cam)
        cs_cam.draw(self.__get_current_ax())

    def plot_rotrender(self):
        self.__init_plot()
        self.current_index_plot = 0
        cs = GlobalCoordinateSystem()
        for idx in range(8):
            print("Create Plot {}".format(idx))
            self.__plot_location_neural_volumes()
            self.__plot_coordinate_system_rotrender(idx)
            self.__plot_ray_marching_positions_from_cam_index(idx)
            ax = self.__get_current_ax()
            cs.draw(ax)
            title = "Position {}".format(idx)
            ax.title.set_text(title)
            self.current_index_plot = self.current_index_plot + 1
            if self.current_index_plot >= self.nof_plots:
                plt.show()
                self.__init_plot()
        plt.show()

    def plot_camera_setup(self):
        self.__init_plot()
        self.current_index_plot = 0
        for i in range(self.nof_cameras):
            print("Create Plot for Camera {}".format(i))
            self.__plot_location_neural_volumes()
            self.__plot_coordinate_system_from_cam_index(i)
            self.__plot_ray_marching_positions_from_cam_index(i)
            cam_key = list(self.krt.keys())[i]
            ax = self.__get_current_ax()
            title = "Kamera: {} - {} von {}".format(cam_key, i + 1, self.nof_cameras)
            ax.title.set_text(title)
            self.current_index_plot = self.current_index_plot + 1
            if self.current_index_plot >= self.nof_plots:
                plt.show()
                self.__init_plot()
        plt.show()
