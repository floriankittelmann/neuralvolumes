import numpy as np
from eval.CoordinateSystem import CoordinateSystem
from eval.CubePlotter import CubePlotter
from config_templates.blender2_config import get_dataset as get_dataset_blender
from config_templates.dryice1_config import get_dataset as get_dataset_dryice
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from models.RayMarchingHelper import RayMarchingHelper
import torch
import copy


class CameraSetupPlotter:
    MODE_BLENDER2_DATASET = 1
    MODE_DRYICE_DATASET = 2

    def __init__(self, mode: int):
        if mode == self.MODE_BLENDER2_DATASET:
            self.ds = get_dataset_blender()
        elif mode == self.MODE_DRYICE_DATASET:
            self.ds = get_dataset_dryice()
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

    def __init_plot(self):
        self.list_ax = []
        fig = plt.figure()
        for i in range(self.nof_plots):
            ax = fig.add_subplot(self.nof_rows, self.nof_cols, i+1, projection='3d')
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
        pixelcoords = torch.from_numpy(pixelcoords.reshape((1, 1024, 667, 2)))
        princpt = torch.from_numpy(princpt.reshape((1, 2)))
        focal = torch.from_numpy(focal.reshape((1, 2)))
        camrot = torch.from_numpy(camrot.reshape((1, 3, 3)))
        campos = torch.from_numpy(campos.reshape((1, 3)))
        dt = 0.1
        ray_helper = RayMarchingHelper(pixelcoords, princpt, focal, camrot, campos, dt)
        list_points_to_plot = [
            {'x': 0, 'y': 0, 'plotX': None, 'plotY': None, 'plotZ': None},
            {'x': 1023, 'y': 0, 'plotX': None, 'plotY': None, 'plotZ': None},
            {'x': 0, 'y': 666, 'plotX': None, 'plotY': None, 'plotZ': None},
            {'x': 1023, 'y': 666, 'plotX': None, 'plotY': None, 'plotZ': None},
            {'x': int(1024 / 2), 'y': int(666 / 2), 'plotX': None, 'plotY': None, 'plotZ': None}
        ]
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
