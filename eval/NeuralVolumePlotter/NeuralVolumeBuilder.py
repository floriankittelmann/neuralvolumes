import pyvista as pv
import numpy as np
import torch

from models.volsamplers.warpvoxel import VolSampler


class NeuralVolumeBuilder:

    MODE_TRAIN_DATASET = 1
    MODE_TEST_DATASET = 2

    def __init__(self, resolution: int, frameidx: int, train_test_mode: int):
        self.frameidx: int = frameidx
        self.mode: int = train_test_mode
        self.resolution: int = resolution

    def __get_meshgrid_uniform_positions(self):
        min: float = -1.0
        max: float = 1.0
        distribution: np.ndarray = np.arange(min, max, (2.0 / self.resolution))
        return np.meshgrid(distribution, distribution, distribution)

    def __get_uniform_positions_torch(self, decout: dict) -> torch.Tensor:
        template: torch.Tensor = decout['template']
        batchsize = template.size()[0]
        x, y, z = self.__get_meshgrid_uniform_positions()
        pos = np.stack((x, y, z), axis=3)
        dimension = int(self.resolution ** 3)
        pos = np.array([pos for i in range(batchsize)])
        pos = pos.reshape((batchsize, 1, dimension, 1, 3))
        torch.cuda.set_device("cuda:0")
        cur_device = torch.cuda.current_device()
        pos = torch.from_numpy(pos)
        return pos.to(cur_device)

    def get_nv_from_model_output(self, decout: dict):
        """
            returns positions (x,y,z) and neuralvolumes (r,g,b,a)
            pos: batchsize,x,y,z coordinates in m
            nv: r,g,b,a in the scale from 0-1
        """
        pos: torch.Tensor = self.__get_uniform_positions_torch(decout)
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
        sample_rgba[:, :, 0:3] = sample_rgb / 255.
        sample_rgba[:, :, 3] = sample_alpha
        sample_rgba = sample_rgba.clip(min=0., max=1.)
        pos: np.ndarray = pos.reshape((batchsize, nof_data_points, 3))
        return pos, sample_rgba

    def get_nv_ground_truth(self):
        """
            returns positions (x,y,z) and neuralvolumes (r,g,b,a)
            pos: x,y,z coordinates in m
            nv: r,g,b,a in the scale from 0-1
        """
        if self.mode == self.MODE_TRAIN_DATASET:
            caption = "train"
        elif self.mode == self.MODE_TEST_DATASET:
            caption = "test"
        else:
            raise Exception("wrong mode provided")
        path = "experiments/blenderLegMovement/data/groundtruth_{}/frame{:04d}.stl".format(
            caption,
            self.frameidx)
        mesh = pv.read(path)

        x, y, z = self.__get_meshgrid_uniform_positions()

        # Create unstructured grid from the structured grid
        grid = pv.StructuredGrid(x * 100., y * 100., z * 100.)
        ugrid = pv.UnstructuredGrid(grid)

        # get part of the mesh within the mesh's bounding surface.
        selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)

        grid = pv.StructuredGrid(x, y, z)
        mask = selection.point_data['SelectedPoints'].view(bool).reshape((self.resolution ** 3))

        alpha = np.zeros((mask.shape[0], 1))
        alpha[mask, 0] = 1.

        r_value = np.mean(np.array([152., 109., 196.])) / 255.
        g_value = np.mean(np.array([106., 73., 150.])) / 255.
        b_value = np.mean(np.array([70., 49., 114.])) / 255.

        colors = np.ones((mask.shape[0], 3))
        colors[mask, 0] = r_value
        colors[mask, 1] = g_value
        colors[mask, 2] = b_value

        return grid.points, np.concatenate((colors, alpha), axis=1)

    def __calculate_mse_loss(
            self,
            pos_model: np.ndarray,
            nv_model: np.ndarray,
            pos_truth: np.ndarray,
            nv_truth: np.ndarray
    ):
        batchsize = nv_model.shape[0]
        nv_model = nv_model.reshape((batchsize * self.resolution ** 3, 4))
        pos_model = pos_model.reshape((batchsize * self.resolution ** 3, 3))

        pos_model, nv_model = self.__sort_nv(pos_model, nv_model)
        pos_truth, nv_truth = self.__sort_nv(pos_truth, nv_truth)

        #pos_nv_model = np.concatenate((pos_model, nv_model), axis=1)
        #pos_nv_ground_truth = np.concatenate((pos_truth, nv_truth), axis=1)
        loss = np.mean((nv_truth * 255. - nv_model * 255.)**2)
        print(loss)
        return loss

    def __sort_nv(self, pos: np.ndarray, nv: np.ndarray):
        ind = np.lexsort((pos[:, 0], pos[:, 1], pos[:, 2]))
        return pos[ind], nv[ind]

    def calculate_mse_loss_from_decout(self, decout: dict, frameidx: torch.Tensor):
        pos_model, nv_model = self.get_nv_from_model_output(decout)
        pos_ground_truth, nv_ground_truth = self.get_nv_ground_truth()
        return self.__calculate_mse_loss(pos_model, nv_model, pos_ground_truth, nv_ground_truth)

    def calculate_mse_loss_from_cached_data(self, list_templates: list, list_pos: list):
        pos_model: np.ndarray = list_pos[self.frameidx]
        nv_model: np.ndarray = list_templates[self.frameidx]
        pos_ground_truth, nv_ground_truth = self.get_nv_ground_truth()
        return self.__calculate_mse_loss(pos_model, nv_model, pos_ground_truth, nv_ground_truth)




