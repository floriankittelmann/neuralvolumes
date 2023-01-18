import numpy as np
import torch
from models.volsamplers.warpvoxel import VolSampler
import pyvista as pv


class NeuralVolumeBuilder:

    def __init__(self, resolution: int):
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
        cur_device = torch.cuda.current_device()
        return torch.from_numpy(pos).to(cur_device)

    def get_nv_from_model_output(self, decout: dict):
        pos: torch.Tensor = self.__get_uniform_positions_torch(decout)
        volsampler: VolSampler = VolSampler()
        sample_rgb, sample_alpha = volsampler(pos=pos, **decout)

        shape = sample_rgb.size()
        nof_data_points = shape[3]
        batchsize = shape[0]

        sample_rgb: torch.Tensor = torch.reshape(sample_rgb, (batchsize, nof_data_points, 3))
        sample_alpha: torch.Tensor = torch.reshape(sample_alpha, (batchsize, nof_data_points, 1))
        pos: torch.Tensor = torch.reshape(pos, (batchsize, nof_data_points, 3))
        volume = torch.cat([sample_rgb, sample_alpha], dim=2)
        return pos, volume

    def get_nv_ground_truth(self, gt_path: str):
        """
            returns two np-array's:
            pos: x,y,z coordinates                  -> shape: (nofPoints, 3)
            nv: neuralvolumes rgba with scale 0-1   -> shape: (nofPoints, 4)
        """
        mesh = pv.read(gt_path)
        x, z, y = self.__get_meshgrid_uniform_positions()

        # Create unstructured grid from the structured grid
        grid = pv.StructuredGrid(x * 100., y * 100., z * 100.)
        ugrid = pv.UnstructuredGrid(grid)

        # get part of the mesh within the mesh's bounding surface.
        selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)

        nof_points = self.resolution ** 3
        grid = pv.StructuredGrid(x, y, z)
        mask = selection.point_data['SelectedPoints'].view(bool).reshape(nof_points)

        alpha = np.zeros((mask.shape[0], 1))
        alpha[mask, 0] = 1.

        r_value = np.mean(np.array([152., 109., 196.]))
        g_value = np.mean(np.array([106., 73., 150.]))
        b_value = np.mean(np.array([70., 49., 114.]))

        colors = np.zeros((mask.shape[0], 3))
        colors[mask, 0] = r_value
        colors[mask, 1] = g_value
        colors[mask, 2] = b_value

        positions = np.zeros(grid.points.shape)
        positions[0:nof_points] = grid.points[0:nof_points, :]
        volume = np.concatenate((colors, alpha), axis=1).astype(np.float32)
        return positions, volume
