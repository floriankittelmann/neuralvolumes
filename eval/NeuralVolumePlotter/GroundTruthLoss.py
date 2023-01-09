import numpy as np
import torch

from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder


def sort_nv(each_batch: np.ndarray):
    """
        used to sort each batch individual
    """
    ind = np.lexsort((each_batch[:, 0], each_batch[:, 1], each_batch[:, 2]))
    return each_batch[ind]


class GroundTruthLoss:

    def __init__(self, decout: dict, pos_truth: torch.Tensor, volume_truth: torch.Tensor, resolution: int):
        self.decout = decout
        self.pos_truth = pos_truth
        self.volume_truth = volume_truth
        self.nv_builder = NeuralVolumeBuilder(resolution)

    def calculate_mse_loss(self):
        pos_model, volume_model = self.nv_builder.get_nv_from_model_output(self.decout)
        pos_truth = self.pos_truth.cpu().numpy()
        volume_truth = self.volume_truth.cpu().numpy()

        pos_nv_model = np.concatenate((pos_model, volume_model), axis=2)
        pos_nv_truth = np.concatenate((pos_truth, volume_truth), axis=2)

        #sort array x,y,z coordinates
        pos_nv_model = np.array(list(map(sort_nv, pos_nv_model)))
        pos_nv_truth = np.array(list(map(sort_nv, pos_nv_truth)))

        volume_model_sorted = pos_nv_model[:, :, 3:7]
        pos_truth_sorted = pos_nv_truth[:, :, 3:7]
        loss = np.mean((pos_truth_sorted - volume_model_sorted)**2)
        return loss
