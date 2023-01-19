import torch
from eval.NeuralVolumePlotter.NeuralVolumeBuilder import NeuralVolumeBuilder


class GroundTruthLoss:

    def __init__(
            self,
            decout: dict,
            pos_truth: torch.Tensor,
            volume_truth: torch.Tensor,
            resolution: int
    ):
        self.decout = decout
        self.pos_truth = pos_truth
        self.volume_truth = volume_truth
        self.nv_builder = NeuralVolumeBuilder(resolution)

    def calculate_mse_loss(self) -> torch.Tensor:
        pos_model, volume_pred = self.nv_builder.get_nv_from_model_output(
            self.decout
        )
        volume_pred = volume_pred / 255. * 2. - 1.
        volume_truth = self.volume_truth / 255. * 2. - 1.
        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn(volume_pred, volume_truth)
        return loss
