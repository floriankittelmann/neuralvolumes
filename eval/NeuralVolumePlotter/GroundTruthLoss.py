import numpy as np
import torch

def calculate_mse_loss(
        pos_model: np.ndarray,
        nv_model: np.ndarray,
        pos_truth: np.ndarray,
        nv_truth: np.ndarray,
        resolution: int):
    batchsize = pos_model.size()[0]

    nv_model = nv_model.reshape((batchsize, resolution ** 3, 4))
    pos_model = pos_model.reshape((batchsize, resolution ** 3, 3))

    nv_truth = nv_truth.reshape((batchsize, resolution ** 3, 4))
    pos_truth = pos_truth.reshape((batchsize, resolution ** 3, 3))

    pos_model, nv_model = sort_nv(pos_model, nv_model)
    pos_truth, nv_truth = sort_nv(pos_truth, nv_truth)

    loss = np.mean((nv_truth * 255. - nv_model * 255.) ** 2)
    print(loss)
    return loss


def sort_nv(pos: np.ndarray, nv: np.ndarray):
    ind = np.lexsort((pos[:, 0], pos[:, 1], pos[:, 2]))
    return pos[ind], nv[ind]


def calculate_mse_loss_from_decout(decout: dict, gt_from_dataloader: torch.Tensor):
    pos_model, nv_model = get_nv_from_model_output(decout)
    ground_truth = gt_from_dataloader.cpu().numpy()
    pos_ground_truth = ground_truth[:, 0:3, :, :, :, :, :]
    nv_ground_truth = ground_truth[:, 3:7, :, :, :, :, :, :, :]
    return calculate_mse_loss(pos_model, nv_model, pos_ground_truth, nv_ground_truth)


def calculate_mse_loss_from_cached_data(
        self,
        list_templates: list,
        list_pos: list,
        gt_path: str):
    pos_model: np.ndarray = list_pos[self.frameidx]
    nv_model: np.ndarray = list_templates[self.frameidx]
    pos_ground_truth, nv_ground_truth = get_nv_ground_truth(gt_path, self.resolution)
    return calculate_mse_loss(pos_model, nv_model, pos_ground_truth, nv_ground_truth)
