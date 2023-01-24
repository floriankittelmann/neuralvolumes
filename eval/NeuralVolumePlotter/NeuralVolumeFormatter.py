import colorsys
from typing import Callable
import numpy as np

THRESHOLD_GRAY_VALUES = 40.

def exclude_by_hsv(row: np.ndarray) -> np.ndarray:
    (r, g, b, a, _) = row
    (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
    if s * 255. < THRESHOLD_GRAY_VALUES or v * 255. < THRESHOLD_GRAY_VALUES:
        row[4] = False
    return row


def exclude_by_alpha(row: np.ndarray) -> np.ndarray:
    (r, g, b, a, _) = row
    if a == 0.:
        row[4] = False
    return row


class NeuralVolumeFormatter:

    def __get_filtered_pos_volume(
            self,
            positions: np.ndarray,
            volume: np.ndarray,
            filter_func: Callable):
        column_to_add = np.full((volume.shape[0], 1), True)
        temp_data = np.concatenate((volume, column_to_add), axis=1)
        temp_data = np.apply_along_axis(filter_func, 1, temp_data)
        print(np.count_nonzero(temp_data[:, 4]))
        mask = temp_data[:, 4].astype(bool)
        return positions[mask, :], volume[mask, :]

    def exclude_gray_from_volume(self, positions: np.ndarray, volume: np.ndarray):
        return self.__get_filtered_pos_volume(positions, volume, exclude_by_hsv)

    def exclude_transparent_points(self, positions: np.ndarray, volume: np.ndarray):
        return self.__get_filtered_pos_volume(positions, volume, exclude_by_alpha)
