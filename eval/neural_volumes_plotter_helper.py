import colorsys
import numpy as np
from matplotlib import pyplot as plt


def set_background_values_transparent(a: np.ndarray) -> np.ndarray:
    lower_bound = 40. / 255.
    upper_bound = 80. / 255.
    if lower_bound <= a[0] <= upper_bound and lower_bound <= a[1] <= upper_bound and lower_bound <= a[2] <= upper_bound:
        a[3] = 0.1
    return a


def exclude_by_hsv(row: np.ndarray) -> np.ndarray:
    (r, g, b, a, _) = row
    (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
    if s * 255. < 25. or v * 255. < 25.:
        row[4] = False
    return row


def plot_hists(volume: np.ndarray):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    ax1.hist(volume[:, 0], bins='auto')
    ax1.set_title("Red values")

    ax2.hist(volume[:, 1], bins='auto')
    ax2.set_title("green values")

    ax3.hist(volume[:, 2], bins='auto')
    ax3.set_title("blue values")

    ax4.hist(volume[:, 3], bins='auto')
    ax4.set_title("alpha values")
    plt.show()



