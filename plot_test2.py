import numpy as np
import matplotlib.pyplot as plt
import colorsys


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


def set_background_values_transparent(a: np.ndarray) -> np.ndarray:
    lowerBound = 40. / 255.
    upperBound = 80. / 255.
    if lowerBound <= a[0] <= upperBound and lowerBound <= a[1] <= upperBound and lowerBound <= a[2] <= upperBound:
        a[3] = 0.1
    return a

def exclude_by_hsv(row: np.ndarray) -> np.ndarray:
    (r, g, b, _, _) = row
    (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
    if s * 255. < 70. or v * 255. < 70.:
        row[4] = False
    return row

if __name__ == "__main__":
    should_plot_hist = False
    set_background_transparent = False
    exclude_gray_colors = True

    with open("volume_cam.npy", 'rb') as f:
        volume = np.load(f)

    with open("position_cam.npy", 'rb') as f:
        positions = np.load(f)

    nof_data_points = positions.shape[1]
    volume = volume.reshape((nof_data_points, 4))
    positions = positions.reshape((nof_data_points, 3))

    volume[:, 0:3] = volume[:, 0:3] / 255.
    volume = volume.clip(min=0., max=1.)

    print(volume.shape)
    if exclude_gray_colors:
        mask = np.full((volume.shape[0], 1), True)
        temp_data = np.concatenate((volume, mask), axis=1)
        temp_data = np.apply_along_axis(exclude_by_hsv, 1, temp_data)
        mask = temp_data[:, 4].astype(bool)
        volume = volume[mask, :]
        positions = positions[mask, :]

    print(volume.shape)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    if should_plot_hist:
        plot_hists(volume)

    if set_background_transparent:
        volume = np.apply_along_axis(set_background_values_transparent, 1, volume)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(x, y, z, c=volume)
    plt.show()
