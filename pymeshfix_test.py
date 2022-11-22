# worth to have a look: https://github.com/NVIDIAGameWorks/kaolin

from pymeshfix import MeshFix
import pyvista as pv
import matplotlib.pyplot as plt
import torch
from models.volsamplers.warpvoxel import VolSampler
import numpy as np
from models.RayMarchingHelper import RayMarchingHelper


def plot_loosing_edges(filenameMesh: str):
    mesh = pv.read(filenameMesh)
    meshfix = MeshFix(mesh)
    meshfix.plot()


def get_distributed_coords(z_value: float, nof_points: int) -> np.ndarray:
    list_coordinates = [(x, y, z_value)
                        for x in np.linspace(-1.0, 1.0, nof_points)
                        for y in np.linspace(-1.0, 1.0, nof_points)]
    start_coords = np.array(list_coordinates)
    return start_coords.reshape((1, nof_points, nof_points, 3))


def plot_nv_from_decout(decout: dict):
    nof_points = 10

    start_coords = get_distributed_coords(-1.0, nof_points)
    direction_coords = np.full((1, nof_points, nof_points, 3), (0.0, 0.0, 1.0))
    dt = 2.0 / float(nof_points)
    t = start_coords
    end_coords = get_distributed_coords(1.0, nof_points)
    print("----")
    print(start_coords.shape)
    print(direction_coords.shape)
    print(type(dt))
    print(t.shape)
    print(end_coords.shape)
    exit()
    raymarching = RayMarchingHelper(
        torch.from_numpy(start_coords),
        torch.from_numpy(direction_coords),
        dt,
        torch.from_numpy(t),
        torch.from_numpy(end_coords),
        RayMarchingHelper.OUTPUT_VOLUME
    )
    rgb, alpha = raymarching.do_raymarching(
        VolSampler(),
        decout,
        False,
        0.0
    )



def plot_nv_pyvista(np_filename: str):
    template = None
    with open(np_filename, 'rb') as f:
        template = np.load(f)
    if template is None:
        raise Exception("should load file")

    density = 50.0
    min = -1.0
    max = 1.0

    distribution = np.arange(min, max, (2.0 / density))
    x, y, z = np.meshgrid(distribution, distribution, distribution)
    pos = np.stack((x, y, z), axis=3)
    dimension = int(density ** 3)

    pos = pos.reshape((1, 1, dimension, 1, 3))
    template_shape = template.shape
    template = template.reshape((1, template_shape[0], template_shape[1], template_shape[2], template_shape[3]))

    torch.cuda.set_device("cuda:0")
    cur_device = torch.cuda.current_device()

    pos = torch.from_numpy(pos)
    pos = pos.to(cur_device)

    template = torch.from_numpy(template)
    template = template.to(cur_device)

    volsampler = VolSampler()
    sample_rgb, sample_alpha = volsampler(pos=pos, template=template)

    bgcolor = np.zeros((1, 3, 1, int(density ** 3), 1))  # black bg color
    bgcolor = torch.from_numpy(bgcolor).to("cuda")

    sample_rgb = sample_rgb + (1. - sample_alpha) * bgcolor
    sample_rgb = sample_rgb.cpu().numpy().reshape((dimension, 3))
    sample_alpha = sample_alpha.cpu().numpy().reshape((dimension, 1))
    pos = pos.cpu().numpy().reshape((dimension, 3))

    gamma_correction_value = (2. / 1.)
    sample_rgb = (np.clip(sample_rgb, 0., 255.) / 255.0) ** gamma_correction_value
    sample_alpha = np.clip(sample_alpha, 0., 255.) / 255.

    # print(sample_rgb.shape) # rgb values per voxel -> (dimension, 3)
    # print(sample_alpha.shape) # alpha values per voxel -> (dimension, 1)
    # print(pos.shape) # position xyz values per voxel -> (dimension, 3)

    shape_plot = (int(density) + 1, int(density) + 1, int(density) + 1)
    x, y, z = (np.indices(shape_plot) / density) * 2.0 - 1.0

    rgb_check = np.sum(sample_rgb, axis=1) > 0.0
    alpha_check = (sample_alpha > 0.0).reshape((dimension, 1))
    rgb_check = rgb_check.reshape(alpha_check.shape)
    all_check = np.logical_and(rgb_check, alpha_check)
    all_test = all_check.reshape((int(density), int(density), int(density)))

    cmap = np.zeros((dimension, 4))
    cmap[:, 0:3] = sample_rgb
    cmap[:, 3] = sample_alpha[:, 0]
    # cmap = np.array([(r, g, b, a) for r, g, b, a in cmap])
    # print(cmap.shape)
    cmap = cmap.reshape((int(density), int(density), int(density), 4))

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, all_test,
              facecolors=cmap,
              edgecolors=[0.0, 0.0, 0.0, 0.0],
              linewidth=0.0)
    plt.show()


def plot_stl_pyvista(filenameMesh: str):
    mesh = pv.read(filenameMesh)

    density = mesh.length / 100
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    print(x.shape)
    print(x[:4])
    x, y, z = np.meshgrid(x, y, z)
    print(x.shape)
    exit()

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)
    mask = selection.point_data['SelectedPoints'].view(bool)
    mask = mask.reshape(x.shape)
    pv.plot(grid.points, cmap=["#00000000", "#000000FF"], scalars=mask)


if __name__ == "__main__":
    # plot voxelized stl file with pyvista
    # filename = "C:\\Users\\Flori\\Desktop\\BaseMesh_Anim.stl"
    # plot_stl_pyvista(filename)

    plot_nv_from_decout()

    # plot neural volumes from np file
    # filename = "test.npy"
    # plot_nv_pyvista(filename)

    # plot voxel test
    # voxel_plot_test()

    # plot_sphere()
