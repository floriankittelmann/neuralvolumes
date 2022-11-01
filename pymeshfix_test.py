
#worth to have a look: https://github.com/NVIDIAGameWorks/kaolin

from pymeshfix import MeshFix
import pyvista as pv
from pyvista import PolyData


def plot_loosing_edges(filenameMesh: str):
    mesh = pv.read(filenameMesh)
    meshfix = MeshFix(mesh)
    meshfix.plot()


def voxelize_surface_mesh(filenameMesh: str):
    mesh = pv.read(filenameMesh)
    voxels = pv.voxelize(mesh, density=0.6)
    print(voxels.get_data_range())
    print(type(voxels))
    print(voxels.points.shape)
    print(voxels.points)
    print(voxels.compute_cell_sizes())
    voxels.plot()


if __name__ == "__main__":
    # examples.native()
    filename = "C:\\Users\\Flori\\Desktop\\BaseMesh_Anim.stl"
    plot_loosing_edges(filename)
    voxelize_surface_mesh(filename)
