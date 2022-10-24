
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
    voxels = pv.voxelize(mesh, density=0.4)
    print(type(voxels))
    print(type(voxels.points))
    voxels.plot()


if __name__ == "__main__":
    # examples.native()
    filename = "C:\\Users\\Flori\\Desktop\\BaseMesh_Anim.stl"
    plot_loosing_edges(filename)
    #voxelize_surface_mesh(filename)
