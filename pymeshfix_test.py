# worth to have a look: https://github.com/NVIDIAGameWorks/kaolin
import copy

from pymeshfix import MeshFix
import pyvista as pv


def plot_loosing_edges(filenameMesh: str):
    mesh = pv.read(filenameMesh)
    meshfix = MeshFix(mesh)
    meshfix.plot()


def voxelize_surface_mesh(filenameMesh: str):
    mesh = pv.read(filenameMesh)
    voxels = pv.voxelize(mesh, density=0.6)
    pv.global_theme.background = "black"
    pl = pv.Plotter()
    voxel2 = copy.deepcopy(voxels)
    voxel2.points = 10 + voxel2.points
    pl.add_mesh(voxels, color="red", opacity=0.85)
    pl.add_mesh(voxel2, color="green")
    pl.show()
    # when adding opacity, it looks like inside it is hollow. But there are voxels as well.
    # can be checked by plotting the cell centers as spheres
    # voxels.cell_centers().plot(render_points_as_spheres=True)


if __name__ == "__main__":
    # examples.native()
    filename = "C:\\Users\\Flori\\Desktop\\BaseMesh_Anim.stl"
    # plot_loosing_edges(filename)
    voxelize_surface_mesh(filename)
