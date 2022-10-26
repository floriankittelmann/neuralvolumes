from config_templates.blender2_config import get_dataset as get_dataset_blender
from config_templates.dryice1_config import get_dataset as get_dataset_dryice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation


# from https://github.com/matplotlib/matplotlib/issues/21688
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class CoordinateSystem:
    def __init__(
            self,
            pos_x: float,
            pos_y: float,
            pos_z: float,
            rot: Rotation,
            arrow_length: float = 0.4,
            linewidth: float = 1.5
    ):
        self.line_width = linewidth

        x_end_point = np.array([arrow_length, 0.0, 0.0])
        y_end_point = np.array([0.0, arrow_length, 0.0])
        z_end_point = np.array([0.0, 0.0, arrow_length])
        rot_x = rot.apply(x_end_point)
        rot_y = rot.apply(y_end_point)
        rot_z = rot.apply(z_end_point)
        self.norm_x = np.array([
            [pos_x, pos_x + rot_x[0]],
            [pos_y, pos_y + rot_x[1]],
            [pos_z, pos_z + rot_x[2]]
        ])
        self.norm_y = np.array([
            [pos_x, pos_x + rot_y[0]],
            [pos_y, pos_y + rot_y[1]],
            [pos_z, pos_z + rot_y[2]]
        ])
        self.norm_z = np.array([
            [pos_x, pos_x + rot_z[0]],
            [pos_y, pos_y + rot_z[1]],
            [pos_z, pos_z + rot_z[2]]
        ])

    def __draw_arrow(self, mode: str, axis_local):
        if mode == "x":
            color = "r"
            xs = self.norm_x[0]
            ys = self.norm_x[1]
            zs = self.norm_x[2]
        elif mode == "y":
            color = "g"
            xs = self.norm_y[0]
            ys = self.norm_y[1]
            zs = self.norm_y[2]
        elif mode == "z":
            color = "b"
            xs = self.norm_z[0]
            ys = self.norm_z[1]
            zs = self.norm_z[2]
        else:
            raise Exception("not defined color")
        format = dict(mutation_scale=20, arrowstyle='-|>', color=color, shrinkA=0, shrinkB=0, linewidth=self.line_width)
        arrow = Arrow3D(xs, ys, zs, **format)
        axis_local.add_artist(arrow)

    def draw(self, axis):
        self.__draw_arrow("x", axis)
        self.__draw_arrow("y", axis)
        self.__draw_arrow("z", axis)
        return axis


class GlobalCoordinateSystem(CoordinateSystem):
    def __init__(self):
        rot_nothing = Rotation.from_matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
        )
        super().__init__(0.0, 0.0, 0.0, rot_nothing)

def draw_camera_setup(ds_mode: str):
    if ds_mode == "blender":
        ds = get_dataset_blender()
        arrow_length = 0.4
    elif ds_mode == "dryice":
        ds = get_dataset_dryice()
        arrow_length = 1.0
    else:
        raise Exception("mode not known")
    krt = ds.get_krt()
    list_cs = []
    pos_x = []
    pos_y = []
    pos_z = []
    for key in krt.keys():
        values = krt[key]
        if ds_mode == "dryice":
            rot_krt = ds.get_rot_of_cam(key)
            pos_krt = ds.get_pos_of_cam(key)
        else:
            rot_krt = values["rot"]
            pos_krt = values["pos"]
        rot_cam = Rotation.from_matrix(rot_krt)
        cs_cam = CoordinateSystem(pos_krt[0], pos_krt[1], pos_krt[2], rot_cam, arrow_length=arrow_length)
        list_cs.append(cs_cam)
        pos_x.append(pos_krt[0])
        pos_y.append(pos_krt[1])
        pos_z.append(pos_krt[2])

    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    pos_z = np.array(pos_z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos_x, pos_y, pos_z)
    cs_global = GlobalCoordinateSystem()
    cs_global.draw(ax)
    for cs_camera in list_cs:
        cs_camera.draw(ax)
    plt.show()


if __name__ == "__main__":
    draw_camera_setup("blender")
