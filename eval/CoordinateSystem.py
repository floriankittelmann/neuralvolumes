import numpy as np
from scipy.spatial.transform import Rotation
from eval.Arrow3D import Arrow3D


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
