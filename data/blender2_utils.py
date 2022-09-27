import math

import numpy as np
from scipy.spatial.transform import Rotation as R

class Camera_in_setup:
    def __init__(
            self,
            radius: float,
            camera_nr: int
    ):
        self.phi_degrees = self.get_phi_degrees_from_cam_nr(camera_nr)
        self.theta_degrees = self.get_theta_degrees_from_cam_nr(camera_nr)
        self.theta = self.theta_degrees / 360.0 * 2 * math.pi
        self.phi = self.phi_degrees / 360.0 * 2 * math.pi
        self.radius = radius

    def get_theta_degrees_from_cam_nr(self, camera_nr: int) -> float:
        interval_theta = math.floor(float(camera_nr) / 3.0)
        return 30.0 * interval_theta

    def get_phi_degrees_from_cam_nr(self, camera_nr: int) -> float:
        interval_phi = (float(camera_nr) % 3) + 1
        return 45.0 * interval_phi

    def get_x(self) -> float:
        return self.radius * math.sin(self.phi) * math.cos(self.theta)

    def get_y(self) -> float:
        return self.radius * math.sin(self.phi) * math.sin(self.theta)

    def get_z(self) -> float:
        return self.radius * math.cos(self.phi) + 0.8

    def get_x_rotation_degrees(self) -> float:
        return self.phi_degrees

    def get_y_rotation_degrees(self) -> float:
        return 0.0

    def get_z_rotation_degrees(self) -> float:
        return self.theta_degrees + 90

    def get_cam_pos(self) -> np.ndarray:
        xyz_pos = [self.get_x(), self.get_y(), self.get_z()]
        return np.array(xyz_pos).astype(np.float32)

    def get_cam_rot_matrix(self) -> np.ndarray:
        xyz_rot = [self.get_x_rotation_degrees(), self.get_y_rotation_degrees(), self.get_z_rotation_degrees()]
        xyz_rot = R.from_euler('xyz', xyz_rot, degrees=True)
        return np.array(xyz_rot.as_matrix()).astype(np.float32)

if __name__ == "__main__":
    radius = 3.5
    for i in range(0, 36):
        print(" ")
        print("{:3}".format(i))
        camera = Camera_in_setup(radius, i)
        print("X: {0:.2f}".format(camera.get_x()))
        print("Y: {0:.2f}".format(camera.get_y()))
        print("Z: {0:.2f}".format(camera.get_z()))

        print("X Rotation: {0:.2f}".format(camera.get_x_rotation_degrees()))
        print("Y Rotation: {0:.2f}".format(camera.get_y_rotation_degrees()))
        print("Z Rotation: {0:.2f}".format(camera.get_z_rotation_degrees()))

