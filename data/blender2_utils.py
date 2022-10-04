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
        #self.phi_training = (180.0 - self.phi_degrees) / 360.0 * 2 * math.pi
        #self.theta_training = (180.0 - self.theta_degrees) / 360.0 * 2 * math.pi

    def get_theta_degrees_from_cam_nr(self, camera_nr: int) -> float:
        interval_theta = math.floor(float(camera_nr) / 3.0)
        return 30.0 * interval_theta

    def get_phi_degrees_from_cam_nr(self, camera_nr: int) -> float:
        interval_phi = (float(camera_nr) % 3) + 1
        return 45.0 * interval_phi

    def get_x(self) -> float:
        phi = self.phi
        theta = self.theta
        return self.radius * math.sin(phi) * math.cos(theta)

    def get_y(self) -> float:
        phi = self.phi
        theta = self.theta
        return self.radius * math.sin(phi) * math.sin(theta)

    def get_z(self) -> float:
        phi = self.phi
        return self.radius * math.cos(phi) + 0.8

    def get_x_rotation_blender_degrees(self) -> float:
        return self.phi_degrees

    def get_y_rotation_blender_degrees(self) -> float:
        return 0.0

    def get_z_rotation_blender_degrees(self) -> float:
        return self.theta_degrees + 90

    def get_cam_pos_training(self) -> np.ndarray:
        xyz_pos = [self.get_x(), self.get_y(), self.get_z()]
        return np.array(xyz_pos).astype(np.float32)

    def get_cam_rot_matrix_training(self) -> np.ndarray:
        xyz_rot = [self.get_x_rotation_blender_degrees(),
                   self.get_y_rotation_blender_degrees(),
                   self.get_z_rotation_blender_degrees()]
        xyz_rot = R.from_euler('xyz', xyz_rot, degrees=True)
        extrinsic_matrix = np.array(xyz_rot.as_matrix()).astype(np.float32)
        rad_rot = 180.0 / 360.0 * 2 * math.pi
        rot_matrix = np.asarray([
            [math.cos(rad_rot), 0, math.sin(rad_rot)],
            [0, 1, 0],
            [-math.sin(rad_rot), 0, math.cos(rad_rot)],
        ])
        final_matrix = extrinsic_matrix.dot(rot_matrix)
        return final_matrix.astype(np.float32)


if __name__ == "__main__":
    radius = 3.5
    for i in range(0, 36):
        print(" ")
        print("{:3}".format(i))
        camera = Camera_in_setup(radius, i)
        print("--- blender ----")
        print("X: {0:.2f}".format(camera.get_x()))
        print("Y: {0:.2f}".format(camera.get_y()))
        print("Z: {0:.2f}".format(camera.get_z()))

        print("X Rotation: {0:.2f}".format(camera.get_x_rotation_blender_degrees()))
        print("Y Rotation: {0:.2f}".format(camera.get_y_rotation_blender_degrees()))
        print("Z Rotation: {0:.2f}".format(camera.get_z_rotation_blender_degrees()))

        print("--- neural volumes ----")
        print(camera.get_cam_pos_training())
        print(R.from_matrix(camera.get_cam_rot_matrix_training()).as_quat())

