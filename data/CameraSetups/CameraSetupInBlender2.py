import math

import numpy as np
from scipy.spatial.transform import Rotation as R


class CameraSetupInBlender2:
    def __init__(self, camera_nr: int):
        self.radius = 3.5  # in meters
        self.phi_degrees = self.get_phi_degrees_from_cam_nr(camera_nr)
        self.theta_degrees = self.get_theta_degrees_from_cam_nr(camera_nr)
        self.theta = self.theta_degrees / 360.0 * 2 * math.pi
        self.phi = self.phi_degrees / 360.0 * 2 * math.pi

    def get_radius(self) -> float:
        return self.radius

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
        return self.radius * math.cos(phi)

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
        y_rot_matrix = np.asarray([
            [math.cos(rad_rot), 0, math.sin(rad_rot)],
            [0, 1, 0],
            [-math.sin(rad_rot), 0, math.cos(rad_rot)],
        ])
        return extrinsic_matrix.dot(y_rot_matrix).T.astype(np.float32)

    def get_focal_length(self):
        focal_length_blender = 40.0
        sensor_width_longer_distance_blender = 36.0
        sensor_width_shorter_distance_blender = sensor_width_longer_distance_blender / float(
            self.get_img_width()) * float(self.get_img_height())
        focal_length_ld_pixels = focal_length_blender / sensor_width_longer_distance_blender * float(
            self.get_img_width())
        focal_length_sd_pixels = focal_length_blender / sensor_width_shorter_distance_blender * float(
            self.get_img_height())
        if focal_length_ld_pixels != focal_length_sd_pixels:
            raise Exception("they should be the same")
        return focal_length_ld_pixels

    def get_principt_height(self):
        return self.get_img_height() * 0.5

    def get_principt_width(self):
        return self.get_img_width() * 0.5

    @staticmethod
    def get_img_height() -> int:
        return 667

    @staticmethod
    def get_img_width() -> int:
        return 1024
