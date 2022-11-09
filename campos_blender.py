from data.CameraSetups.CameraSetupInBlender2 import CameraSetupInBlender2
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    for i in range(0, 36):
        print(" ")
        print("{:3}".format(i))
        camera = CameraSetupInBlender2(i)
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

