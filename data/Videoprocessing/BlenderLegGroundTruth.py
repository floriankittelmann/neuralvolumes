from BlenderLegMovementPrepareConfig import BlenderLegMovementPrepareConfig
import shutil

"""
Please execute this code in the blender scripting environment:
import bpy
import subprocess
import sys
scene = bpy.data.scenes['Scene']

last_frame = scene.frame_end
for i in range(0, last_frame + 1):
    scene.frame_set(i)
    path_stl = '<path>\\frame{:04d}.stl'.format(i)
    bpy.ops.export_mesh.stl(filepath=path_stl)
"""


class BlenderLegGroundTruth:
    PATH_SLIGHTLY_RAISED = "experiments/blenderLegMovement/data/blender/groundtruth_legSlightlyRaised"
    PATH_MEDIUM_RAISED = "experiments/blenderLegMovement/data/blender/groundtruth_legMediumRaised"
    PATH_HIGH_RAISED = "experiments/blenderLegMovement/data/blender/groundtruth_legHighRaised"

    PATH_DST_TRAIN = "experiments/blenderLegMovement/data/groundtruth_train"
    PATH_DST_TEST = "experiments/blenderLegMovement/data/groundtruth_test"

    def __init__(self):
        ds_config = BlenderLegMovementPrepareConfig()
        self.amplitudes_train: list[str] = [self.PATH_SLIGHTLY_RAISED, self.PATH_MEDIUM_RAISED]
        self.frame_subsampling_train: list[int] = ds_config.get_frame_subsampling_train()
        self.phase_shift_subsampling_train: list[int] = ds_config.get_phase_shift_subsampling_train()

        self.amplitudes_test: list[str] = [self.PATH_SLIGHTLY_RAISED, self.PATH_HIGH_RAISED]
        self.frame_subsampling_test: list[int] = ds_config.get_frame_subsampling_test()
        self.phase_shift_subsampling_test: list[int] = ds_config.get_phase_shift_subsampling_test()

    def prepare_ground_truth_train(self):
        frame_id_to_write = 0
        for path_ground_truth in self.amplitudes_train:
            for frame_frequence in self.frame_subsampling_train:
                for phase_shift in self.phase_shift_subsampling_train:
                    frame_id_to_write = self.write_100frames(
                        path_ground_truth,
                        frame_frequence,
                        phase_shift,
                        self.PATH_DST_TRAIN,
                        frame_id_to_write
                    )
                    print("finished 100 frames")

    def prepare_ground_truth_test(self):
        frame_id_to_write = 0
        for path_ground_truth in self.amplitudes_test:
            for frame_frequence in self.frame_subsampling_test:
                for phase_shift in self.phase_shift_subsampling_test:
                    frame_id_to_write = self.write_100frames(
                        path_ground_truth,
                        frame_frequence,
                        phase_shift,
                        self.PATH_DST_TEST,
                        frame_id_to_write
                    )
                    print("finished 100 frames")

    def write_100frames(
            self,
            path_ground_truth: str,
            frame_frequence: int,
            phase_shift: int,
            dst_path: str,
            frame_id_to_write: int
    ):
        for i in range(100):
            frame_to_take = frame_frequence * i + phase_shift
            while frame_to_take >= 100:
                frame_to_take = frame_to_take - 100
            path_frame_source = "{}/frame{:04d}.stl".format(path_ground_truth, frame_to_take)
            path_frame_dst = "{}/frame{:04d}.stl".format(dst_path, frame_id_to_write)
            shutil.copyfile(path_frame_source, path_frame_dst)
            frame_id_to_write = frame_id_to_write + 1
        return frame_id_to_write


if __name__ == "__main__":
    process = BlenderLegGroundTruth()
    print("start test")
    process.prepare_ground_truth_train()
    print("start train")
    process.prepare_ground_truth_test()
