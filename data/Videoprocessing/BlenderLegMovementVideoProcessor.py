import cv2
import os
from BlenderLegMovementPrepareConfig import BlenderLegMovementPrepareConfig


class BlenderLegMovementVideoProcessor:

    WIDTH_OUTPUT_IMG = 668
    HEIGHT_OUTPUT_IMG = 1024

    PATH_TRAIN_DATASET = "experiments/blenderLegMovement/data/train"
    PATH_TEST_DATASET = "experiments/blenderLegMovement/data/test"

    def __init__(self):
        ds_config = BlenderLegMovementPrepareConfig()
        self.amplitudes_train: list[str] = ds_config.get_amplitudes_train()
        self.frame_subsampling_train: list[int] = ds_config.get_frame_subsampling_train()
        self.phase_shift_subsampling_train: list[int] = ds_config.get_phase_shift_subsampling_train()

        self.amplitudes_test: list[str] = ds_config.get_amplitudes_test()
        self.frame_subsampling_test: list[int] = ds_config.get_frame_subsampling_test()
        self.phase_shift_subsampling_test: list[int] = ds_config.get_phase_shift_subsampling_test()

    def subsample_train_dataset(self):
        frame_id_to_write = 0
        for path_video in self.amplitudes_train:
            for frame_frequence in self.frame_subsampling_train:
                for phase_shift in self.phase_shift_subsampling_train:
                    frame_id_to_write = self.write_for_each_cam_video(
                        path_video=path_video,
                        frame_frequence=frame_frequence,
                        phase_shift=phase_shift,
                        output_path=self.PATH_TRAIN_DATASET,
                        cur_frame_id_to_write=frame_id_to_write,
                        nof_frames_to_write=100
                    )

    def subsample_test_dataset(self):
        frame_id_to_write = 0
        for path_video in self.amplitudes_test:
            for frame_frequence in self.frame_subsampling_test:
                for phase_shift in self.phase_shift_subsampling_test:
                    frame_id_to_write = self.write_for_each_cam_video(
                        path_video=path_video,
                        frame_frequence=frame_frequence,
                        phase_shift=phase_shift,
                        output_path=self.PATH_TEST_DATASET,
                        cur_frame_id_to_write=frame_id_to_write,
                        nof_frames_to_write=100
                    )

    def write_for_each_cam_video(
            self,
            path_video: str,
            frame_frequence: int,
            phase_shift: int,
            output_path: str,
            cur_frame_id_to_write: int,
            nof_frames_to_write: int = 100,
    ) -> int:
        for i in range(36):
            cam_output_folder_path = output_path + "/" + "{:03d}".format(i)
            cam_video_path = path_video + "/" + "0000-0100.{:03d}.mkv".format(i)
            print("Output Folder: " + cam_output_folder_path)
            print("Video Path: " + cam_video_path)
            self.write_frame_by_frame_to_folder(
                cam_output_folder_path=cam_output_folder_path,
                cam_index=i,
                cam_video_path=cam_video_path,
                frame_frequence=frame_frequence,
                phase_shift=phase_shift,
                nof_frames_to_write=nof_frames_to_write,
                cur_frame_id_to_write=cur_frame_id_to_write
            )
        return cur_frame_id_to_write + nof_frames_to_write

    def write_frame_by_frame_to_folder(
            self,
            cam_output_folder_path: str,
            cam_index: int,
            cam_video_path: str,
            frame_frequence: int,
            phase_shift: int,
            nof_frames_to_write: int,
            cur_frame_id_to_write: int
    ):
        if not os.path.exists(cam_output_folder_path):
            os.makedirs(cam_output_folder_path)
        cap = cv2.VideoCapture(cam_video_path)
        frame_id_output_id = 0
        frame_id_to_retrieve = phase_shift
        frame_id_in_video = 0
        while cap.grab():
            frame_id_in_video = frame_id_in_video + 1
            flag, frame = cap.retrieve()
            if not flag or frame_id_in_video != frame_id_to_retrieve:
                continue
            else:
                image_path = "{}/cam{:03d}_frame{:04d}.jpg".format(
                    cam_output_folder_path,
                    cam_index,
                    cur_frame_id_to_write)
                cur_frame_id_to_write = cur_frame_id_to_write + 1
                frame = cv2.resize(frame, (self.WIDTH_OUTPUT_IMG, self.HEIGHT_OUTPUT_IMG))
                cv2.imwrite(image_path, frame)
                frame_id_to_retrieve = frame_id_to_retrieve + frame_frequence
                frame_id_output_id = frame_id_output_id + 1
                if frame_id_to_retrieve > BlenderLegMovementPrepareConfig.NOF_FRAMES_OF_THE_VIDEOS:
                    frame_id_to_retrieve = frame_id_to_retrieve - BlenderLegMovementPrepareConfig.NOF_FRAMES_OF_THE_VIDEOS
                    cap = cv2.VideoCapture(cam_video_path)
                    frame_id_in_video = 0
            if frame_id_output_id >= nof_frames_to_write:
                break


if __name__ == "__main__":
    video_processor = BlenderLegMovementVideoProcessor()
    video_processor.subsample_train_dataset()
    video_processor.subsample_test_dataset()
