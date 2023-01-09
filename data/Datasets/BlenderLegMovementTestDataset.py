from data.Datasets.Blender2Dataset import Blender2Dataset


class BlenderLegMovementTestDataset(Blender2Dataset):
    def get_images_path(self):
        return "experiments/blenderLegMovement/data/test"

    def get_bg_img_path(self):
        return "experiments/blenderLegMovement/data/bg.jpg"

    def get_frame_index_dataset(self, frame_number: int) -> float:
        total_frames = 100
        return float(frame_number % total_frames) / float(total_frames) * 2.0 - 1.0

    def ground_truth_path(self):
        return "experiments/blenderLegMovement/data/groundtruth_test"
