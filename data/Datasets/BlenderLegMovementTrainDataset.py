from data.Datasets.Blender2Dataset import Blender2Dataset


class BlenderLegMovementTrainDataset(Blender2Dataset):
    def get_images_path(self):
        return "experiments/blenderLegMovement/data/train"

    def get_bg_img_path(self):
        return "experiments/blenderLegMovement/data/bg.jpg"

    def get_frame_index_dataset(self, frame_number: int) -> int:
        return frame_number % 100
