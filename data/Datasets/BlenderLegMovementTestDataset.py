from data.Datasets.Blender2Dataset import Blender2Dataset


class BlenderLegMovementTestDataset(Blender2Dataset):
    def get_images_path(self):
        return "experiments/blenderLegMovement/data/test"

    def get_bg_img_path(self):
        return "experiments/blenderLegMovement/data/bg.jpg"

    def get_frame_index_dataset(self, frame_number: int) -> int:
        return frame_number % 100
