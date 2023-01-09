from data.Datasets.Blender2Dataset import Blender2Dataset


class BlenderLegMovementTestDataset(Blender2Dataset):
    def get_images_path(self):
        return "experiments/blenderLegMovement/data/test"

    def get_bg_img_path(self):
        return "experiments/blenderLegMovement/data/bg.jpg"

    def ground_truth_path(self):
        return "experiments/blenderLegMovement/data/groundtruth_test"
