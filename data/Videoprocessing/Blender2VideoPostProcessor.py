import cv2
import os


class Blender2VideoPostProcessor:
    WIDTH_OUTPUT_IMG = 667
    HEIGHT_OUTPUT_IMG = 1024

    def write_frame_by_frame_to_folder(self, cam_index: str, video_path: str):
        folder_path = current_path + "/" + cam_index
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while cap.grab():
            frame_id = frame_id + 1
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                image_path = "{}/cam{}_frame{:04d}.jpg".format(folder_path, cam_index, frame_id)
                frame = cv2.resize(frame, (self.WIDTH_OUTPUT_IMG, self.HEIGHT_OUTPUT_IMG))
                cv2.imwrite(image_path, frame)
                print(image_path)


if __name__ == "__main__":
    current_path = "experiments/blender2/data"
    for i in range(36):
        folder_name = "{:03d}".format(i)
        video_name = "0000-0500.{:03d}.mkv".format(i)
        print("Foldername: " + folder_name)
        print("Videoname: " + video_name)
        video_path = current_path + "/" + video_name
        postProcessor = Blender2VideoPostProcessor()
        postProcessor.write_frame_by_frame_to_folder(folder_name, video_path)
