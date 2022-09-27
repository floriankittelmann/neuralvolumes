import cv2
import os


def write_frame_by_frame_to_folder(cam_index: str, video_path: str):
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
            frame = cv2.resize(frame, (1024, 667))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(image_path, frame)
            print(image_path)


if __name__ == "__main__":
    video_files = {
        '001': '0000-0130.001.mkv',
        '002': '0000-0130.002.mkv',
        '003': '0000-0130.003.mkv',
        '004': '0000-0130.004.mkv',
        '005': '0000-0130.005.mkv',
        '006': '0000-0130.006.mkv'
    }

    current_path = os.getcwd()
    for folder_name in video_files.keys():
        print(folder_name)
        video_name = video_files[folder_name]
        video_path = current_path + "/" + video_name
        write_frame_by_frame_to_folder(folder_name, video_path)
