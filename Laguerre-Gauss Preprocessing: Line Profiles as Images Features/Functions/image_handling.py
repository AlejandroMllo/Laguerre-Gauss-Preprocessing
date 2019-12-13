import os

import matplotlib.image as mpimg
import numpy as np
import cv2 as cv2


def rgb2grayscale(rgb_img):

    rgb_img = np.array(rgb_img)

    if rgb_img.ndim != 3:
        return rgb_img

    img = np.zeros(rgb_img.shape)
    img[:, :, 0] = rgb_img[:, :, 0] * 0.2125  # RED
    img[:, :, 1] = rgb_img[:, :, 1] * 0.7154  # GREEN
    img[:, :, 2] = rgb_img[:, :, 2] * 0.0721  # BLUE

    return np.sum(img, axis=2)


def find_files(path, sorted_by_idx=True):
    files = next(os.walk(path))[2]

    if sorted_by_idx:
        files = sorted(
            files, key=lambda f: int("".join(list(filter(str.isdigit, f))))
        )

    return np.array(files)


def load_images(image_names, path):
    images = []

    for name in image_names:
        img_path = path + name
        img = mpimg.imread(img_path)
        img = rgb2grayscale(img)
        images.append(img)

    return images


def load_videos(videos_names, path, return_frames=True):

    videos = []

    for name in videos_names:
        vid_path = path + name
        video = cv2.VideoCapture(vid_path)

        if return_frames:
            success, frame = video.read()
            i = 0
            while success:
                if i % 50 == 0:
                    videos.append(np.array(frame))
                success, frame = video.read()
                i += 1
        else:
            videos.append(video)

    return videos


def show_video(video):

    if not video.isOpened():
        print("Error opening video stream or file")

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()