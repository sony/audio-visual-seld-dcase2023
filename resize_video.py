# Copyright 2023 Sony Group Corporation.

import os
import glob
import tqdm
import moviepy.editor  # for resize
from moviepy.video.io.VideoFileClip import VideoFileClip


def main_per_take(path_input, path_output, new_width):
    theta_video = VideoFileClip(path_input)
    theta_video = theta_video.resize(width=new_width)
    theta_video.write_videofile(path_output)


def main():
    path_dir = "./data_dcase2023_task3/video_dev"

    old_dir_name = "/video_dev/"
    new_dir_name = "/video_360x180_dev/"
    new_width = 360  # lower resolution video (w=360, h=180)

    path_files = glob.glob("{}/*/*.mp4".format(path_dir))  # e.g., "./data_dcase2023_task3/video_dev/dev-train-tau/fold3_room12_mix001.mp4"
    path_files = sorted(path_files)

    for path_input in tqdm.tqdm(path_files):
        path_output = path_input.replace(old_dir_name, new_dir_name)
        os.makedirs(os.path.dirname(path_output), exist_ok=True)
        main_per_take(path_input, path_output, new_width)


if __name__ == "__main__":
    main()
