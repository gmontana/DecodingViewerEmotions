import os
from youtubesearchpython import VideosSearch
import urllib.request
from pytube import YouTube
import json
import shutil

import json
import argparse

def new_dir(path):
    if os.path.exists(path):
        print("exist:  ", path)

        return 1
    else:
        os.mkdir(path)
        return -1

from mvlib.video2frames import parse_video_set

def main():

    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str) ##dataset_info.args
    args = parser.parse_args()
    print("args", args)


    with open(f"{args.config}") as f:
        dataset_info = json.load(f)

    data_dir =  dataset_info["dataset"]["data_dir"]
    dir_videos =  dataset_info["dataset"]["dir_videos"]
    dir_frames = dataset_info["dataset"]["dir_frames"]
    dir_audios = dataset_info["dataset"]["dir_audios"]

    new_dir(f"{data_dir}/{dir_frames}")
    new_dir(f"{data_dir}/{dir_audios}")


    list_videos = os.listdir(f"{data_dir}/{dir_videos}")
    list_videos_toframe = []
    for id in list_videos:
        f_mp4 = f"{data_dir}/{dir_videos}/{id}"
        id = id.replace('.mp4', "")
        list_videos_toframe.append([id, f_mp4])
        print([id, f_mp4])

    parse_video_set(list_videos_toframe, f"{data_dir}/{dir_frames}", fps=10, mode="frames", SIZE_SPLIT=1000)
    parse_video_set(list_videos_toframe, f"{data_dir}/{dir_audios}", fps=None, mode="audio", SIZE_SPLIT=1000)


if __name__ == '__main__':
    main()

