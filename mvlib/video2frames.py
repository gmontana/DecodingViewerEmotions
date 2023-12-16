import csv
import re
import numpy as np
import os


import collections
from collections import Counter

from mvlib.utils  import save_pickle, load_pickle , save_dict, create_clean_DIR


def run_frames(split,  output_folder, fps , ERRORs):

    print(f"run_frames: {len(split)}  {output_folder} fps:{fps}")

    for i, [id, file_mp4] in enumerate(split):
        #print(i, id, file_mp4)

        create_clean_DIR(f"{output_folder}/{id}")
        if fps > 0:
            cmd_frames = f"ffmpeg -loglevel panic -i {file_mp4}  -vf \"scale=-1:256,fps={fps}\" -q:v 0 \"{output_folder}/{id}/%06d.jpg\" "
        else:
            cmd_frames = f"ffmpeg -loglevel panic -i {file_mp4}  -vf \"scale=-1:256\" -q:v 0 \"{output_folder}/{id}/%06d.jpg\" "
        #print(cmd_frames)

        try:
            os.system(cmd_frames)
        except:
            print(f"An exception occurred {id}\n")
            ERRORs[id] = 1
        #exit()



def run_audio(split,  output_folder,  ERRORs):

    for i, [id, file_mp4] in enumerate(split):

        cmd_audio = f"ffmpeg -loglevel panic -i {file_mp4}   {output_folder}/{id}.wav"
        print(cmd_audio)

        try:
            os.system(cmd_audio)
        except:
            print(f"An exception occurred {id}\n")
            ERRORs[id] = 1


def split_data(data, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(data), n):
        yield data[i:i + n]

def parse_video_set(list_videos, output_folder, fps = None, mode="frames", SIZE_SPLIT = 1000):

    #list_videos = [[VID, path_video]]

    splits = list(split_data(list_videos, SIZE_SPLIT))

    import multiprocessing
    manager = multiprocessing.Manager()
    ERRORs = manager.dict()

    for i, split in enumerate(splits):
        print(i, len(split))



    threads = []
    for i, split in enumerate(splits):

        if mode == "frames":
           thread = multiprocessing.Process(target=run_frames, args=(split, output_folder, fps ,ERRORs))
        if mode == "audio":
           thread = multiprocessing.Process(target=run_audio,  args=(split, output_folder,ERRORs))


        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


    print(f"Total Errors conversion: {mode} {len(ERRORs)}")
    return ERRORs














       

