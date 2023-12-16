import os
import argparse
import json

from lib.utils.utils import loadarg , create_clean_DIR




def run_frames(FID, param,  error_conversion):
    fps =  param["fps"]
    path_frame_folder = param["path_frame_folder"]
    frames_tmpl  = param["frames_tmpl"]

    frames_tmpl = frames_tmpl.rstrip(".jpg")
    frames_tmpl = frames_tmpl.strip("}")
    frames_tmpl = frames_tmpl.strip("{")
    frames_tmpl = frames_tmpl.strip(":")
    frames_tmpl = "%" + frames_tmpl


    print("run_frames: ", len(FID),  path_frame_folder, "frames_tmpl:", frames_tmpl)
    #exit()
    print("run_frames fps: ", fps)

    for i, [id, file_video] in enumerate(FID):
        #print(i, id, file_mp4)

        create_clean_DIR(f"{path_frame_folder}/{id}")
        if fps > 0:
            cmd_frames = f"ffmpeg  -i {file_video}  -vf \"scale=-1:256,fps={fps}\" -q:v 0 \"{path_frame_folder}/{id}/{frames_tmpl}.jpg\" "
        else:
            cmd_frames = f"ffmpeg  -i {file_video}  -vf \"scale=-1:256\" -q:v 0 \"{path_frame_folder}/{id}/{frames_tmpl}.jpg\" "
#-loglevel panic
        #print(cmd_frames)
        #exit()
        try:
            os.system(cmd_frames)
        except:
            error_conversion[id] = "video"
            print(f"An exception occurred {id}\n")


def run_audio(HHV_split, param, error_conversion):

    path_data_out_audios =  param["path_audio_folder"]


    for i, [id, file_mp4] in enumerate(HHV_split):

        cmd_audio = f"ffmpeg -loglevel panic -i {file_mp4}   {path_data_out_audios}/{id}.wav"
        print("cmd_audio ", cmd_audio )

        try:
            os.system(cmd_audio)
        except:
            error_conversion[id] = "audio"
            print(f"An exception occurred {id}\n")


def split_data(data, n):
    """Yield successive n-sized chunks from data"""
    for i in range(0, len(data), n):
        yield data[i:i + n]

def multirun_video(FID, param, THREAD_N):

    SIZE_SPLIT = len(FID) // THREAD_N
    splits = list(split_data(FID, SIZE_SPLIT))

    for i, split in enumerate(splits):
        print(i, len(split), split[0])

    import multiprocessing
    manager = multiprocessing.Manager()
    error_conversion = manager.dict()

    threads = []
    for i, split in enumerate(splits):
        thread = multiprocessing.Process(target=run_frames, args=(split, param, error_conversion))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return error_conversion


def multirun_audio(FID, param, THREAD_N ):

    SIZE_SPLIT = len(FID) // THREAD_N
    splits = list(split_data(FID, SIZE_SPLIT))

    for i, split in enumerate(splits):
        print(i, len(split), split[0])

    path_data_out_audios = param["path_audio_folder"]
    create_clean_DIR(f"{path_data_out_audios}")

    import multiprocessing
    manager = multiprocessing.Manager()
    error_conversion = manager.dict()

    threads = []
    for i, split in enumerate(splits):
        thread = multiprocessing.Process(target=run_audio, args=(split,  param, error_conversion))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return error_conversion



import sys

def main():


    print(sys.argv)

    input_video_dir = sys.argv[1]

    path_frame_dir = f'{input_video_dir}_frames}'

    try:
        os.mkdir(path_frame_dir)
    except OSError:
        print("Error MkDir")




if __name__ == '__main__':
    main()