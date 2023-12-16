import argparse
import os
from lib.utils.utils import loadarg

import collections
from collections import Counter

from mvlib.utils  import save_pickle, load_pickle , save_dict
from mvlib.mvideo_lib  import Video , VideoDB


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args_in = parser.parse_args()
    args = loadarg(args_in.config)

    fileDescriptionVideos = f'{args["dataset"]["fileDescriptionVideos"]}'
    fileIndividualProfiles = f'{args["dataset"]["fileIndividualProfiles"]}'
    fileVDB = f'{args["dataset"]["fileVDB"]}'
    video_dir_path = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_videos"]}'
    frames_dir_path = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_frames"]}'
    audios_dir_path = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_audios"]}'

    VDB = load_pickle(fileVDB)
    VDB.filtrMarket( market_filtr=826)
    print("clip_length", VDB.clip_length, VDB.jump)

    VDB.get_aggregated_profiles()  # self.APTV
    VDB.add_Emotions()

    clip_length = 5
    VDB.get_dAPTV(clip_length)
    jump = 0.5
    VDB.get_dAPTV_porogs(jump, type="top")
    VDB.get_positive_ID()
    VDB.get_negative_ID()

    #exit()
    #dir_save = f'{args["dataset"]["data_dir"]}'
    #VDB.split_train_validation_test(dir_save, rate=[80, 10, 10])
    exit()

    # videos --> frame&audio
    if not os.path.exists(frames_dir_path):
        os.makedirs(frames_dir_path)
        os.makedirs(audios_dir_path)

    VDB.parse_video_set(audios_dir_path, mode="audio")
    VDB.parse_video_set(frames_dir_path,  mode ="frames")


    exit()


    # recompute VDB for new clip_length or jump
    clip_length = 5
    VDB.get_dAPTV(clip_length)
    jump = 0.5
    VDB.get_dAPTV_porogs(jump, type="top")

    VDB.get_positive_ID()
    VDB.get_negative_ID()





             
        

if __name__ == '__main__':
    main()

