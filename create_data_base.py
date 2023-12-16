import argparse
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


    VDB = VideoDB(fileDescriptionVideos,fileIndividualProfiles)
    VDB.parseDescriptionVideos()
    VDB.add_video_dir_path(video_dir_path)
    VDB.parseIndividualProfiles() #self.IndividualProfiles
    VDB.get_aggregated_profiles() #self.APTV
    VDB.add_Emotions()

    clip_length = 5
    VDB.get_dAPTV(clip_length)
    jump = 0.5
    VDB.get_dAPTV_porogs(jump, type="top")
    VDB.get_positive_ID()
    VDB.get_negative_ID()

    save_pickle(fileVDB,VDB)



             
        

if __name__ == '__main__':
    main()

