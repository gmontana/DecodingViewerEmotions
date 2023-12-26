
from PIL import Image
import os

import torch
import torchaudio
import torchvision

from collections import Counter

import random

import numpy as np
from numpy.random import randint

from lib.dataset.audio import get_audio_x
from lib.dataset.video import get_video_x
from mvlib.utils import save_pickle, load_pickle
from mvlib.mvideo_lib import VideoDB, Video


class PredictFullVideoDataSet(torch.utils.data.Dataset):
    def __init__(self, input_file,   args={}, mode_train_val="validation"):
        """
        Initialize the dataset for full video prediction.

        Parameters:
        ----------
        input_file : str
            Path to the input file containing data information.
        args : dict
            Configuration arguments including paths and processing parameters.
        mode_train_val : str
            Mode of the dataset, which can be 'training', 'validation', 'test', or 'predict'.

        """

        self.input_file = input_file

        self.args = args
        self.clip_length = args["emotion_jumps"]["clip_length"]
        self.fps = args["dataset"]["fps"]

        param_dataset = args["dataset"]
        self.path_video_imagefolders = f'{param_dataset["data_dir"]}/{param_dataset["dir_frames"]}'
        self.path_audio_folder = f'{param_dataset["data_dir"]}/{param_dataset["dir_audios"]}'

        self.param_adcumen = args["emotion_jumps"]
        self.clip_length = self.param_adcumen["clip_length"]

        self.mode_train_val = mode_train_val

        self.parse_input_file()

        # self.__getitem__(100)

        # exit()

    def __getitem__(self, index):
        """
        Retrieve a single item from the dataset.

        Parameters:
        ----------
        index : int
            Index of the item to retrieve.

        Returns:
        -------
        tuple
            A tuple containing ID, time, video tensor, and audio tensor for the given index.
        """

        record = self.records[index]
        x_video, x_audio = torch.zeros(1), torch.zeros(1)
        ID = record["ID"]
        t = record["t"]

        if self.args["TSM"]["video_segments"] > 0:
            x_video = get_video_x(record,  self.args, self.mode_train_val)

        if self.args["TSM"]["audio_segments"] > 0:
            x_audio = get_audio_x(record,  self.args, self.mode_train_val)

        return ID, t, x_video, x_audio

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        -------
        int
            The number of items in the dataset.
        """
        return len(self.records)

    def parse_input_file(self):
        """
        Parse the input file to process and prepare the data records.
        This method reads the input file, processes each line, and prepares the data records for prediction tasks. 
        It handles both video and audio segments based on the configuration provided in 'args'.
        """
        line_data_t = []
        self.records_data = {}

        if self.input_file != None:
            lines = [x.strip().split(' ') for x in open(self.input_file)]
        else:
            lines = [[ID] for ID in os.listdir(self.path_video_imagefolders)]

        count_FILTERED_BY_SIZE, count_GOOD = 0, 0

        for i, item in enumerate(lines):

            line_data = {}

            ID = item[0]
            line_data["ID"] = ID

            if self.args["TSM"]["video_segments"] > 0:
                ifolder = f"{self.path_video_imagefolders}/{ID}"
                if not os.path.isdir(ifolder):
                    continue
                ifolder_size = int(len(os.listdir(ifolder)))
                if ifolder_size < 2:
                    count_FILTERED_BY_SIZE += 1
                    continue

                line_data["imagefolder"] = ifolder
                line_data["imagefolder_size"] = ifolder_size

            if self.args["TSM"]["audio_segments"] > 0:
                wavefile = f"{self.path_audio_folder}/{ID}.wav"
                if not os.path.isfile(wavefile):
                    continue
                line_data["audio_file"] = wavefile

            ifolder_size = line_data["imagefolder_size"]
            T = ifolder_size // self.fps
            for t in range(T - self.clip_length):
                line_data_new = line_data.copy()
                line_data_new["t"] = t
                line_data_t.append(line_data_new)

            count_GOOD += 1

        print("count_FILTERED_BY_SIZE, count_GOOD",
              count_FILTERED_BY_SIZE, count_GOOD)

        self.records = line_data_t

        print("get_records:",  len(self.records))


def GetDataSetPredict(args, mode_train_val="training"):
    """
    Create and return a dataset object for prediction.

    Parameters:
    ----------
    args : dict
        Configuration arguments including paths and processing parameters.
    mode_train_val : str
        Mode of the dataset, which can be 'training', 'validation', 'test', or 'predict'.

    Returns:
    -------
    PredictFullVideoDataSet
        The created dataset object for prediction tasks.
    """
    param_dataset = args["dataset"]

    if mode_train_val == "training":
        file_list = f'{param_dataset["file_train_list"]}'
    elif mode_train_val == "validation":
        file_list = f'{param_dataset["file_val_list"]}'
    elif mode_train_val == "test":
        file_list = f'{param_dataset["file_test_list"]}'
    elif mode_train_val == "predict":
        file_list = f'{param_dataset["file_predict_list"]}'
    else:
        file_list = None
        print(
            "Error: GetDataSetPredict: file_list is not defined , check args[dataset]\n")

    ADS = PredictFullVideoDataSet(
        file_list, args=args, mode_train_val=mode_train_val)

    return ADS


def GetDataLoadersPredict(ADS, args):
    """
    Create and return a DataLoader for the prediction dataset.

    Parameters:
    ----------
    ADS : PredictFullVideoDataSet
        The prediction dataset object for which to create the DataLoader.
    args : dict
        Configuration arguments including batch size and number of workers.

    Returns:
    -------
    DataLoader
        The created DataLoader object for the prediction dataset.
    """

    data_loader = torch.utils.data.DataLoader(
        ADS,
        batch_size=args["net_run_param"]["batch_size"], num_workers=args["net_run_param"]["num_workers"],
        shuffle=False, pin_memory=True, drop_last=False
    )

    return data_loader
