"""
predict_annotate_video.py

Overview:
This script is designed to annotate videos with predictions, typically emotion predictions, and assemble the annotated
frames back into videos. It processes a list of video IDs, applies annotations to frames based on the provided
predictions, and uses ffmpeg to assemble the annotated frames back into videos with audio. The script is useful for
visualizing the predictions made by a model on video data, providing a visual and intuitive way to review and analyze
the model's performance.

The script supports multimodal inputs and can be used in conjunction with models to make and assemble predictions from
all time windows of a video. It is particularly useful in scenarios where predictions need to be visualized on the
original video data, such as in emotion recognition tasks.

Key Components:
- IndicatorOnImage: A class for adding indicators (emotion labels) on images.
- Annotation Loop: Loops over each video ID, applying annotations for each time window based on the predictions.
- Video Assembly: Uses ffmpeg to assemble the annotated frames back into videos with audio.

Usage:
The script is typically run from the command line with the necessary paths for data configuration, predicted results,
and output directory specified. It then automatically processes each video, applying annotations and assembling the
results into annotated videos.

Example Command:
python predict_annotate_video.py --data <data_config_file> --predicted <path_to_predicted> --output <output_dir_for_videos>

This command would run the script, executing the annotation and assembly process for each video specified in the
prediction file. The script is useful in scenarios where visualizing model predictions directly on the video data is
required.

Note:
Ensure that the 'ffmpeg' tool is installed and accessible in your environment as it is used for video assembly. The
specific configurations, models, and settings are defined within the script and may need to be adjusted based on the
requirements of the annotation tasks or the available models. Ensure that the input data, prediction results, and
output directories are correctly set up and accessible.
"""


import argparse
import json
import os
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, accuracy_score
import warnings
from sklearn.exceptions import DataConversionWarning

import statistics
import random
from collections import Counter


from PIL import Image, ImageDraw
from PIL import ImageFont


from lib.utils.utils import loadarg,  AverageMeter, get_labels, remap_acc_to_emotions, multiclass_accuracy

import pickle


def save_pickle(file_save, mydict):
    f = open(file_save, "wb")
    pickle.dump(mydict, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_pickle(file_pickle):
    f = open(file_pickle, "rb")
    mydict = pickle.load(f)
    f.close()
    return mydict


def new_dir(path):
    if os.path.exists(path):
        print("exist:  ", path)
        return 1
    else:
        os.mkdir(path)
        return -1


def load_predict(file_predict, *args):
    """
    Load prediction results from files.

    Parameters
    ----------
    file_predict : str
        The base path for prediction files.
    *args : list
        Additional arguments specifying the types of predictions to load.

    Returns
    -------
    list
        A list of prediction results.

    Description
    -----------
    This function iterates over the specified types of predictions, constructs the file path for each,
    and loads the prediction results using the load_pickle function. It is typically used to load
    multiple prediction results for further processing or analysis.
    """
    print("load_predict", args)
    yV = []
    for keyV in args[0]:
        print("keyV", keyV)
        file_save = f"{file_predict}_{keyV}"
        yV.append(load_pickle(file_save))
    return yV


def save_predict(file_predict,  **kwargs):
    """
    Save prediction results to files.

    Parameters
    ----------
    file_predict : str
        The base path for prediction files.
    **kwargs : dict
        Keyword arguments where keys are the types of predictions and values are the predictions to save.

    Description
    -----------
    This function iterates over the specified types of predictions and their corresponding results,
    constructs the file path for each, and saves the results using the save_pickle function. It is
    typically used to save multiple prediction results after processing or generating them.
    """
    for keyV in kwargs:
        file_save = f"{file_predict}_{keyV}"
        save_pickle(file_save, kwargs[keyV])


def assemble(y_V, y_t, y_id):
    """
    Assemble prediction results for analysis.

    Parameters
    ----------
    y_V : list
        List of predicted values.
    y_t : list
        List of true values.
    y_id : list
        List of identifiers for the predictions.

    Returns
    -------
    dict
        A dictionary representing the assembled profile of predictions.

    Description
    -----------
    This function iterates over the provided predictions, true values, and identifiers, assembling them
    into a dictionary structure for easy analysis. Each entry in the dictionary corresponds to a unique
    identifier and contains the associated predictions and true values. It is typically used to organize
    prediction results for further processing or visualization.
    """
    profile_V, stat_V = {}, Counter()
    for i, id in enumerate(y_id):
        p_V, t = y_V[i],  y_t[i]
        # print("p_V, t", p_V, t)
        if id not in profile_V:
            profile_V[id] = []
        profile_V[id].append([p_V, t])
        stat_V[p_V] += 1
    print("stat_V", stat_V)

    return profile_V


class IndicatorOnImage:
    """
    A class for adding indicators (emotion labels) on images.

    Parameters
    ----------
    BarNames : dict
        A dictionary mapping integer IDs to string labels.
    Title : str, optional
        A title for the indicator, if any.
    scale : int, optional
        A scaling factor for the size of the indicators.
    position : str, optional
        The position of the indicator on the image (e.g., "LL" for lower left).

    Methods
    -------
    add_on_image(file_image, indicator_values, output_path=None):
        Adds indicators to the specified image and saves the result.

    Description
    -----------
    This class is used to add visual indicators, such as emotion labels, to images. It supports customizing
    the appearance and position of the indicators. It is typically used to annotate frames with predictions
    before assembling them back into videos.
    """

    def __init__(self,  BarNames, Title=None, scale=1, position="LL"):
        self.Title = Title
        self.BarNames = BarNames  # dictinary: int ->string

        self.N = len(self.BarNames)

        self.size_Title_H = 0
        if self.Title != None:
            self.size_Title_H = int(20 * scale)

        self.size = (int(150 * scale),
                     int(15 * self.N * scale + self.size_Title_H))

        self.size_line_H = int(15 * scale)
        self.TitleFont = int(14 * scale)
        self.BarFont = int(10 * scale)

    def add_on_image(self, file_image, indicator_values, output_path=None):
        # indicator_values:  dictinary: int -> int
        if output_path == None:
            output_path = file_image

        image = Image.open(file_image)
        draw = ImageDraw.Draw(image)
        (W, H) = image.size

        draw.rectangle((0,  H - self.size[1], self.size[0], H), fill="black")

        # myFont = ImageFont.truetype('FreeMono.ttf', 15)
        # draw.text((0,  H - self.size[1]), f"{Title}", (255, 255, 255), font=myFont)

        for i, id in enumerate(indicator_values):
            (x, y, x1, y1) = (0, H - self.size[1] + self.size_Title_H + i * self.size_line_H + 10) + (
                40, H - self.size[1] + self.size_Title_H + i * self.size_line_H)
            value = indicator_values[id]
            name = self.BarNames[id]

            # myFont = ImageFont.truetype('FreeMono.ttf', 13)
            myFont = ImageFont.load_default()
            draw.text((x, y-8),  f"{name}:", (255, 255, 255), font=myFont)
            if value < 3:
                line_color = (102, 3, 252)  # "blue"
            elif value < 5:
                line_color = 3, 157, 252  # ""blue2""
            elif value < 7:
                line_color = (231, 252, 3)  # "yellow"
            elif value < 9:
                line_color = (252, 123, 3)  # "orange"
            else:
                line_color = (252, 15, 3)  # "red"
            draw.line((100+x, y, 100+x+5*value, y), width=4, fill=line_color)

        image.save(output_path)


def main():

    global args_model, args_data

    parser = argparse.ArgumentParser()
    # arg.json file (specifies dataset for prediction)
    parser.add_argument("--data", type=str)
    parser.add_argument('--predicted', type=str)  # profile_P:  for prediction
    parser.add_argument('--output', type=str)  # <id> for prediction
    args_in = parser.parse_args()

    # Specify dataset for prediction args.json file
    if args_in.data:
        args = loadarg(args_in.data)
    else:
        print("No data for prediction is provided, you need to specify --data <path_to_data_config_file>")
        print("usage: python3 predict_annotate_video.py --data <data_config_file> --data <data_config_file> --predicted <path_to_predicted> --output <output_dir_for_videos>")
        exit()

    # Creates folder for output results
    if args_in.predicted:
        path_save_predicted = f'{args_in.predicted}'
    else:
        print("No data for prediction is provided, you need to specify --predicted <path_to_predicted>")
        print(
            "<path_to_predicted> file with { id_video: [[label,time_0], [label,time_1], .. , [label,time_end]]")
        print("where id_video: video file id (without .mp4) for each video form --data <dataset_info.json>")
        print("label -> [0,1,2,..,7] predicted emotion id for time_K ")
        exit()

    # Specify file with prediction to save
    # file_predictV2 = f'{path_save_predicted}/predict'
    # print("file_predictV2", file_predictV2)

    # Load prediction
    # yV = ["y_pred", "y_t", "y_id"]
    # [y_V, y_t, y_id] = load_predict(file_predictV2, yV)

    # Assemble prediction for each video
    # profile_P = assemble(y_V, y_t, y_id)  # profile_V[id].append([p_V,t])
    profile_P = loadarg(f'{args_in.predicted}')

    print("profile_P.keys():", profile_P.keys())

    file_predict_list = f'{args["dataset"]["file_predict_list"]}'
    dir_videos = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_videos"]}'
    dir_frames = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_frames"]}'
    dir_audios = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_audios"]}'

    # Creates folder for output results
    if args_in.output:
        args_output = loadarg(args_in.output)
        output_dir = f'{args_output["dir_videos"]}'
        new_dir(output_dir)
        dir_videos_embar = f'{output_dir}/{args["dataset"]["name"]}/videos'
        dir_frames_embar = f'{output_dir}/{args["dataset"]["name"]}/frames'
        new_dir(dir_videos_embar)
        new_dir(dir_frames_embar)
    else:
        dir_videos_embar = f'{dir_videos}_embar'
        dir_frames_embar = f'{dir_frames}_embar'
        new_dir(dir_videos_embar)
        new_dir(dir_frames_embar)

    print("file_predict_list", file_predict_list)
    print("dir_frames", dir_frames)
    print("dir_audios", dir_audios)

    BarNames = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
                4: "Happiness",  5: "Neutral", 6: "Sadness", 7: "Surprise"}
    IonI = IndicatorOnImage(BarNames)  # Title = None

    # indicator_values = {4: 8, 6: 2, 0: 5, 3: 3}

    with open(file_predict_list, 'r') as f:
        ids = f.readlines()

    count = 0
    # Strips the newline character
    window_10 = []
    for id in ids:
        id = id.strip()
        print("id", id)
        file_audio = f'{id}.wav'
        dir_frames_embar_id = f'{dir_frames_embar}/{id}'
        new_dir(dir_frames_embar_id)

        for t in range(2):
            frame_s, frame_e = t * 10, (t + 1) * 10
            stat = {}
            for v in [0, 1, 2, 3, 4, 5, 6, 7]:
                stat[v] = 0

            for f in range(frame_s, frame_e):
                f_str = str(f+1).zfill(6)  # format 102 -> 000102
                file_image = f"{dir_frames}/{id}/{f_str}.jpg"
                IonI.add_on_image(
                    file_image, stat, output_path=f"{dir_frames_embar_id}/{f_str}.jpg")

        for [l, t] in profile_P[id]:
            print("l,t:", l, t)

            frame_s, frame_e = (t+2)*10, ((t+2)+1)*10
            window_10.append(l)
            if len(window_10) > 10:
                del window_10[0]
            stat = {}
            for v in [0, 1, 2, 3, 4, 5, 6, 7]:
                stat[v] = 0
            for v in window_10:
                if v in [0, 1, 2, 3, 4, 5, 6, 7]:
                    stat[v] += 1

            for f in range(frame_s, frame_e):
                f_str = str(f+1).zfill(6)  # format 102 -> 000102
                file_image = f"{dir_frames}/{id}/{f_str}.jpg"
                # print("file_image", file_image)

                # indicator_values = {4: 8, 6: 2, 0: 5, 3: 3}
                IonI.add_on_image(
                    file_image, stat, output_path=f"{dir_frames_embar_id}/{f_str}.jpg")

                #
        cmd_assemble = f"ffmpeg -framerate 10 -i \"{dir_frames_embar_id}/%06d.jpg\" -i {dir_audios}/{file_audio} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \"{dir_videos_embar}/{id}.mp4\" "
        print(cmd_assemble)
        try:
            os.system(cmd_assemble)
        except:
            print(f"An exception occurred \n")

        # exit()


if __name__ == '__main__':
    main()
