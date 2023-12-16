import argparse
import json
import os
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score , accuracy_score
import warnings
from sklearn.exceptions import DataConversionWarning

import statistics
import random
from collections import Counter


from PIL import Image, ImageDraw
from PIL import ImageFont


from lib.utils.utils import loadarg,  AverageMeter , get_labels , remap_acc_to_emotions, multiclass_accuracy

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

def load_predict(file_predict, *args ):
    print("load_predict", args)
    yV = []
    for keyV in args[0]:
        print("keyV", keyV)
        file_save = f"{file_predict}_{keyV}"
        yV.append(load_pickle(file_save))
    return yV

def save_predict(file_predict,  **kwargs):

    for keyV in kwargs:
        file_save = f"{file_predict}_{keyV}"
        save_pickle(file_save, kwargs[keyV])



def assemble( y_V, y_t, y_id):
    profile_V, stat_V = {}, Counter()
    for i, id in enumerate(y_id):
        p_V, t  = y_V[i],  y_t[i]
        #print("p_V, t", p_V, t)
        if id not in profile_V: profile_V[id] = []
        profile_V[id].append([p_V,t])
        stat_V[p_V] +=1
    print("stat_V", stat_V)

    return profile_V



class IndicatorOnImage:

    def __init__(self,  BarNames, Title=None, scale=1, position = "LL" ):
        self.Title = Title
        self.BarNames = BarNames # dictinary: int ->string

        self.N = len(self.BarNames)


        self.size_Title_H = 0
        if self.Title != None: self.size_Title_H = int(20 * scale)

        self.size = (int(150 * scale), int(15 * self.N * scale + self.size_Title_H))

        self.size_line_H = int(15 * scale)
        self.TitleFont = int(14 * scale)
        self.BarFont = int(10 * scale)

    def add_on_image(self, file_image, indicator_values, output_path = None):
        #indicator_values:  dictinary: int -> int
        if output_path == None: output_path = file_image

        image = Image.open(file_image)
        draw = ImageDraw.Draw(image)
        (W, H) = image.size

        draw.rectangle((0,  H - self.size[1], self.size[0], H), fill="black")

        #myFont = ImageFont.truetype('FreeMono.ttf', 15)
        #draw.text((0,  H - self.size[1]), f"{Title}", (255, 255, 255), font=myFont)

        for i, id in enumerate(indicator_values):
            (x,y, x1,y1) = (0, H - self.size[1] + self.size_Title_H + i * self.size_line_H +10) + (40, H - self.size[1] + self.size_Title_H + i * self.size_line_H)
            value = indicator_values[id]
            name = self.BarNames[id]

            #myFont = ImageFont.truetype('FreeMono.ttf', 13)
            myFont = ImageFont.load_default()
            draw.text((x,y-8),  f"{name}:", (255, 255, 255), font=myFont)
            if value < 3: line_color = (102, 3, 252)#"blue"
            elif value < 5: line_color = 3, 157, 252#""blue2""
            elif value < 7: line_color = (231, 252, 3)#"yellow"
            elif value < 9: line_color = (252, 123, 3)#"orange"
            else: line_color = (252, 15, 3) #"red"
            draw.line((100+x,y, 100+x+5*value,y), width=4, fill=line_color)

        image.save(output_path)




def main():

    global args_model, args_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)    # arg.json file (specifies dataset for prediction)
    parser.add_argument('--predicted', type=str)  # profile_P:  for prediction
    parser.add_argument('--output', type=str)  # <id> for prediction
    args_in = parser.parse_args()

    """ specify dataset for prediction args.json file"""
    if args_in.data:
        args = loadarg(args_in.data)
    else:
        print("No data for prediction is provided, you need to specify --data <path_to_data_config_file>")
        print("usage: python3 predict_annotate_video.py --data <data_config_file> --data <data_config_file> --predicted <path_to_predicted> --output <output_dir_for_videos>")
        exit()

    """ creates folder for output results"""
    if args_in.predicted:
        path_save_predicted = f'{args_in.predicted}'
    else:
        print("No data for prediction is provided, you need to specify --predicted <path_to_predicted>")
        print("<path_to_predicted> file with { id_video: [[label,time_0], [label,time_1], .. , [label,time_end]]")
        print("where id_video: video file id (without .mp4) for each video form --data <dataset_info.json>")
        print("label -> [0,1,2,..,7] predicted emotion id for time_K ")
        exit()


    """ specify file with prediction to save """
    #file_predictV2 = f'{path_save_predicted}/predict'
    #print("file_predictV2", file_predictV2)


    """load prediction"""
    #yV = ["y_pred", "y_t", "y_id"]
    #[y_V, y_t, y_id] = load_predict(file_predictV2, yV)

    """assemble prediction for each video"""
    #profile_P = assemble(y_V, y_t, y_id)  # profile_V[id].append([p_V,t])
    profile_P = loadarg(f'{args_in.predicted}')

    print("profile_P.keys():", profile_P.keys())

    file_predict_list = f'{args["dataset"]["file_predict_list"]}'
    dir_videos = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_videos"]}'
    dir_frames = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_frames"]}'
    dir_audios = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_audios"]}'

    """ creates folder for output results"""
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

    BarNames = {0: "Anger", 1: "Contempt" , 2: "Disgust", 3: "Fear", 4: "Happiness",  5: "Neutral" , 6: "Sadness", 7: "Surprise" }
    IonI = IndicatorOnImage(BarNames)  # Title = None

    #indicator_values = {4: 8, 6: 2, 0: 5, 3: 3}


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
            for v in [0,1,2,3,4,5,6,7]:
                stat[v] = 0

            for f in range(frame_s , frame_e):
                f_str = str(f+1).zfill(6) # format 102 -> 000102
                file_image = f"{dir_frames}/{id}/{f_str}.jpg"
                IonI.add_on_image(file_image, stat, output_path=f"{dir_frames_embar_id}/{f_str}.jpg")

        for [l,t] in profile_P[id]:
            print("l,t:", l,t)

            frame_s , frame_e = (t+2)*10, ((t+2)+1)*10
            window_10.append(l)
            if len(window_10) > 10: del window_10[0]
            stat = {}
            for v in [0,1,2,3,4,5,6,7]:
                stat[v] = 0
            for v in window_10:
                if v in [0,1,2,3,4,5,6,7]:
                    stat[v] +=1

            for f in range(frame_s , frame_e):
                f_str = str(f+1).zfill(6) # format 102 -> 000102
                file_image = f"{dir_frames}/{id}/{f_str}.jpg"
                #print("file_image", file_image)

                # indicator_values = {4: 8, 6: 2, 0: 5, 3: 3}
                IonI.add_on_image(file_image, stat, output_path=f"{dir_frames_embar_id}/{f_str}.jpg")

                #
        cmd_assemble = f"ffmpeg -framerate 10 -i \"{dir_frames_embar_id}/%06d.jpg\" -i {dir_audios}/{file_audio} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \"{dir_videos_embar}/{id}.mp4\" "
        print(cmd_assemble)
        try:
            os.system(cmd_assemble)
        except:
            print(f"An exception occurred \n")

        #exit()

if __name__ == '__main__':
    main()








