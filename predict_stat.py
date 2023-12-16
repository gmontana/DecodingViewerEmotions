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


def make_cell_overleaf( v):
    return "\makecell {" + v + "}\n"

def make_cell_overleaf_youtube_link( id, title):
    link = "\makecell{ \href{https://www.youtube.com/watch?v=" + id +"}{" +  title + "}}"
    return link


def main():

    global args_model, args_data

    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str)  # profile_P:  for prediction
    parser.add_argument('--predicted', type=str)  # profile_P:  for prediction
    args_in = parser.parse_args()


    """ creates folder for output results"""
    if args_in.predicted:
        path_save_predicted = f'{args_in.predicted}'
    else:
        print("No data for prediction is provided, you need to specify --predicted <path_to_predicted>")
        print("<path_to_predicted> file with { id_video: [[label,time_0], [label,time_1], .. , [label,time_end]]")
        print("where id_video: video file id (without .mp4) for each video form --data <dataset_info.json>")
        print("label -> [0,1,2,..,7] predicted emotion id for time_K ")
        exit()

    info = loadarg(f'{args_in.info}')

    """load prediction"""
    #yV = ["y_pred", "y_t", "y_id"]
    #[y_V, y_t, y_id] = load_predict(file_predictV2, yV)

    """assemble prediction for each video"""
    #profile_P = assemble(y_V, y_t, y_id)  # profile_V[id].append([p_V,t])
    profile_P = loadarg(f'{args_in.predicted}')

    EIDS = {0: "Anger",
            1: "Contempt",
            2: "Disgust",
            3: "Fear",
            4: "Happiness",
            5: "Neutral",
            6: "Sadness",
            7: "Surprise"
            }

    print("profile_P.keys():", profile_P.keys())

    stat_V , stat_VID = Counter(), {}
    for id in profile_P:
        stat_VID[id] = Counter()
        for [p_V,t] in profile_P[id]:
            stat_VID[id][p_V] += 1
            stat_V[p_V] += 1

    stat_VID_R, tableROWS = {} , []
    for id in profile_P:
        stat_VID_R[id] = Counter()
        #print("stat_VID: ", id, info[id]["title"], stat_VID[id])
        all = 0
        for eid in stat_VID[id]:
            all += stat_VID[id][eid]

        for eid in stat_VID[id]:
            stat_VID_R[id][eid]  = stat_VID[id][eid] / all

        #print("stat_VID_R: ", id, stat_VID_R[id])
        #\href{http://www.overleaf.com}{Something Linky}


        title = info[id]["title"]

        title_link_cell = make_cell_overleaf_youtube_link(id, title)
        count = 0
        top_emotion_str1, top_emotion_str2, top_emotion_str3 = "", "", ""

        for eid, rate in sorted(stat_VID_R[id].items(), key=lambda item: item[1], reverse=True):

            emotion = EIDS[eid]
            rate = str(round(stat_VID_R[id][eid], 2))

            rate4 = str(round(stat_VID_R[id][4], 2))
            rate6 = str(round(stat_VID_R[id][6], 2))

            count +=1
            if count == 1: top_emotion_str1 = emotion + ": " + rate
            if count == 2: top_emotion_str2 = emotion + ": " + rate
            if count == 3: top_emotion_str3 = emotion + ": " + rate

        top_emotion_str1 = make_cell_overleaf(top_emotion_str1)
        top_emotion_str2 = make_cell_overleaf(top_emotion_str2)
        top_emotion_str3 = make_cell_overleaf(top_emotion_str3)

        table_row = title_link_cell + "&\n"

        table_row += top_emotion_str1 + "&\n"
        table_row += top_emotion_str2 + "\n"
        #table_row += top_emotion_str3 + " &\n"

        table_row +=  " \\\\"

        tableROWS.append([table_row , rate6])


    for row in sorted(tableROWS, key=lambda x: x[1], reverse=True):
        print(f"{row[0]} \n")





if __name__ == '__main__':
    main()


#




