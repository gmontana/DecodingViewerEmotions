import shutil
import json
import os
import torch
import torch.nn as nn
from collections import Counter
from re import search



def report_model_param(args):


    print("args.TSM.video_segments:", args["TSM"]["video_segments"])
    print("args.TSM.audio_segments:", args["TSM"]["audio_segments"])
    print("args.TSM.motion:", args["TSM"]["motion"])

    param_TSM_main = args["TSM"]["main"]
    print("param_TSM_main", param_TSM_main)
    print("param_TSM_main shift_temporal", args["TSM"]["shift_temporal"])
    print("param_TSM_main", args["TSM"]["motion_param"])




def report_epoch_results( best_score, val_score , train_score , is_best_score, is_best_score_balanced):
    best_str = f'report_epoch_results\n'
    best_str += f'val_score {round(val_score["unbalanced"]*100, 2)}, val_score_balanced {round(val_score["balanced"]*100,2)}\n'
    best_str += f'train_score {round(train_score["unbalanced"]*100,2)}, train_score_balanced {round(train_score["balanced"]*100,2)}\n'
    best_str += f'best_score {round(best_score["unbalanced"]*100,2)}, best_score_balanced {round(best_score["balanced"]*100,2)}\n'
    best_str += f"is_best_score {is_best_score}\n"
    best_str += f"is_best_score_balanced {is_best_score_balanced}\n"
    print(best_str)


