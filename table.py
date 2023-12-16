import subprocess
import os
import time

import json
import argparse
from lib.utils.utils import loadarg,  AverageMeter , get_num_class
from lib.utils.set_folders import check_rootfolders , get_file_results , define_rootfolders


def mknewfolder(folder):
    if os.path.exists(folder):
        print('folder exist ', folder)
    else:
        print('creating folder ' + folder)
        os.mkdir(folder)



def main():

    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_ids", type=str)  # 0_1_2_3
    args_in = parser.parse_args()




    script = "train.py"
    agrs_json = "config/adcumen1_ortigia_BG.json"

    ##########################################################
    screen_id = "BG4000"

    dirs_p1 = [

        ["RGB", "4", "imagenet",
         "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_4_0_0_adcumen"],
        ["RGB_audio", "4", "imagenet",
         "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_4_0_1_adcumen"],
        ["RGB", "8", "imagenet",
         "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_8_0_0_adcumen"],
        ["RGB_audio", "8", "imagenet",
         "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_8_0_1_adcumen"],

        ["RGB", "16", "imagenet",
         "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_0_adcumen"],
        ["RGB_audio", "16", "imagenet",
         "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_1_adcumen"],

        ["RGB", "4", "INET21K",  "logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_4_0_0_adcumen"],
        ["RGB_audio", "4", "INET21K",  "logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_4_0_1_adcumen"],
        ["RGB", "8", "INET21K",     "logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_8_0_0_adcumen"],
        ["RGB_audio", "8", "INET21K", "logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_8_0_1_adcumen"],

        ["RGB", "16", "INET21K",     "logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_0_adcumen"],
        ["RGB_audio","16", "INET21K","logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen"],

    ]

    ##########################################################
    screen_id = "BG0"
    dirs_p1 = [

        ["RGB", "4", "imagenet",
         "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_4_0_0_adcumen"],
        ["RGB_audio", "4", "imagenet",
         "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_4_0_1_adcumen"],
        ["RGB", "8", "imagenet",
         "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_8_0_0_adcumen"],
        ["RGB_audio", "8", "imagenet",
         "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_8_0_1_adcumen"],

        ["RGB", "16", "imagenet",
         "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_0_adcumen"],
        ["RGB_audio", "16", "imagenet",
         "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_1_adcumen"],

        ["RGB", "4", "INET21K",
         "logs/screen_multiclass_may/ad_timm_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_8_0_0_adcumen"],
        ["RGB_audio", "4", "INET21K",
         "logs/screen_multiclass_may/ad_timm_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_8_0_1_adcumen"],
        ["RGB", "8", "INET21K",
         "logs/screen_multiclass_may/ad_timm_4f_8f_16f_j05_BG0/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_8_0_0_adcumen"],
        ["RGB_audio", "8", "INET21K",
         "logs/screen_multiclass_may/ad_timm_4f_8f_16f_j05_BG0/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_8_0_1_adcumen"],

        ["RGB", "16", "INET21K",
         "logs/screen_multiclass_may/ad_timm_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_0_adcumen"],
        ["RGB_audio", "16", "INET21K",
         "logs/screen_multiclass_may/ad_timm_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen"],

    ]

    results_screen = {}
    file_results_screen = f'tmp/results/{screen_id}.all'

    for i,[modality, frames, pretrained, dir] in enumerate(dirs_p1):
        results_screen[i] = {}

        run_id = f"{screen_id}_{modality}_{frames}_{pretrained}"
        cmd = f"python3 {script} --config {agrs_json} --run_id {run_id} --cuda_ids {args_in.cuda_ids} --model {dir}"
        print("cmd\n",cmd)

        os.system(cmd)


        mode_train_val = ["validation", "test"]
        for mode in mode_train_val:
            file_results = f'tmp/results/{mode}_{run_id}'
            print("open:", file_results)
            with open(file_results) as f:
                results = json.load(f)
                results_screen[i][mode] = [modality, frames, pretrained, results]

        #break


    with open(file_results_screen, 'w') as f:
        json.dump(results_screen, f, indent=4)

if __name__ == '__main__':
    main()