
"""
predict_screen.py

Overview:
This script is a utility for automating the execution of predictions across multiple configurations or models. It is designed to run the 'predict.py' script with various settings, iterating over different data configurations, models, and modes. The script is particularly useful for scenarios where predictions need to be made and compared across a range of models or settings, such as in model evaluation or hyperparameter tuning tasks.

The script sets up a series of configurations, each corresponding to a different model or setting, and executes the 'predict.py' script for each configuration. It handles different modes of operation, such as validation and testing, and allows for predictions to be run on specified CUDA devices.

Functions:
- mknewfolder: Creates a new folder if it doesn't exist.
- main: Orchestrates the setup and execution of predictions for each configuration.

Usage:
The script is typically run from the command line with the necessary CUDA IDs specified. It then automatically runs the 'predict.py' script with the predefined configurations and modes, facilitating the batch execution of predictions across multiple settings.

Example Command:
python predict_screen.py --cuda_ids 0_1_2_3

This command would run the script, executing predictions for each configuration in 'dirs_p1' on the specified CUDA devices. The script is useful in batch processing scenarios where multiple predictions need to be run and compared.

Note:
The specific configurations, models, and settings are defined within the script and may need to be adjusted based on the requirements of the prediction tasks or the available models. Ensure that the 'predict.py' script and the specified model directories are correctly set up and accessible.
"""


import subprocess
import os
import time
import json
import argparse
from lib.utils.utils import loadarg,  AverageMeter, get_num_class
from lib.utils.set_folders import check_rootfolders, get_file_results, define_rootfolders


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

    script = "predict.py"
    agrs_json = {"bg0": "config/adcumen1_ortigia.json",
                 "bg4000": "config/adcumen1_ortigia_BG.json", }

    ##########################################################
    screen_id = "f16"

    dirs_p1 = [

        [f'{agrs_json["bg0"]}', "BG0", "RGB_audio", "16", "imagenet",
            "logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_1_adcumen"],
        [f'{agrs_json["bg4000"]}', "BG4000", "RGB_audio", "16", "imagenet",
            "logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_1_adcumen"],
        [f'{agrs_json["bg0"]}', "BG0", "RGB_audio", "16", "INET21K",
            "logs/screen_multiclass_may/ad_timm_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen"],
        [f'{agrs_json["bg4000"]}', "BG4000", "RGB_audio", "16", "INET21K",
            "logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen"],

    ]

    ##########################################################

    for i, [agrs_file, BG, modality, frames, pretrained, dir] in enumerate(dirs_p1):

        run_id = f"{screen_id}_{BG}_{modality}_{frames}_{pretrained}"
        cmd = f"python3 {script} --data {agrs_file} --id {run_id} --cuda_ids {args_in.cuda_ids} --model {dir}"

        mode_train_val = ["validation", "test"]
        for mode in mode_train_val:
            cmd_mode = f"{cmd} --type {mode}"
            print("cmd_mode\n", cmd_mode)
            # continue
            # exit()
            os.system(cmd_mode)

        # break


if __name__ == '__main__':
    main()
