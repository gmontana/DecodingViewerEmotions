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




    script = "predict.py"
    agrs_json = {"bg0":"config/adcumen1_ortigia.json",  "bg4000":"config/adcumen1_ortigia_BG.json", }

    ##########################################################
    screen_id = "f16"

    dirs_p1 = [

        [f'{agrs_json["bg0"]}', "BG0", "RGB_audio", "16", "imagenet","logs/screen_multiclass_may/ad_inet_4f_8f_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_1_adcumen"],
        [f'{agrs_json["bg4000"]}', "BG4000","RGB_audio", "16", "imagenet","logs/screen_multiclass_may/ad_inet_16f_8f_4f_j05_BG4000/1_1_2_3_4_5_6_7_8_0.5_0.8_0.01_1_resnet50_16_0_1_adcumen"],
        [f'{agrs_json["bg0"]}', "BG0", "RGB_audio", "16", "INET21K","logs/screen_multiclass_may/ad_timm_16f_j05_BG0/1_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen"],
        [f'{agrs_json["bg4000"]}' , "BG4000", "RGB_audio","16", "INET21K","logs/screen_multiclass_may/ad_timm_16f_8f_4f_j05_BG4000/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen"],

    ]

    ##########################################################

    for i,[agrs_file , BG ,modality, frames, pretrained, dir] in enumerate(dirs_p1):


        run_id = f"{screen_id}_{BG}_{modality}_{frames}_{pretrained}"
        cmd = f"python3 {script} --data {agrs_file} --id {run_id} --cuda_ids {args_in.cuda_ids} --model {dir}"



        mode_train_val = ["validation", "test"]
        for mode in mode_train_val:
            cmd_mode = f"{cmd} --type {mode}"
            print("cmd_mode\n", cmd_mode)
            #continue
            #exit()
            os.system(cmd_mode)


        #break




if __name__ == '__main__':
    main()