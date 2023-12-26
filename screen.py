import subprocess
import os
import time
from datetime import date
import json
import argparse
from lib.utils.utils import loadarg,  AverageMeter, get_num_class
from lib.utils.set_folders import check_rootfolders, get_file_results, define_rootfolders


def mknewfolder(folder):
    """
    Create a new folder if it does not exist.

    Parameters:
    ----------
    folder : str
        The path of the folder to create.

    """
    if os.path.exists(folder):
        print('folder exist ', folder)
    else:
        print('creating folder ' + folder)
        os.mkdir(folder)


def p2str(p1):
    """
    Convert a parameter list to a string representation.

    Parameters:
    ----------
    p1 : list or other
        The parameter to convert to string.

    Returns:
    -------
    str
        A string representation of the parameter.
    """
    p1_str = p1
    if isinstance(p1, list):
        p1_str = "_".join([str(int) for int in p1])
    return p1_str


def main():
    """
    Main function to set up and run a series of experiments.
    This script reads configuration, sets up directories for results, and iterates over a range of parameters to run experiments. 
    It is used for training or evaluating a machine learning model with different configurations.

    """

    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--cuda_ids", type=str)  # 0_1_2_3

    args_in = parser.parse_args()
    args = loadarg(args_in.config_path)

    today = date.today().strftime("%d_%m_%Y")

    screen_id = "screen_" + today
    script = "train.py"

    screen_folder = f'{args["root_folder"]}/{screen_id}'
    mknewfolder(screen_folder)
    screen_folder = f'{args["root_folder"]}/{screen_id}/results'
    mknewfolder(screen_folder)

    file_results_screen = f'{screen_folder}/screen_results.json'

    print("screen_folder:", screen_folder)

    results_screen = {}

    # [ 0, 1, 2, 3, 4 , 5,  7,   8]#[1, 3, 4, 5] # # # [7] #
    screen_p1 = screen_emotion_ids = ["p"]
    screen_p2 = screen_jump = [0.1, 0.5, 1, 2, 3]
    # screen_p2 =  screen_porog = [0.2]
    # inet->0.8, timm 0.5 screen_dropout
    # screen_model_segment_config
    # screen_dataset_n_mels
    # screen_model_net_param_shift_temporal_modality_f_div
    # screen_arch
    # screen_dropout
    # screen_video_transform_param_RandomHorizontalFlip
    # screen_arch
    # screen_consensus_type
    # screen_net_optim_param_epochs_decay_start #
    screen_p3 = screen_dropout = [0.5]
    screen_p4 = screen_lr = [0.1]  # inet-> 0.01, timm->0.1
    screen_p5 = screen_last_pool = [1]
    # , "resnet50_timm","densenet201","resnet50_timm" "densenet201"  #"densenet201"
    screen_p6 = screen_arch = ["resnet50_timm"]
    # , [16,0,0] [8,0,1], [8,0,0], [4,0,1], [4,0,0]
    screen_p7 = screen_segment_config = [[16, 0, 1]]

    # Iterate over all combinations of parameters
    for i1, p1 in enumerate(screen_p1):
        for i2, p2 in enumerate(screen_p2):
            for i3, p3 in enumerate(screen_p3):
                for i4, p4 in enumerate(screen_p4):
                    for i5, p5 in enumerate(screen_p5):
                        for i6, p6 in enumerate(screen_p6):
                            for i7, p7 in enumerate(screen_p7):
                                number_of_runs = 1
                                for rid in range(number_of_runs):

                                    args["TSM"]["video_segments"], args["TSM"]["audio_segments"] = p7[0], p7[2]
                                    if p7[1] > 0:
                                        args["TSM"]["motion"] = True

                                    args["TSM"]["main"]["arch"] = p6
                                    args["TSM"]["main"]["dropout"] = p3
                                    args["net_optim_param"]["lr"] = p4
                                    args["TSM"]["main"]["last_pool"] = p5
                                    args["emotion_jumps"]["jump"] = p2

                                    # args["emotion_jumps"]["emotion_ids"] = p1

                                    screen_id = f'{rid}'
                                    pstr_list = [p2str(p1), p2str(p2), p2str(
                                        p3), p2str(p4), p2str(p5), p2str(p6), p2str(p7)]
                                    for pstr in pstr_list:
                                        screen_id += f'_{pstr}'

                                    args["root_folder"] = f'{screen_folder}'
                                    results_screen[screen_id] = {}

                                    """saving args to screen folder"""
                                    agrs_json = f'{screen_folder}/args.json'
                                    with open(agrs_json, 'w') as f:
                                        json.dump(args, f, indent=4)

                                    cmd = f"python3 {script} --config {agrs_json} --run_id {screen_id} "
                                    if args_in.cuda_ids:
                                        cmd += f" --cuda_ids {args_in.cuda_ids}"

                                    args["run_id"] = screen_id
                                    args = define_rootfolders(args)
                                    print("result.json:",
                                          f'{args["output_folder"]}/result.json')

                                    print("cmd:", cmd)
                                    # exit()

                                    os.system(cmd)
                                    continue

                                    file_results = get_file_results(args)
                                    print("open:", file_results)
                                    with open(file_results) as f:
                                        results = json.load(f)
                                        results_screen[screen_id] = results["best_score"]

                                    with open(file_results_screen, 'w') as f:
                                        json.dump(results_screen, f, indent=4)
                                    print("file_results_screen",
                                          file_results_screen)

    # Save the final results
    with open(file_results_screen, 'w') as f:
        json.dump(results_screen, f, indent=4)


if __name__ == '__main__':
    main()
