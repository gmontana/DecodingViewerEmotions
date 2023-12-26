import subprocess
import os
import time
from datetime import date
import json
import argparse
from lib.utils.utils import loadarg,  AverageMeter, get_num_class
from lib.utils.set_folders import check_rootfolders, get_file_results, define_rootfolders


def p2str(p1):
    # Function to convert parameter list to string representation
    return "_".join([str(int) for int in p1]) if isinstance(p1, list) else str(p1)


def parse_arguments():
    # Function to parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--cuda_ids", type=str)  # e.g., "0_1_2_3"
    return parser.parse_args()


def mknewfolder(folder):
    # Function to create a new folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)


def setup_directories(args, today):
    # Function to set up directories for saving results
    screen_id = "screen_" + today
    screen_folder_base = os.path.join(args["root_folder"], screen_id)
    mknewfolder(screen_folder_base)
    results_folder = os.path.join(screen_folder_base, 'results')
    mknewfolder(results_folder)
    return screen_folder_base, results_folder


def run_experiment(args, args_in, script, screen_id, results_folder):
    # Function to run an experiment with given parameters
    agrs_json = os.path.join(results_folder, 'args.json')
    with open(agrs_json, 'w') as f:
        json.dump(args, f, indent=4)

    cmd = f"python3 {script} --config {agrs_json} --run_id {screen_id} "
    if args_in.cuda_ids:
        cmd += f" --cuda_ids {args_in.cuda_ids}"

    args["run_id"] = screen_id
    args = define_rootfolders(args)
    os.system(cmd)


def main():
    # Main function to set up and run a series of experiments
    args_in = parse_arguments()
    args = loadarg(args_in.config_path)
    today = date.today().strftime("%d_%m_%Y")
    script = "train.py"

    screen_folder_base, results_folder = setup_directories(args, today)
    file_results_screen = os.path.join(results_folder, 'screen_results.json')

    results_screen = {}

    # Define parameter ranges for the experiments
    screen_p1 = screen_emotion_ids = ["p"]
    screen_p2 = screen_jump = [0.1, 0.5, 1, 2, 3]
    screen_p3 = screen_dropout = [0.5]
    screen_p4 = screen_lr = [0.1]
    screen_p5 = screen_last_pool = [1]
    screen_p6 = screen_arch = ["resnet50_timm"]
    screen_p7 = screen_segment_config = [[16, 0, 1]]

    # Iterate over all combinations of parameters
    for p1 in screen_p1:
        for p2 in screen_p2:
            for p3 in screen_p3:
                for p4 in screen_p4:
                    for p5 in screen_p5:
                        for p6 in screen_p6:
                            for p7 in screen_p7:
                                screen_id = f'{p2str(p1)}_{p2str(p2)}_{p2str(p3)}_{p2str(p4)}_{p2str(p5)}_{p2str(p6)}_{p2str(p7)}'
                                run_experiment(
                                    args, args_in, script, screen_id, results_folder)

    # Save the final results
    with open(file_results_screen, 'w') as f:
        json.dump(results_screen, f, indent=4)


if __name__ == '__main__':
    main()
