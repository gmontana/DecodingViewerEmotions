import os
import json
import argparse
from datetime import date
from itertools import product
from lib.utils.utils import loadarg, define_rootfolders


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    -------
    Namespace
        The namespace containing all command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--cuda_ids", type=str)  # e.g., "0_1_2_3"
    return parser.parse_args()


def mknewfolder(folder):
    """
    Create a new folder if it does not exist.

    Parameters:
    ----------
    folder : str
        The path of the folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def setup_directories(args, today):
    """
    Set up directories for saving results.

    Parameters:
    ----------
    args : dict
        Configuration arguments including paths.
    today : str
        Today's date as a string.

    Returns:
    -------
    tuple
        Tuple containing the base screen folder and results folder paths.
    """
    screen_id = "screen_" + today
    screen_folder_base = os.path.join(args["root_folder"], screen_id)
    mknewfolder(screen_folder_base)
    results_folder = os.path.join(screen_folder_base, 'results')
    mknewfolder(results_folder)
    return screen_folder_base, results_folder


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
    return "_".join([str(int) for int in p1]) if isinstance(p1, list) else str(p1)


def run_experiment(args, args_in, script, screen_id, results_folder):
    """
    Run an experiment with the given parameters.

    Parameters:
    ----------
    args : dict
        Configuration arguments including paths and parameters.
    args_in : Namespace
        Parsed command-line arguments.
    script : str
        The script to execute for the experiment.
    screen_id : str
        Unique identifier for the experiment run.
    results_folder : str
        Path to the folder where results should be saved.
    """
    args_json = os.path.join(results_folder, 'args.json')
    with open(args_json, 'w') as f:
        json.dump(args, f, indent=4)

    cmd = f"python3 {script} --config {args_json} --run_id {screen_id} "
    if args_in.cuda_ids:
        cmd += f" --cuda_ids {args_in.cuda_ids}"

    args["run_id"] = screen_id
    args = define_rootfolders(args)
    os.system(cmd)


def main():
    """
    Main function to set up and run a series of experiments.
    This script reads configuration, sets up directories for results, and iterates over a range of parameters to run experiments. 
    It is used for training or evaluating a machine learning model with different configurations.
    """
    args_in = parse_arguments()
    args = loadarg(args_in.config_path)
    today = date.today().strftime("%d_%m_%Y")
    script = "train.py"

    screen_folder_base, results_folder = setup_directories(args, today)
    file_results_screen = os.path.join(results_folder, 'screen_results.json')

    results_screen = {}

    # Define parameter ranges for the experiments using a dictionary
    experiment_params = {
        "emotion_ids": ["p"],
        "jump": [0.1, 0.5, 1, 2, 3],
        "dropout": [0.5],
        "lr": [0.1],
        "last_pool": [1],
        "arch": ["resnet50_timm"],
        "segment_config": [[16, 0, 1]]
    }

    # Iterate over all combinations of parameters
    for combination in product(*experiment_params.values()):
        emotion_ids, jump, dropout, lr, last_pool, arch, segment_config = combination
        screen_id = f'{p2str(emotion_ids)}_{p2str(jump)}_{p2str(dropout)}_{p2str(lr)}_{p2str(last_pool)}_{p2str(arch)}_{p2str(segment_config)}'
        run_experiment(args, args_in, script, screen_id, results_folder)

    # Save the final results
    with open(file_results_screen, 'w') as f:
        json.dump(results_screen, f, indent=4)


if __name__ == '__main__':
    main()

