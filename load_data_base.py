"""
load_data_base.py

Overview:
This script is designed to load and process a video database for use in machine learning models. It includes functions to 
parse command-line arguments, load configuration files, and process the video database according to specified parameters. 
The script is typically used as a preliminary step in a larger pipeline, preparing video data for training or evaluation.

Functions:
- parse_arguments: Parses command-line arguments for the script.
- load_configuration: Loads and validates the configuration file specified by the user.
- process_video_database: Loads and processes the video database based on the provided configuration.
- main: The main function orchestrating the loading and processing of the video database.

Usage:
The script is typically run from the command line with an argument specifying the configuration file. For example:
python load_data_base.py --config path/to/config.json

The configuration file should specify paths and parameters for the video database, including where to find the video
data and any preprocessing or filtering steps required. The script then processes the video database accordingly and
prepares it for further use in the machine learning pipeline.
"""


import argparse
import os
import logging
from lib.utils.utils import loadarg, load_pickle
from mvlib.mvideo_lib import VideoDB

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    -------
    args : Namespace
        The arguments namespace containing all command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load and process a video database.")
    parser.add_argument("--config", type=str,
                        help="Path to the configuration file.", required=True)
    args = parser.parse_args()
    return args


def load_configuration(config_path):
    """
    Load and validate the configuration file.

    Parameters:
    ----------
    config_path : str
        Path to the configuration file.

    Returns:
    -------
    config : dict
        The loaded and validated configuration dictionary.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file does not exist: {config_path}")
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}")

    config = loadarg(config_path)

    # Validate necessary keys in config
    required_keys = ["dataset"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            raise KeyError(f"Missing required config key: {key}")

    return config


def process_video_database(config):
    """
    Load and process the video database based on the provided configuration.

    Parameters:
    ----------
    config : dict
        The configuration dictionary containing necessary paths and parameters.

    Returns:
    -------
    None
    """
    # Extracting file paths and directories from the configuration
    dataset_config = config["dataset"]
    video_db_file = dataset_config["fileVDB"]

    if not os.path.exists(video_db_file):
        logging.error(f"Video database file does not exist: {video_db_file}")
        raise FileNotFoundError(
            f"Video database file does not exist: {video_db_file}")

    video_db = load_pickle(video_db_file)
    # Example of a processing step, adjust as needed
    video_db.filtrMarket(market_filtr=826)

    # Additional processing steps can be added here

    logging.info("Video database loaded and processed successfully.")


def main():
    """
    Main function to orchestrate loading and processing a video database.
    """
    try:
        args = parse_arguments()
        config = load_configuration(args.config)
        process_video_database(config)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
