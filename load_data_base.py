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

