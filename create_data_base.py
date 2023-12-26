import argparse
import os
import logging
from lib.utils.utils import loadarg
from mvlib.utils import save_pickle
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
        description="Create a video database from descriptions and profiles.")
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


def create_video_database(config):
    """
    Create and populate the VideoDB object.

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
    fileDescriptionVideos = dataset_config["fileDescriptionVideos"]
    fileIndividualProfiles = dataset_config["fileIndividualProfiles"]
    fileVDB = dataset_config["fileVDB"]
    video_dir_path = os.path.join(
        dataset_config["data_dir"], dataset_config["dir_videos"])

    # Creating and populating the VideoDB object
    video_db = VideoDB(fileDescriptionVideos, fileIndividualProfiles)
    video_db.parseDescriptionVideos()
    video_db.add_video_dir_path(video_dir_path)
    video_db.parseIndividualProfiles()
    video_db.get_aggregated_profiles()
    video_db.add_Emotions()

    # Use constants from config or set defaults
    clip_length = config.get("emotion_jumps", {}).get("clip_length", 5)
    jump = config.get("emotion_jumps", {}).get("jump", 0.5)

    video_db.get_dAPTV(clip_length)
    video_db.get_dAPTV_porogs(jump, type="top")
    video_db.get_positive_ID()
    video_db.get_negative_ID()

    save_pickle(fileVDB, video_db)
    logging.info("Video database created and saved successfully.")


def main():
    """
    Main function to orchestrate creating a video database from descriptions and profiles.
    """
    try:
        args = parse_arguments()
        config = load_configuration(args.config)
        create_video_database(config)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise


if __name__ == '__main__':
    main()

