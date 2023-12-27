# Data files

DatasetAdcumenSBS.csv 

    The file DatasetAdcumenSBS.csv is related to the emotion data used in the create_data_base.py script.

    Columns:
        - AdvertId: Identifier for the video advert
        - AdvertResultId: Possibly an identifier for a specific viewing session or result.
        - Second: Indicates the specific second in the video this data row corresponds to.
        - Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise: These columns represent different emotions, likely indicating the percentage or degree to which each emotion is expressed or perceived at that particular second in the video.

    This file seems to be a detailed breakdown of emotional responses to specific segments (seconds) of videos. Each row corresponds to a one-second interval of a video, providing a distribution of emotional responses for that interval.
    The data format aligns with what might be used in the create_data_base.py script for processing emotion-related information. The script could use this data to enhance the video database (VDB) with detailed, time-specific emotional responses.

    Usage in create_data_base.py:
    While the script does not explicitly detail how it processes this file, it likely uses the emotion data to compute aggregated profiles and identify significant emotional changes (jumps) in video content.
    The add_Emotions method in the script might be responsible for integrating this emotion data into the VDB.

DatasetAdcumenStarR.csv

    Columns:
        - AdvertId: Identifier for the video or advertisement.
        - AdvertResultId: Identifier for a specific viewing session or result.
        - MarketId: Market identifier, possibly indicating the target audience or region.
        - Title: Title of the video or advertisement.
        - Duration: Duration of the video in seconds.
        - Media: URL to the video content.
        - AdRecordCreatedDate: Date when the video record was created.
        - DecimalStarRating: A rating for the video, possibly averaged from viewer feedback.
        - 9-16. AngerPercentage, ContemptPercentage, DisgustPercentage, FearPercentage, HappinessPercentage, NeutralPercentage, SadnessPercentage, SurprisePercentage: These columns represent the average percentage of each emotion expressed or perceived for the entire video.

    Interpretation:
    This file provides a detailed overview of each video, including metadata (like title, duration, and URL) and aggregated emotional responses.
    The emotional data seems to be averaged over the entire duration of the video, providing a general sense of how viewers reacted to the content.
    Usage in the "adcumen" Project:
    While create_data_base.py does not explicitly reference this file, the data could be valuable for broader analyses, such as understanding overall viewer sentiment, comparing different videos, or correlating video characteristics with emotional responses.
    The file could be used in other parts of the project for tasks like generating reports, comparing viewer reactions across different markets, or evaluating the impact of video content.

advert_respondent_data.csv

    Columns:
        - RespondentAnswerFaceTracingId: Likely a unique identifier for each response or data entry.
        - RespondentID: Identifier for the individual respondent.
        - advertId: Identifier for the specific advertisement being responded to.
        - EmotionString: A string encoding the emotional response, seemingly structured as a series of segments where each segment contains a time offset, an emotion identifier, and possibly other data.

    Interpretation:
    EmotionString Format: The EmotionString column seems to encode time-specific emotional responses. For example, 0.00#6#|10.89#8#| might mean that at 0.00 seconds, the emotion with ID 6 was recorded, and at 10.89 seconds, the emotion with ID 8 was recorded. The emotion IDs likely correspond to specific emotions (e.g., happiness, sadness, etc.).
    This file provides a detailed mapping of how individual respondents reacted emotionally to different advertisements over time.

    Usage in the "adcumen" Project:
    While the create_data_base.py script does not explicitly reference this file, it could be integral to other parts of the project that analyze respondent-level emotional reactions to video content.
    The data could be used to understand how emotional responses vary over the course of an advertisement and differ among respondents.
    It might also be useful for aggregating and analyzing viewer reactions at specific moments in a video, which could inform content creation and marketing strategies.

videos

    A folder with all the videos

# Data preprocessing

create_data_base.py

    Parses original adcumen data from specified CSV files and processes them into a database format. It handles video data, individual profiles, and aggregates profiles for further analysis.

    This script is crucial for preprocessing and structuring video data, making it ready for tasks like training machine learning models or further analysis. 
    The VDB file it creates serves as a comprehensive database of processed video data.

    Functionality:
    - Initializes a VideoDB object and processes various input files to populate it with relevant data.
    - Parses video descriptions and individual profiles, computes aggregated profiles, adds emotion data, and identifies positive and negative samples.
    - Computes differential aggregated profile time-variant (dAPTV) data and identifies positive and negative samples based on a specified jump parameter.

    Jump Detection:
    The script uses the get_dAPTV and get_dAPTV_porogs methods of the VideoDB class to compute dAPTV data and identify jumps.
    A "jump" is detected as a 5-second segment containing a high proportion of viewers who have selected a particular emotion. This is determined by the jump parameter and the dAPTV computation.

    Computing dAPTV (get_dAPTV Method):
        Functionality: This method computes the Differential Aggregated Profile Time-Variant (dAPTV) data.
        Process: It likely involves analyzing the change in aggregated emotional responses over time. The clip_length parameter (set to 5 in the script) specifies the length of video segments for this computation. This means the emotional response is aggregated and analyzed in 5-second intervals throughout the video.

    Identifying Jumps (get_dAPTV_porogs Method):
        Functionality: This method is used to identify significant changes or "jumps" in the dAPTV data.
        Process: The jump parameter (set to 0.5 in the script) likely represents a threshold or criteria to determine what constitutes a significant change in emotional response. The method probably compares the dAPTV data against this threshold to identify moments where there is a substantial shift in viewer emotions.
        Type Parameter: The type="top" argument suggests that the method is specifically looking for the top or most significant changes, possibly the moments with the highest emotional response or the largest shifts in emotion.

    Positive and Negative ID Identification:
        After identifying jumps, the script uses get_positive_ID and get_negative_ID methods to categorize these jumps as positive or negative emotional responses.

    Input Files and Fields:
    - fileDescriptionVideos: Contains descriptions of videos. The script uses this file to parse and add video information to the VDB.
    - fileIndividualProfiles: Contains individual profiles data. The script processes this file to build aggregated profiles.
    - fileVDB: The path where the processed VDB will be saved.

    VDB Structure:
    - The processed VDB is saved as a pickle file at the location specified by fileVDB.
    - The VDB file contains structured data including video descriptions, individual profiles, aggregated profiles, emotion data, and identifiers for positive and negative samples.
    - The VDB (Video Database) is a structured representation of video data, including descriptions, profiles, and computed features like aggregated profiles and emotion data.
    - It includes methods for parsing video descriptions, individual profiles, computing differential aggregated profile time-variant (dAPTV) data, and identifying positive and negative samples based on a specified jump parameter.
    - The VDB is enhanced with emotion data and includes methods for handling this emotion-related information.

    Output:
    -The primary output is the processed VDB, saved as a pickle file at the location specified by fileVDB.
    -This VDB file contains all the processed and structured data ready for use in further analysis or model training.

    Additional Details:
    The script sets specific parameters like clip_length and jump for processing the VDB.
    It uses these parameters to compute dAPTV and identify positive and negative samples.

    Sequence of Operations:
        Initialization: Parses command-line arguments to get the configuration file path.
        Loading Configuration: Uses loadarg to load the configuration settings from the provided file.
        File Paths Setup: Sets up paths for video descriptions (fileDescriptionVideos), individual profiles (fileIndividualProfiles), and the output VDB file (fileVDB). Also sets paths for directories containing videos, frames, and audios.
        VideoDB Initialization: Initializes a VideoDB object with the paths to the video descriptions and individual profiles.
        Parsing Video Descriptions: Calls parseDescriptionVideos to process the video descriptions file.
        Adding Video Directory Path: Adds the path to the video directory to the VDB.
        Parsing Individual Profiles: Processes individual profiles using parseIndividualProfiles.
        Aggregated Profiles Computation: Computes aggregated profiles with get_aggregated_profiles.
        Adding Emotions: Enhances the VDB with emotion data using add_Emotions.
        Computing dAPTV: Computes differential aggregated profile time-variant (dAPTV) data with a specified clip_length.
        Identifying Jumps: Computes dAPTV porogs (thresholds) with a specified jump parameter and identifies positive and negative samples.
        Saving VDB: Saves the processed VDB as a pickle file at the location specified by fileVDB.

    Parameters:
        Configuration File: Specified via command-line argument --config. Contains paths and settings for processing the VDB.
        clip_length: Used in get_dAPTV to define the length of video clips for dAPTV computation.
        jump: Used in get_dAPTV_porogs to set the threshold for identifying significant changes in aggregated profiles (jumps).

    Output:
        The primary output is the processed VDB, saved as a pickle file at the location specified in the configuration file.

# Data loading

load_data_base.py

    This script is essential for loading and updating a video database with additional processing, making it ready for tasks like training machine learning models or further analysis.

    Input:
    - A configuration file specifying paths to the video database, video descriptions, individual profiles, and directories for videos, frames, and audios.
    - Command-line argument to specify the configuration file.

    Output:
    - Processes the video database to update it with additional information and possibly convert videos into frames and audio.
    - The updated VDB is not explicitly saved in this script, but it's prepared for further use.

    Initialization: The script starts by parsing the configuration file to get paths and parameters.
    Loading VideoDB: It loads an existing VideoDB object from a pickle file.

    Data Processing:
    - Filters the VideoDB based on a market filter.
    - Retrieves aggregated profiles and adds emotion data to the VideoDB.
    - Computes differential aggregated profile time-variant (dAPTV) data with specified clip length and jump parameters.
    - Identifies positive and negative samples based on the computed dAPTV.

    Video Conversion: The script checks for the existence of frame and audio directories and, if they don't exist, creates them. It then parses the video set to convert videos into frames and audio files.

    Recomputing VDB: Optionally, the script can recompute the VDB for new clip lengths or jump parameters, updating the positive and negative IDs.

# Video editing example

edit_video/example_ffmpeg_edit_video.py

    An example script demonstrating how to use FFmpeg for video editing tasks such as extracting frames and audio from a video file.

# Library - dataset

lib/dataset/audio.py 

    Input: Audio files, sampling rate, and various parameters for audio processing (like number of segments, window sizes, etc.).
    Output: Processed audio data in the form of spectrograms suitable for input into machine learning models.

    This file contains functions for processing audio data. It includes methods for shifting waveforms, converting waveforms to spectrograms, and augmenting audio data.

lib/dataset/dataset.py

    Input: Input file containing data records, configuration parameters for the dataset, and mode (training, validation, etc.).
    Output: A dataset object that can be used by PyTorch data loaders for model training and validation.

    Defines the MultiJumpDataSet class, which is a custom dataset class for handling multi-jump data in training and validation. It includes methods for parsing input files, handling positive and negative data records, and dataset utilities.

lib/dataset/dataset_predict.py 

    Input: Input file containing data records for prediction, configuration parameters for the dataset.
    Output: A dataset object tailored for prediction tasks, suitable for use with PyTorch data loaders.

    Defines the PredictFullVideoDataSet class for handling datasets specifically for prediction tasks. It includes methods for parsing input files and preparing data for prediction.

lib/dataset/video.py 

    This script plays a crucial role in the preprocessing pipeline for video data, ensuring that the input to the model is in the correct format and has undergone necessary transformations to enhance model performance.
    This file contains functions for loading and transforming video data into a format suitable for machine learning models.
    It includes methods for loading images from video frames, applying various augmentations (like random cropping, resizing, flipping, color jittering, grayscale conversion, and Gaussian blur), and preparing data for training and validation.
    The script also provides functions for sampling indices from video frames for training and validation, which is crucial for handling video segments.

    Input:
    -Video data in the form of directories containing video frames.
    -Parameters for video processing, such as the number of segments, frames per segment, clip length, and augmentation parameters.
    -Mode of operation (training or validation), which influences how the video data is processed and augmented.

    Output:
    -Processed video data in the form of tensors, ready to be fed into a machine learning model.
    -The output is a multi-dimensional tensor representing the processed video segments, which can be used for training or validating video-based models.


# Library - model

These files collectively provide the necessary components for building a comprehensive neural network model capable of handling video and audio data, with a focus on video classification tasks.

lib/model/backbone.py

    Defines the BackBone class, which is responsible for creating the base model of the neural network. It includes methods for loading model weights, inserting temporal shifts, and handling attention mechanisms.

    Input: Model parameters, including architecture details and temporal shift parameters.
    Output: A backbone model that can be integrated into a larger neural network architecture.

lib/model/model.py

    Defines the VCM class, a neural network model for video classification. It integrates various components like input processing, backbone network, and a final classification layer.

    Input: Video and audio data, along with model parameters.
    Output: Predictions from the neural network.

lib/model/motion.py

    Contains the MDFM class for motion detection and feature extraction from video frames. It includes methods for applying Gaussian and sharpening filters, and handling motion-based features.

    Input: Video frame data and motion detection parameters.
    Output: Processed video data with motion features extracted.

lib/model/prepare_input.py

    Defines the X_input class for preparing input data for the model. It handles the integration of video, audio, and motion data into a unified format.

    Input: Video and audio data, along with parameters for handling motion data.
    Output: A combined tensor of video, audio, and motion data ready for input into the model.

lib/model/temporal_fusion.py

    Input:
    -Neural network layers to which the shifts will be applied.
    -Parameters for the shifts, including the number of video, audio, and motion segments, input mode, division factor for features (f_div), and depth of the shift.
    -Mode of operation (e.g., shift_temporal, shift_temporal_modality).

    Output:
    -Modified neural network layers with temporal and modality shifts integrated.
    -These shifts enable the network to process temporal information more effectively and integrate features from different modalities (like video and audio).

    Defines the TempoModalShift class, which is responsible for applying temporal and modality shifts to the input features of a neural network layer.
    The script includes methods for shifting features in time (shift_temporal) and modality (shift_temporal_modality), allowing the model to better capture temporal dynamics and cross-modality interactions.
    It also provides a utility function make_Shift to apply these shifts to different layers of a network, such as ResNet or DenseNet.

# Library - utilities 

The lib/utils directory in the "adcumen" repository contains several utility scripts, each serving a specific purpose in the machine learning pipeline. 

Each of these scripts plays a crucial role in the overall functionality of the machine learning pipeline, providing necessary tools for configuration, reporting, model management, and utility operations.

config_args.py

    Handles the adjustment and configuration of arguments used throughout the model's pipeline.
    Input: User-defined arguments and default settings.
    Output: Adjusted and configured arguments for use in the model.

report.py

    Provides functions for reporting model parameters and epoch results.
    Input: Model arguments, training/validation scores, and best scores.
    Output: Printed reports of model configurations and training/validation performance.

saveloadmodels.py

    Includes functions for saving and loading model states, handling checkpoints, and printing model layers.
    Input: Model, optimizer, checkpoint paths, and model states.
    Output: Loaded or saved model states, and printed information about model layers.

set_folders.py

    Manages the creation and organization of directories for storing model outputs and checkpoints.
    Input: Model arguments and directory paths.
    Output: Created or verified directories and stored configuration files.

set_gpu.py

    Configures GPU settings for the model, including setting up DataParallel for multi-GPU training.
    Input: Model arguments and CUDA device IDs.
    Output: Configured model for specified GPU settings.

utils.py

    A collection of utility functions for tasks like saving/loading arguments, creating directories, calculating metrics, and more.
    Input: Various inputs depending on the function, including model arguments, file paths, and dataset-specific information.
    Output: Various outputs including updated average values, saved/loaded configurations, directory paths, class counts, accuracy scores, and confusion matrices.

# Training 

train.py

    Input:
    -A configuration file specifying model parameters, training settings, and dataset paths.
    -Command-line arguments for model directory, CUDA device IDs, and other training options.

    Output:
    -A trained model.
    -Performance metrics such as accuracy and loss.
    -Checkpoint files containing the trained model's state.

    Model Initialization: The script initializes the model (VCM) with components like input processing (X_input) and backbone network (BackBone). The model is set up with CUDA devices and wrapped in a DataParallel module if necessary.
    Optimizer and Criterion: It sets up an optimizer (SGD) and a loss criterion (CrossEntropyLoss).
    Data Loading: The script prepares datasets for training, validation, and testing using GetDataSet and GetDataLoaders. It handles different scenarios, such as fixed thresholds for data selection.
    Training Process: The main training loop iterates over epochs. For each epoch, it:
    Runs a training step on the training dataset.
    Validates the model on the validation dataset.
    Optionally, tests the model on a test dataset.
    Computes metrics like accuracy and confusion matrix.
    Saves checkpoints and updates the best model based on validation accuracy.

    Functions:
    run_epoch: Orchestrates the training, validation, and testing for a single epoch.
    train: Handles the training process for each batch, including forward and backward passes.
    validate: Evaluates the model on the validation or test dataset.
    validate_external: An external validation function for additional validation scenarios.
    Results Saving: The script saves training history and best scores in a JSON file.

# Prediction
`
predict.py

    Input:
    -A configuration file specifying the dataset for prediction.
    -A trained model directory containing the model weights and configuration.
    -Command-line arguments for specifying the type of data (validation, test, predict), model directory, and CUDA device IDs.

    Output:
    -Predictions for each video in the dataset, including predicted labels, timestamps, and video IDs.
    -Assembled profiles for each video based on the predictions.
    -The predictions and profiles are saved in specified output directories.

    The script starts by parsing command-line arguments to get the model and dataset configurations.
    It initializes the model (VCM) with the appropriate input processing (X_input) and backbone network (BackBone).
    The model is loaded with pre-trained weights from a checkpoint file.
    A dataset for prediction is prepared using GetDataSetPredict and GetDataLoadersPredict.
    The predictV2 function is called to make predictions on the dataset. This function processes each batch of data, performs inference using the model, and collects the predictions.
    The predictions are saved using the save_predict function.
    Finally, the script assembles the predictions for each video, creating a profile that includes the predicted labels and timestamps. This profile is saved in a JSON file.

predict_annotate_video.py

    This script is designed for annotating videos with predicted emotions.
    It loads predictions and overlays them on video frames, creating a visual representation of the model's predictions.
    The script also handles the assembly of video frames into annotated videos.

    Input:
    - Command-line arguments for specifying the dataset, predicted data, and output directory.
    - Predicted emotion data for each video.

    Output:
    - Annotated video frames and assembled videos with emotion indicators overlaid.


# Automation of experiments

screen.py

    This script is designed to automate the process of running multiple training experiments with different configurations.
    It sets up various parameters for the experiments, such as dropout rates, learning rates, and architecture configurations.
    The script then iteratively runs the train.py script with these different configurations, storing the results in a specified directory.

    Input:
    - Command-line arguments for configuration path and CUDA IDs.
    - A JSON configuration file that specifies various training parameters.
    Output:
    - The script generates multiple training runs with different configurations and stores their results in a structured format.
    - Outputs include training results for each configuration, saved in a specified results directory.

predict_screen.py

    This script automates the execution of the predict.py script with various configurations.
    It sets up different parameters for prediction experiments, such as modality, number of frames, and pretrained model types.
    The script runs the prediction script with these configurations and stores the results.

    Input:
    - Command-line arguments for CUDA IDs and configuration files.
    - Predefined configurations within the script for different prediction setups.

    Output:
    - The script generates multiple prediction runs with different configurations and collects their results.

# Statistical analyses 

plots.py

    This script is used for generating plots and visualizations based on the model's performance and dataset characteristics.
    It includes functions for copying frames from videos, generating statistics from the video database, and plotting ROC curves.
    The script can handle various types of data visualizations, including distribution of clicks, signal-to-noise distributions, and aggregated profile visualizations.

    Input:
    - Command-line arguments for configuration files and specific modes of operation.
    - A configuration file that specifies dataset and model parameters.

    Output:
    - Various plots and visualizations based on the input data and model performance.
    - The script can output ROC curves, statistical distributions, and other relevant visualizations for analyzing the model and dataset.

table.py

    This script is crucial for conducting extensive experiments with different model configurations, facilitating a comprehensive analysis of model performance across various settings.

    This script automates the execution of the train.py script with various configurations and collects the results.
    It sets up different parameters for the experiments, such as modality (RGB or RGB_audio), number of frames, and pretrained model types (Imagenet or INET21K).
    The script iteratively runs the training script with these configurations and stores the results for further analysis.

    Input:
    - Command-line arguments for CUDA IDs.
    - Predefined configurations within the script for different model setups.

    Output:
    - The script generates multiple training runs with different configurations and collects their results.
    - Outputs include training results for each configuration, saved in a structured format in a specified results directory.

predict_stat.py

    This script is used for generating statistical analysis of prediction results.
    It processes predicted emotion data and compiles statistics for each video.
    The script also formats these statistics for presentation, such as in a LaTeX table format.

    Input:
    - Command-line arguments for specifying the predicted data and additional information.
    - Predicted emotion data and video information.

    Output:
    - Statistical analysis of the predicted emotions, formatted for presentation or further analysis.

