{
    "help": "General description or comments about the configuration file.",
    
    "emotion_jumps": {
        "emotion_ids": "List of integers representing emotion IDs.",
        "clip_length": "Integer representing the length of the clip in seconds.",
        "jump": "Float representing the jump size for clip processing.",
        "porog": "Float representing the threshold value for some processing step.",
        "background_size": "Integer representing the size of the background. Use -1 for default or unspecified size."
    },

    "dataset": {
        "name": "Name of the dataset, e.g., 'adcumen'.",
        "data_dir": "Base directory for all data related to the dataset.",
        "dir_videos": "Subdirectory within data_dir containing video files.",
        "dir_frames": "Subdirectory within data_dir containing frame data.",
        "dir_audios": "Subdirectory within data_dir containing audio files.",
        "dirDataAdcumen": "Subdirectory for specific adcumen data.",
        "fileDescriptionVideos": "Path to the file containing video descriptions.",
        "fileIndividualProfiles": "Path to the file containing individual profiles.",
        "fileVDB": "Path to the file where the video database (VDB) will be saved.",
        "file_train_list": "Path to the file containing training set list.",
        "file_val_list": "Path to the file containing validation set list.",
        "file_test_list": "Path to the file containing test set list.",
        "video_img_param": "Dictionary containing parameters related to video image processing.",
        "video_augmentation": "Dictionary containing settings for video data augmentation.",
        "audio_img_param": "Dictionary containing parameters related to audio image processing.",
        "audio_augmentation": "Dictionary containing settings for audio data augmentation.",
        "fps": "Integer representing frames per second for video processing.",
        "AdcumenDB": "Path to the Adcumen database file."
    },

    "TSM": {
        "video_segments": "Number of segments into which a video is divided.",
        "audio_segments": "Number of segments into which audio is divided.",
        "motion": "Boolean indicating whether motion processing is enabled.",
        "main": "Dictionary containing main architecture parameters for the model.",
        "shift_temporal": "Dictionary containing parameters for temporal shift operations.",
        "shift_temporal_modality": "Dictionary containing parameters for temporal modality shift operations.",
        "shift_spatial": "Dictionary containing parameters for spatial shift operations.",
        "motion_param": "Dictionary containing parameters related to motion processing."
    },

    "net_run_param": {
        "epochs": "Number of training epochs.",
        "batch_size": "Size of each training batch.",
        "num_workers": "Number of workers for data loading."
    },

    "net_optim_param": {
        "lr": "Learning rate for the optimizer.",
        "lr_decay": "List representing learning rate decay schedule.",
        "momentum": "Momentum for the optimizer.",
        "gd": "Gradient descent parameter.",
        "weight_decay": "Weight decay for regularization."
    },

    "save_epoch": "List of epochs at which to save the model.",
    "root_folder": "Root directory for saving logs and outputs."
}

