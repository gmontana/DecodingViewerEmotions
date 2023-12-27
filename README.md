# Adcumen project

Scripts for training/prediction and compute descriptive statistics of adcumen data.

## Preprocessing of Adcumen data 

This repo contains scripts to parse original adcumen data:

- create_data_base.py: parse original data from "DataAdcumen/advert_respondent_data.csv" and "DataAdcumen/DatasetAdcumenStarR.csv 

- load_data_base.py:  convert videos from the database into frame and audio  (needs output from create_data_base.py) 

- plots.py: creates all "Descriptive statistics" used in the manuscript (needs output from create_data_base.py) 

All scripts need configuration from config/adcumen**.json to specify paths to input and output files, also parametes like "jump": 0.5 and "clip_length": 5; which means top 0.5 percentile and clip duration 5 seconds.

## Configuration    

Configuration file: config/adcumen**.json specify:

- "dataset": paths to data

- "emotion_jumps": parameters of emotion jumps

- "TSM": parameters of video classification model (to train/predict)

- "root_folder": paths to save trained model weights and parameters

## Training

- train.py: to train video classification model or to validate video classification model (if option --model <path_to_model> is specified)

The training function is also called by the screen.py script which is used to set up different experiments.

## Prediction

- predict.py: to make full video prediction (whether video has at least one emotion jump of a given kind )

An example of using predict.py:
`` 
python3 predict.py --data config/adcumen1_ortigia.json  --id f16_BG0_RGB_audio_16_INET21K --type test --model logs/screen_multiclass_may/ad_timm_16f_j05_BG0/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen
``

where the parameters are:

    --data (config/adcumen1_ortigia.json): This parameter specifies the path to the JSON configuration file that contains settings or data definitions needed for predictions. This file typically includes paths to datasets, preprocessing details, and other relevant information required to set up the data for prediction.

    --id (f16_BG0_RGB_audio_16_INET21K): This is an identifier for the prediction task. It might be used to name output files or directories, making it easier to track and organize results, especially when making predictions on different sets or types of data.

    --type (test): This parameter specifies the mode or type of prediction being run. Common types might include test, validation, or predict. Each type might correspond to different data sets or different modes of operation in the script. For example, test might use a held-out portion of the data to evaluate the model's performance.

    --model (logs/screen_multiclass_may/ad_timm_16f_j05_BG0/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen): This parameter specifies the path to the directory containing the trained model and its associated files, like weights (checkpoint) and configuration (args.json). The script will load this model for making predictions. The path typically includes the directory where the model is saved, and it's essential that this directory contains all necessary files for loading the model.

- predict_annotate_video.py: to make predictions at the full video level


## Automaation

- screen.py: automate the training

An example is

```
python3 screen.py --config config/adcumen1_ortigia.json --cuda_ids 0_1_2_3
```

this would screen jumps 0.1 0.5 1 2 3 percentiles

Results would be stored in folder:  logs/screen_23_01_2023/results


- screen_predict.py: automate the predictions
