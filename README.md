# Adcumen project
Scripts for training/prediction and compute descriptive statistics of adcumen data.

## Preprocessing of Adcumen data 
This repo contains scripts to parse original adcumen data:

- create_data_base.py: parse original data from "DataAdcumen/advert_respondent_data.csv" and "DataAdcumen/DatasetAdcumenStarR.csv 
- load_data_base.py:  convert videos from the database into frame and audio  (needs output from create_data_base.py) 
- plots.py: creates all "Descriptive statistics" from the paper (needs output from create_data_base.py) 

All scripts need configuration from config/adcumen**.json to specify paths to input and output files, also parametes like "jump": 0.5 and "clip_length": 5; which means top 0.5 percentile and clip duration 5 seconds.

## Configuration    
Configuration file: config/adcumen**.json specify:

- "dataset": paths to data
- "emotion_jumps": parameters of emotion jumps
- "TSM": parameters of video classification model (to train/predict)
- "root_folder": paths to save trained model weights and parameters


## Training/Predict

- train.py: to train video classification model or to validate video classification model (if option --model <path_to_model> is specified)
- predict.py: to make full video prediction (whether video has at least one emotion jump of a given kind )


## Screen

python3 screen.py --config config/adcumen1_ortigia.json --cuda_ids 0_1_2_3

this would screen jumps 0.1 0.5 1 2 3 percentiles

Results would be stored in folder:  logs/screen_23_01_2023/results


