
"""
This script is used for making predictions with a trained model. It loads a model and its weights, sets up the data
for prediction, and runs the model to make predictions on new data. The predictions are then saved for further analysis.

The script handles different modes of operation such as validation, test, or actual prediction and allows for
customization of various parameters through command-line arguments.

Functions:
- save_pickle: Saves a dictionary object to a file using pickle.
- load_pickle: Loads a dictionary object from a pickle file.
- load_predict: Loads prediction results from files.
- save_predict: Saves prediction results to files.
- predictV2: Makes predictions using the model on the provided data loader.
- run_model: Sets up and runs the model for prediction.
- assemble: Assembles prediction results for analysis.
- main: The main function to run the prediction process.
"""

import argparse
import json
import os
import math
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import warnings
from sklearn.exceptions import DataConversionWarning
import statistics
import random
from collections import Counter
from lib.dataset.dataset import GetDataSet, GetDataLoaders
from lib.dataset.dataset_predict import GetDataSetPredict, GetDataLoadersPredict
from lib.utils.utils import loadarg,  AverageMeter, get_labels, remap_acc_to_emotions, multiclass_accuracy
from lib.utils.config_args import adjust_args_in
from lib.utils.set_gpu import set_model_DataParallel, set_cuda_device
from lib.utils.set_folders import check_rootfolders, get_file_results
from lib.utils.saveloadmodels import checkpoint_acc, save_timepoint, load_model, checkpoint_confusion_matrix
from lib.model.model import VCM
from lib.model.prepare_input import X_input
from lib.model.backbone import BackBone
from lib.model.policy import gradient_policy
import pickle


def save_pickle(file_save, mydict):
    """
    Save a dictionary object to a file using pickle.

    Parameters
    ----------
    file_save : str
        The file path where the dictionary should be saved.
    mydict : dict
        The dictionary to save.
    """
    f = open(file_save, "wb")
    pickle.dump(mydict, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_pickle(file_pickle):
    """
    Load a dictionary object from a pickle file.

    Parameters
    ----------
    file_pickle : str
        The file path of the pickle file to load the dictionary from.

    Returns
    -------
    dict
        The loaded dictionary.
    """
    f = open(file_pickle, "rb")
    mydict = pickle.load(f)
    f.close()
    return mydict


def load_predict(file_predict, *args):
    """
    Load prediction results from files.

    Parameters
    ----------
    file_predict : str
        The base path for prediction files.
    *args : list
        Additional arguments specifying the types of predictions to load.

    Returns
    -------
    list
        A list of prediction results.
    """
    print("load_predict", args)
    yV = []
    for keyV in args[0]:
        print("keyV", keyV)
        file_save = f"{file_predict}_{keyV}"
        yV.append(load_pickle(file_save))
    return yV


def save_predict(file_predict,  **kwargs):
    """
    Save prediction results to files.

    Parameters
    ----------
    file_predict : str
        The base path for prediction files.
    **kwargs : dict
        Keyword arguments where keys are the types of predictions and values are the predictions to save.
    """
    for keyV in kwargs:
        file_save = f"{file_predict}_{keyV}"
        save_pickle(file_save, kwargs[keyV])


def predictV2(val_loader, model, device):
    """
    Make predictions using the model on the provided data loader.

    Parameters
    ----------
    val_loader : DataLoader
        The data loader containing the data to predict on.
    model : torch.nn.Module
        The model to use for making predictions.
    device : torch.device
        The device to run the model on.

    Returns
    -------
    dict
        A dictionary containing prediction results including 'y_pred', 'y_t', and 'y_id'.
    """

    model.eval()
    y_pred, y_t, y_id = [], [], []
    with tqdm(total=len(val_loader)) as tqd:
        for i, (ID, t, x_video, x_audio) in enumerate(val_loader):
            if i == 0:
                print("x_video.size():", x_video.size())
                print("x_audio.size():", x_audio.size())
                print("ID:", ID)
                print("t:", t)
            # if i > 10:break

            if x_video.size() != 1:
                x_video = x_video.to(device)
            if x_audio.size() != 1:
                x_audio = x_audio.to(device)

            output = model(x_video, x_audio)  # , xA

            _, y_pred_max = torch.max(output.data.cpu(), 1)
            y_pred_new = y_pred_max.cpu().tolist()
            y_t_new = t.tolist()

            y_pred.extend(y_pred_new)
            y_t.extend(y_t_new)
            y_id.extend(ID)

            tqd.set_postfix(loss='{:05d}'.format(len(y_pred)))
            tqd.update()

    return {"y_pred": y_pred, "y_t": y_t, "y_id": y_id}


def run_model(args_in, args, path_checkpoint, file_predictV2, mode_train_val="test"):
    """
    Set up and run the model for prediction.

    Parameters
    ----------
    args_in : Namespace
        Parsed command line arguments.
    args : dict
        Arguments or configuration for the model and prediction.
    path_checkpoint : str
        Path to the model checkpoint file.
    file_predictV2 : str
        File path to save prediction results.
    mode_train_val : str, optional
        Mode of operation (e.g., "test", "validation", "predict"). Default is "test".

    """

    print(f"run_model {mode_train_val} {file_predictV2}")

    model_X = X_input(args["TSM"])
    model_BB = BackBone(args["TSM"])
    model = VCM(args["TSM"], model_X, model_BB)

    """set_cuda_device"""
    device, device_id, args = set_cuda_device(args_in, args)
    model, DataParallel = set_model_DataParallel(args, model)
    model.to(device)

    print("device, device_id", device, device_id)
    print("DataParallel", DataParallel)

    optimizer = torch.optim.SGD(model.parameters(), args["net_optim_param"]["lr"],
                                momentum=args["net_optim_param"]["momentum"],
                                weight_decay=args["net_optim_param"]["weight_decay"])

    (model, optimizer, data_state) = load_model(path_checkpoint,
                                                model, optimizer, DataParallel=DataParallel, Filter_layers={})

    args["net_run_param"]["batch_size"] = 8
    ModelDataSet = GetDataSetPredict(args, mode_train_val=mode_train_val)
    ValidDataLoader = GetDataLoadersPredict(ModelDataSet, args)

    # {"y_pred": y_pred, "y_t": y_t, "y_id": y_id}
    YP = predictV2(ValidDataLoader, model, device)
    save_predict(file_predictV2, **YP)


def assemble(y_V, y_t, y_id):
    """
    Assemble prediction results for analysis.

    Parameters
    ----------
    y_V : list
        List of predicted values.
    y_t : list
        List of true values.
    y_id : list
        List of identifiers for the predictions.

    Returns
    -------
    dict
        A dictionary representing the assembled profile of predictions.
    """
    profile_V, stat_V = {}, Counter()
    for i, id in enumerate(y_id):
        p_V, t = y_V[i],  y_t[i]
        # print("p_V, t", p_V, t)
        if id not in profile_V:
            profile_V[id] = []
        profile_V[id].append([p_V, t])
        stat_V[p_V] += 1
    print("stat_V", stat_V)

    return profile_V


# python3 predict.py --data config/adcumen1_ortigia.json  --id f16_BG0_RGB_audio_16_INET21K --type test --model logs/screen_multiclass_may/ad_timm_16f_j05_BG0/0_1_2_3_4_5_6_7_8_0.5_0.5_0.1_1_resnet50_timm_16_0_1_adcumen


def main():
    """
    The main function to run the prediction process. It handles command-line arguments, sets up the model and data,
    runs the prediction, and saves the results.
    """

    global args_model, args_data

    parser = argparse.ArgumentParser()
    # arg.json file (specifies dataset for prediction)
    parser.add_argument("--data", type=str)
    # directory with trained model (must have <dir>/args.json and <dir>/checkpoint/unbalanced.ckpt.pth.tar)
    parser.add_argument("--model", type=str)
    parser.add_argument("--cuda_ids", type=str)  # 0_1_2_3
    parser.add_argument('--type', type=str)  # validation, test, predict
    parser.add_argument('--id', type=str)  # <id> for prediction

    args_in = parser.parse_args()

    """ load model for prediction"""
    if args_in.model:
        arg_model = f"{args_in.model}/args.json"
        path_checkpoint = f"{args_in.model}/checkpoint/balanced.ckpt.pth.tar"
        args_model = loadarg(arg_model)
        args = args_model
    else:
        print("Please provide --model <path_to_model_folder>, which should be used for predictions")
        exit()

    """ specify type of data """
    if args_in.type == "validation":
        mode_train_val = "validation"
    elif args_in.type == "test":
        mode_train_val = "test"
    else:
        mode_train_val = "predict"

    """ specify dataset for prediction args.json file"""
    if args_in.data:
        args_data = loadarg(args_in.data)
        args_data["dataset"]["video_img_param"] = args["dataset"]["video_img_param"]
        args_data["dataset"]["video_augmentation"] = args["dataset"]["video_augmentation"]
        args_data["dataset"]["audio_img_param"] = args["dataset"]["audio_img_param"]
        args_data["dataset"]["audio_augmentation"] = args["dataset"]["audio_augmentation"]
        args["dataset"] = args_data["dataset"]
    else:
        print("No data for prediction is provided, you need to specify --data <path_to_data_config_file>")
        exit()

    """ creates folder for output results"""
    if args_in.id:
        run_id = args_in.id
        path_save_predicted = f'{args["dataset"]["data_dir"]}/predicted'
        path_save_results = f'{path_save_predicted}/{run_id}'
        os.makedirs(path_save_predicted, exist_ok=True)
        os.makedirs(path_save_results, exist_ok=True)

    else:
        path_save_predicted = f'{args["dataset"]["data_dir"]}/predicted'
        path_save_results = f'{args["dataset"]["data_dir"]}/predicted/NoId'
        os.makedirs(path_save_predicted, exist_ok=True)
        os.makedirs(path_save_results, exist_ok=True)

    """ specify file with prediction to save """
    file_predictV2 = f'{path_save_results}/{mode_train_val}'
    print("file_predictV2", file_predictV2)

    """ if cuda specified -> make prediction else -> load prediction"""
    if args_in.cuda_ids:
        print("cuda devices:", args_in.cuda_ids)
    else:
        args_in.cuda_ids = "0"

    run_model(args_in, args, path_checkpoint,
              file_predictV2, mode_train_val=mode_train_val)

    """load prediction"""
    yV = ["y_pred", "y_t", "y_id"]
    [y_V, y_t, y_id] = load_predict(file_predictV2, yV)

    """assemble prediction for each video"""
    profile_P = assemble(y_V, y_t, y_id)  # profile_V[id].append([p_V,t])

    file_results = f"{file_predictV2}_predicted_profiles"
    with open(file_results, 'w') as f:
        json.dump(profile_P, f, indent=4)


if __name__ == '__main__':
    main()
