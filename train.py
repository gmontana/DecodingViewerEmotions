import argparse
import json
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import confusion_matrix
from collections import Counter
from lib.dataset.dataset import GetDataSet, GetDataLoaders
from lib.utils.utils import loadarg, AverageMeter, get_labels, remap_acc_to_emotions, multiclass_accuracy
from lib.utils.config_args import adjust_args_in
from lib.utils.set_gpu import set_model_DataParallel, set_cuda_device
from lib.utils.set_folders import check_rootfolders, get_file_results
from lib.utils.saveloadmodels import checkpoint_acc, save_timepoint, load_model, checkpoint_confusion_matrix
from lib.model.model import VCM
from lib.model.prepare_input import X_input
from lib.model.backbone import BackBone
from lib.model.policy import gradient_policy


def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    Namespace
        The namespace containing the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)  # config/
    parser.add_argument("--resume", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--cuda_ids", type=str)  # 0_1_2_3
    parser.add_argument("--run_id", type=str)
    parser.add_argument('--market_filtr', nargs="+", type=int)
    parser.add_argument('--porog', type=int)
    return parser.parse_args()


def load_and_adjust_args(args_in):
    """
    Load and adjust the configuration arguments.

    Parameters
    ----------
    args_in : Namespace
        The parsed command line arguments.

    Returns
    -------
    dict
        The adjusted arguments dictionary.
    """
    args = loadarg(args_in.config)
    args = adjust_args_in(args, args_in)
    return args


def initialise_model(args_in, args):
    """
    Initialize and return the model along with optimizer and criterion.

    Parameters
    ----------
    args_in : dict
        The input arguments from command line.
    args : dict
        The arguments loaded from the configuration file.

    Returns
    -------
    tuple
        A tuple containing the model, DataParallel, device, device_id, optimizer, and criterion.

    """

    model_X = X_input(args["TSM"])
    model_BB = BackBone(args["TSM"])
    model = VCM(args["TSM"], model_X, model_BB)

    device, device_id, args = set_cuda_device(args_in, args)
    model, DataParallel = set_model_DataParallel(args, model)
    model.to(device)

    print("device, device_id", device, device_id)
    print("DataParallel", DataParallel)

    optimizer = torch.optim.SGD(model.parameters(), args["net_optim_param"]["lr"],
                                momentum=args["net_optim_param"]["momentum"],
                                weight_decay=args["net_optim_param"]["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    return model, DataParallel, device, device_id, optimizer, criterion


def validate_external_data(args, args_data, model, device, criterion, mode_train_val="validation"):
    """
    Validate the model using external data.

    Parameters
    ----------
    args : dict
        The arguments loaded from the configuration file.
    args_data : dict
        The arguments specific to the data.
    model : torch.nn.Module
        The model to validate.
    device : torch.device
        The device to run the model on.
    criterion : torch.nn.modules.loss
        The criterion to use for validation.
    mode_train_val : str, optional
        The mode of validation (default is "validation").

    Returns
    -------
    None

    """
    ValidDataSet = GetDataSet(args_data, mode_train_val=mode_train_val)
    ValidDataSet.unite_positive_negative()
    args["net_run_param"]["batch_size"] = 16
    # , mode_train_val="validation"
    ValidDataLoader = GetDataLoaders(ValidDataSet, args)
    val_score, y_pred, y_true = validate(
        ValidDataLoader, model, device, criterion, 0)

    matrix = confusion_matrix(y_true, y_pred)
    all = matrix.sum(axis=1)
    accE = matrix.diagonal() / matrix.sum(axis=1)
    print("matrix\n", matrix)
    accE = remap_acc_to_emotions(accE, ValidDataSet)

    """saving results to file_results at args.root_folder"""

    result = {'best_score': val_score,  'accE': accE}
    file_results = f'tmp/results/{mode_train_val}_{args["run_id"]}'
    with open(file_results, 'w') as f:
        json.dump(result, f, indent=4)


def run_epoch(epoch, DataLoaders, model, device, optimizer, criterion, DataParallel=False):
    """
    Run one epoch of training and validation.

    Parameters
    ----------
    epoch : int
        The current epoch number.
    DataLoaders : list
        List containing data loaders for training, validation, and testing.
    model : torch.nn.Module
        The model to train and validate.
    device : torch.device
        The device to run the model on.
    optimizer : torch.optim
        The optimizer for training.
    criterion : torch.nn.modules.loss
        The criterion to use for training.
    DataParallel : bool, optional
        Flag to indicate if DataParallel is used (default is False).

    Returns
    -------
    dict
        A dictionary containing various metrics and results from the epoch.
    """

    global best_score, test_score, history_score

    [TrainDataLoader, ValidDataLoader, TestDataLoader,
     TrainDataSet, ValidDataSet, TestDataSet] = DataLoaders

    # Unite positive and negative samples if necessary
    TrainDataSet.unite_positive_negative()
    train_score = train(TrainDataLoader, model, device,
                        criterion, optimizer, epoch)

    ValidDataSet.unite_positive_negative()
    val_score, y_pred, y_true = validate(
        ValidDataLoader, model, device, criterion, epoch)

    acc, conf_matrix = multiclass_accuracy(y_true, y_pred)
    accE = remap_acc_to_emotions(acc, ValidDataSet)

    TestDataSet.unite_positive_negative()
    test_score, y_pred, y_true = validate(
        TestDataLoader, model, device, criterion, epoch)

    acc_test, conf_matrix_test = multiclass_accuracy(y_true, y_pred)
    accE_test = remap_acc_to_emotions(acc, TestDataSet)

    # Checkpoints
    best_score["unbalanced"], is_best_score = checkpoint_acc(
        "unbalanced", args["checkpoint"], epoch, best_score["unbalanced"], val_score["unbalanced"],  model, optimizer, DataParallel)
    best_score["balanced"], is_best_score = checkpoint_acc(
        "balanced", args["checkpoint"], epoch, best_score["balanced"], val_score["balanced"], model, optimizer, DataParallel)

    best_score["unbalanced"], is_best_score = checkpoint_confusion_matrix(
        "unbalanced", args["checkpoint"], epoch,  best_score["unbalanced"], val_score["unbalanced"], conf_matrix, accE)
    best_score["balanced"], is_best_score = checkpoint_confusion_matrix(
        "balanced", args["checkpoint"], epoch, best_score["balanced"], val_score["balanced"], conf_matrix, accE)

    # Compile results into a dictionary or similar structure
    epoch_results = {
        "train_score": train_score,
        "val_score": val_score,
        "test_score": test_score,
        "y_pred": y_pred,
        "y_true": y_true,
        "acc": acc,
        "acc_test": acc_test,
        "conf_matrix": conf_matrix,
        "conf_matrix_test": conf_matrix_test,
        "accE": accE,
        "accE_test": accE_test,
        "best_score": best_score,
        "is_best_score": is_best_score
    }

    return epoch_results


def save_epoch_results(epoch_results, epoch, args):
    """
    Save the results of an epoch to a file.

    Parameters
    ----------
    epoch_results : dict
        The results from an epoch of training/validation.
    epoch : int
        The current epoch number.
    args : dict
        The script arguments or other configuration.
    """
    # Determine file path and save the results
    file_results = get_file_results(args, epoch)  # Adjust as needed
    with open(file_results, 'w') as f:
        json.dump(epoch_results, f, indent=4)


def validate(val_loader, model, device, criterion, epoch):
    """
    Validate the model for a single epoch using the given data loader.

    Parameters
    ----------
    val_loader : DataLoader
        The data loader for validation data.
    model : torch.nn.Module
        The model to be validated.
    device : torch.device
        The device on which to perform validation (CPU or GPU).
    criterion : torch.nn.modules.loss
        The loss function used for validation.
    epoch : int
        The current epoch number.

    Returns
    -------
    dict
        A dictionary containing validation scores and other metrics, including:
        - 'balanced': The balanced accuracy score.
        - 'unbalanced': The unbalanced accuracy score.
    list
        A list of predicted labels from the validation data.
    list
        A list of true labels from the validation data.

    Notes
    -----
    This function assumes that the model output and labels are in a format compatible with the provided criterion and accuracy functions. It also updates the model to evaluation mode before running validation and tracks the loss and predictions for each batch in the data loader.

    """

    score = {}
    score["balanced"], score["unbalanced"] = 0.00, 0.00
    losses = AverageMeter()
    model.eval()
    y_pred, y_true = [], []
    with tqdm(total=len(val_loader)) as t:
        for i, (ID,  y, x_video, x_audio) in enumerate(val_loader):
            if i == 0:
                print("x_video.size():", x_video.size(), x_audio.size())
            # exit()
            if x_video.size() != 1:
                x_video = x_video.to(device)
            if x_audio.size() != 1:
                x_audio = x_audio.to(device)
            y = y.long().to(device)

            output = model(x_video, x_audio)  # , xA

            loss = criterion(output, y)
            losses.update(loss.item(), y.size(0))

            _, y_pred_max = torch.max(output.data.cpu(), 1)
            y_pred.extend(y_pred_max.cpu().tolist())
            y_true.extend(y.cpu().tolist())

            t.set_postfix(loss='{:05.5f}'.format(float(losses.avg)))
            t.update()

    from lib.utils.utils import check_true_pred
    check_true_pred(y_pred, y_true)

    score["unbalanced"] = accuracy_score(y_true, y_pred)
    score["balanced"] = balanced_accuracy_score(y_true, y_pred)

    print(
        f'Testing Results:  epoch: {epoch} loss_cc: {round(float(losses.avg),3)}  accuracy_balanced: {round(float(score["balanced"])*100,2)} accuracy: {round(float(score["unbalanced"])*100,2)}')
    return score, y_pred, y_true


def train(train_loader, model, device, criterion, optimizer, epoch):

    score = {}
    score["balanced"], score["unbalanced"] = 0.00, 0.00
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        losses_cc, accuracy_cc, accuracy_bc = AverageMeter(), AverageMeter(),  AverageMeter()

        # switch to train mode
        model.train()
        y_pred, y_true = [], []

        lr = optimizer.param_groups[-1]['lr']
        print(f'training: epoch: {epoch} lr: {float(lr)} ')
        with tqdm(total=len(train_loader)) as t:
            for i, (ID,  y, x_video, x_audio) in enumerate(train_loader):

                if i == 0:
                    print("x_video", x_video.size())
                    print("x_audio", x_audio.size())
                    print("y", y.size())

                if x_video.size() != 1:
                    x_video = x_video.to(device)
                if x_audio.size() != 1:
                    x_audio = x_audio.to(device)
                y = y.long().to(device)

                output = model(x_video, x_audio)  # , xA
                loss_cc = criterion(output, y)
                losses_cc.update(loss_cc.item(), y.size(0))

                _, y_pred_max = torch.max(output.data.cpu(), 1)

                y_pred_new = y_pred_max.detach().cpu().tolist()
                y_true_new = y.cpu().tolist()
                y_pred.extend(y_pred_new)
                y_true.extend(y_true_new)

                score["unbalanced"] = accuracy_score(y_true_new, y_pred_new)
                score["balanced"] = balanced_accuracy_score(
                    y_true_new, y_pred_new)

                accuracy_cc.update(score["unbalanced"], y.size(0))
                accuracy_bc.update(score["balanced"], y.size(0))

                loss = loss_cc
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                t.set_postfix(loss='{:05.5f}'.format(float(losses_cc.avg)),
                              accuracy='{:05.3f}'.format(
                                  float(accuracy_cc.avg)*100),
                              accuracy_Balanced='{:05.3f}'.format(
                                  float(accuracy_bc.avg) * 100)
                              )

                t.update()

        score["unbalanced"] = accuracy_score(y_true, y_pred)
        score["balanced"] = balanced_accuracy_score(y_true, y_pred)

        print(
            f'Training Results:  epoch: {epoch} loss_cc: {round(float(losses_cc.avg), 3)}  accuracy_balanced: {round(float(score["balanced"]) * 100, 2)} accuracy: {round(float(score["unbalanced"]) * 100, 2)}')
        return score


def setup_scores():
    """
    Initialize the scoring dictionaries.

    Returns
    -------
    tuple
        Tuple containing best_score and history_score dictionaries.
    """
    best_score, history_score = {}, {}
    best_score["balanced"], best_score["unbalanced"] = 0.00, 0.00
    return best_score, history_score


def model_initialization(args_in):
    """
    Initialize the model and load the checkpoint if specified.

    Parameters
    ----------
    args_in : Namespace
        The parsed command line arguments.

    Returns
    -------
    tuple
        Tuple containing initialized model and other related objects.
    """
    args_model = loadarg(f"{args_in.model}/args.json")
    path_checkpoint = f"{args_in.model}/checkpoint/balanced.ckpt.pth.tar"

    model, DataParallel, device, device_id, optimizer, criterion = initialise_model(
        args_in, args_model)
    model, optimizer, data_state = load_model(
        path_checkpoint, model, optimizer, DataParallel=DataParallel, Filter_layers={})
    print("data_state", data_state)

    return model, DataParallel, device, device_id, optimizer, criterion


def setup_data_loaders(args_in, args):
    """
    Setup data loaders for training, validation, and testing.

    Parameters
    ----------
    args_in : Namespace
        The parsed command line arguments.
    args : dict
        The adjusted arguments dictionary.

    Returns
    -------
    list
        List of data loaders for training, validation, and testing.
    """
    if args_in.porog:
        fixed_porog = args["emotion_jumps"]["porog"]
        TrainDataSet = GetDataSet(
            args, mode_train_val="training", fixed_porogs=fixed_porog)
        ValidDataSet = GetDataSet(
            args, mode_train_val="validation", fixed_porogs=fixed_porog)
    else:
        TrainDataSet = GetDataSet(
            args, mode_train_val="training", fixed_porogs=-1)
        ValidDataSet = GetDataSet(
            args, mode_train_val="validation", fixed_porogs=-1)
        TestDataSet = GetDataSet(args, mode_train_val="test", fixed_porogs=-1)

    TrainDataLoader = GetDataLoaders(TrainDataSet, args)
    ValidDataLoader = GetDataLoaders(ValidDataSet, args)
    TestDataLoader = GetDataLoaders(TestDataSet, args)

    return [TrainDataLoader, ValidDataLoader, TestDataLoader, TrainDataSet, ValidDataSet, TestDataSet]


def main():
    """
    Main function to run the training process.
    """

    global args, best_score, history_score
    best_score, history_score = {}, {}
    best_score["balanced"], best_score["unbalanced"] = 0.00, 0.00

    args_in = parse_arguments()
    args = load_and_adjust_args(args_in)

    # Model initialization and other setup based on the provided code
    if args_in.model:
        args_model = loadarg(f"{args_in.model}/args.json")
        path_checkpoint = f"{args_in.model}/checkpoint/balanced.ckpt.pth.tar"

        model, DataParallel, device, device_id, optimizer, criterion = initialise_model(
            args_in, args_model)
        model, optimizer, data_state = load_model(
            path_checkpoint, model, optimizer, DataParallel=DataParallel, Filter_layers={})
        print("data_state", data_state)

        args_data = args_model.copy()
        args_data["dataset"] = args["dataset"]

        if args_in.market_filtr:
            args_data["market_filtr"] = args_in.market_filtr
            print("market_filtr", args_data["market_filtr"])

        args["run_id"] = args_in.run_id

        if "file_val_list" in args_data["dataset"]:
            validate_external_data(args, args_data, model, device,
                                   criterion, mode_train_val="validation")
            if "file_test_list" in args_data["dataset"]:
                validate_external_data(args, args_data, model,
                                       device, criterion, mode_train_val="test")
        else:

            validate_external_data(args, args_data, model, device,
                                   criterion, mode_train_val="predict")

        exit()

    DataLoaders = setup_data_loaders(args_in, args)
    # Assuming TrainDataSet is at index 3
    args["TSM"]["num_class"] = len(DataLoaders[3].map_label)
    print("args.num_class", args["TSM"]["num_class"])

    model, DataParallel, device, device_id, optimizer, criterion = initialise_model(
        args_in, args)
    cudnn.benchmark = True
    check_rootfolders(args)
    print('Results would be stored at: ', args["output_folder"])

    # Training loop
    for epoch in range(args["start_epoch"], args["last_epoch"]):

        from lib.utils.report import report_model_param
        report_model_param(args)

        if 'save_epoch' in args:
            if epoch in args["save_epoch"]:
                save_timepoint(
                    f"epoch_{epoch}", args["checkpoint"],  model, optimizer, DataParallel)

        gradient_policy(args, epoch, optimizer)

        # Run the epoch and get results
        epoch_results = run_epoch(
            epoch, DataLoaders, model, device, optimizer, criterion, DataParallel)

        # Save results
        save_epoch_results(epoch_results, epoch, args)


if __name__ == '__main__':
    main()
