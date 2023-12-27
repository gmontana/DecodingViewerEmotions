

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def freeze_layers(net, FilterPos, DataParallel=False):
    """
    Freeze specific layers of the network while allowing others to continue training.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose layers are to be partially frozen.
    FilterPos : list or dict
        Identifiers (usually names or indices) of layers that should remain trainable.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function iterates through all parameters of the model and sets 'requires_grad' to False
    to freeze layers. It then selectively re-enables training for layers specified in FilterPos.
    """
    if DataParallel:
        for param_name, param in net.module.named_parameters():
            param.requires_grad = False
            for layer_id in FilterPos:
                if layer_id in param_name:
                    param.requires_grad = True
    else:
        for param_name, param in net.named_parameters():
            param.requires_grad = False
            for layer_id in FilterPos:
                if layer_id in param_name:
                    print("fffreeze_layers ", layer_id)
                    param.requires_grad = True


def unfreeze_layers(net, FilterNeg, DataParallel=False):
    """
    Unfreeze specific layers of the network while keeping others frozen.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose layers are to be partially unfrozen.
    FilterNeg : list or dict
        Identifiers (usually names or indices) of layers that should remain frozen.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function iterates through all parameters of the model and sets 'requires_grad' to True
    to unfreeze layers. It then selectively disables training for layers specified in FilterNeg.
    """

    if DataParallel:
        for param_name, param in net.module.named_parameters():
            param.requires_grad = True
            for layer_id in FilterNeg:
                if layer_id in param_name:
                    param.requires_grad = False
    else:
        for param_name, param in net.named_parameters():
            param.requires_grad = True
            for layer_id in FilterNeg:
                if layer_id in param_name:
                    print("unfreeze_layers ", layer_id)
                    param.requires_grad = False


def count_free_layers(net, DataParallel=False):
    """
    Count the number of trainable (free) and frozen layers in the network.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose layers are to be counted.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Returns
    -------
    tuple
        A tuple containing the count of trainable layers and frozen layers, respectively.

    Notes
    -----
    This function iterates through all parameters of the model and counts the number of parameters
    that have 'requires_grad' set to True (trainable) and False (frozen).
    """
    count_free, count_freeze = 0, 0
    if DataParallel:
        for param in net.module.parameters():
            if param.requires_grad:
                count_free += 1
            else:
                count_freeze += 1
    else:
        for param in net.parameters():
            if param.requires_grad:
                count_free += 1
            else:
                count_freeze += 1

        # Additional logging for models with few frozen parameters
        if count_freeze < 10:
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    print("freeze", name)

    return count_free, count_freeze


def freeze_policy(args, net, epoch, optimizer, DataParallel=False):
    """
    Apply a freezing policy to the network based on the current epoch. This involves freezing
    or unfreezing layers as specified in the 'args' configuration.

    Parameters
    ----------
    args : dict
        The script arguments or other configuration, specifically containing the 'net_optim_policy' dictionary
        with 'freeze_layers' and 'unfreeze_layers' for the freeze/unfreeze schedule.
    net : torch.nn.Module
        The neural network model whose layers are to be frozen/unfrozen.
    epoch : int
        The current epoch number.
    optimizer : torch.optim.Optimizer
        The optimizer being used in the training.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    The function reads the freeze and unfreeze schedule from 'args' and applies the appropriate
    freezing or unfreezing policy to the model's layers based on the current epoch.
    """
    FilterFreezeA, FilterFreezeB = {}, {}
    for i in range(0, len(args.net_optim_policy["freeze_layers"]), 2):
        layer_id = args.net_optim_policy["freeze_layers"][i]
        epoch_i = args.net_optim_policy["freeze_layers"][i+1]

        if epoch == epoch_i:
            FilterFreezeA[layer_id] = 1

    for i in range(0, len(args.net_optim_policy["unfreeze_layers"]), 2):
        layer_id = args.net_optim_policy["unfreeze_layers"][i]
        epoch_start = args.net_optim_policy["unfreeze_layers"][i + 1]

        if epoch < epoch_start:
            FilterFreezeB[layer_id] = 1

    if len(FilterFreezeA) > 0:
        freeze_layers(net, FilterFreezeA, DataParallel=DataParallel)
    else:
        unfreeze_layers(net, FilterFreezeB, DataParallel=DataParallel)


def gradient_policy(args, epoch, optimizer):
    """
    Adjust the learning rate of the optimizer based on the epoch according to a predefined schedule.

    This function assumes that the learning rate decay schedule is provided in the 'lr_decay' key of the
    'net_optim_param' dictionary within the 'args'. The 'lr_decay' list is expected to contain pairs of values,
    where each pair consists of a learning rate multiplier and the epoch number at which this multiplier should be applied.

    Parameters
    ----------
    args : dict
        The script arguments or other configuration, specifically containing the 'net_optim_param' dictionary
        with 'lr' for the initial learning rate and 'lr_decay' for the decay schedule.
    epoch : int
        The current epoch number.
    optimizer : torch.optim.Optimizer
        The optimizer being used in the training, which will have its learning rate adjusted.

    Notes
    -----
    The 'lr_decay' list is expected to be in the format: [multiplier1, epoch1, multiplier2, epoch2, ...].
    For example, [0.1, 30, 0.01, 50] means reduce the learning rate to 0.1x at epoch 30 and then to 0.01x at epoch 50.
    This function modifies the learning rate of the last parameter group of the optimizer.
    """
    # Extract learning rate decay parameters
    lr_decay = args["net_optim_param"]["lr_decay"]
    print("lr_decay", lr_decay, len(lr_decay))

    # Loop through the decay schedule and adjust learning rate if necessary
    for i in range(0, len(lr_decay), 2):
        # Check if the current epoch has reached the epoch for the next decay step
        if epoch >= lr_decay[i+1]:
            # Adjust the learning rate
            optimizer.param_groups[-1]['lr'] = lr_decay[i] * \
                args["net_optim_param"]["lr"]

    # Print the current learning rate and epoch for logging or debugging
    print("run_epoch lr", optimizer.param_groups[-1]['lr'], "epoch:", epoch)


def get_optim_policy(args, net, epoch, optimizer, DataParallel=False):
    """
    Apply the optimization policy for the network based on the current epoch. This involves adjusting
    learning rates, freezing/unfreezing layers, and other optimization-related adjustments.

    Parameters
    ----------
    args : dict
        The script arguments or other configuration, specifically containing the 'net_optim_policy' dictionary
        with various policies for optimization.
    net : torch.nn.Module
        The neural network model to apply optimization policies to.
    epoch : int
        The current epoch number.
    optimizer : torch.optim.Optimizer
        The optimizer being used in the training.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function orchestrates the application of various optimization policies including freezing layers,
    adjusting learning rates, and other epoch-dependent modifications.
    """
    freeze_policy(args, net, epoch, optimizer, DataParallel)
    gradient_policy(args, epoch, optimizer)

    # Additional policies based on the epoch
    if args.net_optim_policy["freeze_BN_layers"] < epoch:
        freeze_BN_layers(net, DataParallel)

    if args.net_optim_policy["freeze_BLOCK_layers_soft"] < epoch:
        freeze_BLOCK_layers_soft(net, epoch, DataParallel)

    count_free, count_freeze = count_free_layers(
        net, DataParallel=DataParallel)
    print("count_free_layers, count_freeze_layers", count_free, count_freeze)
    print("args.net_optim_policy", args.net_optim_policy)
    print("run_epoch lr", optimizer.param_groups[-1]['lr'], "epoch:", epoch)


def freeze_BN_layers(net, DataParallel=False):
    """
    Freeze Batch Normalization (BN) layers in the network.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose BN layers are to be frozen.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function iterates through all parameters of the model and sets 'requires_grad' to False
    for parameters in Batch Normalization layers to freeze them.
    """
    if DataParallel:
        for param_name, param in net.module.named_parameters():
            if "norm" in param_name or "bn" in param_name:
                param.requires_grad = False
    else:
        for param_name, param in net.named_parameters():
            if "norm" in param_name or "bn" in param_name:
                param.requires_grad = False


def freeze_BLOCK_layers_hard(net, epoch,  DataParallel=False):
    """
    Apply a hard freeze policy to specific blocks of layers in the network based on the current epoch.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose block layers are to be frozen/unfrozen.
    epoch : int
        The current epoch number.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function applies a hard freeze policy, meaning it completely freezes or unfreezes
    entire blocks of layers based on the current epoch. The specific blocks and epochs are
    determined by the training schedule.
    """

    print("freeze_BLOCK_layers_hard", epoch)

    if DataParallel:
        for param_name, param in net.module.named_parameters():
            param.requires_grad = False
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        print("param_name", param_name)
                        param.requires_grad = True
    else:
        for param_name, param in net.named_parameters():
            param.requires_grad = False
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        print("param_name", param_name)
                        param.requires_grad = True


def freeze_BLOCK_layers_soft(net, epoch,  DataParallel=False):
    """
    Apply a soft freeze policy to specific blocks of layers in the network based on the current epoch.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose block layers are to be softly frozen/unfrozen.
    epoch : int
        The current epoch number.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function applies a soft freeze policy, meaning it selectively freezes or unfreezes
    layers within specific blocks based on the current epoch. The specific blocks and epochs are
    determined by the training schedule.
    """

    print("freeze_BLOCK_layers_soft", epoch)

    if DataParallel:
        for param_name, param in net.module.named_parameters():
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        # print("freeze_BLOCK_layers_soft", param_name)
                        param.requires_grad = False
    else:
        for param_name, param in net.named_parameters():
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        # print("param_name", param_name)
                        param.requires_grad = False


def restart_fc_layers(net, DataParallel=False):
    """
    Reinitialize the weights of fully connected (fc) layers in the network.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model whose fc layers are to be reinitialized.
    DataParallel : bool, optional
        Flag indicating whether the model is wrapped with DataParallel. Default is False.

    Notes
    -----
    This function reinitializes the weights of the fully connected layers to their default
    initialization state. This might be used in transfer learning or other scenarios where
    reinitialization of certain layers is beneficial.
    """

    print("restart_fc_layers")

    if DataParallel:

        if len(net.module.num_class) >= 1:
            normal_(net.module.model.last_fc_0.weight, 0, 0.001)
            constant_(net.module.model.last_fc_0.bias, 0)
        if len(net.module.num_class) >= 2:
            normal_(net.module.model.last_fc_1.weight, 0, 0.001)
            constant_(net.module.model.last_fc_1.bias, 0)
        if len(self.num_class) >= 3:
            normal_(net.module.model.last_fc_2.weight, 0, 0.001)
            constant_(net.module.model.last_fc_2.bias, 0)

    else:
        if len(net.num_class) >= 1:
            normal_(net.model.last_fc_0.weight, 0, 0.001)
            constant_(net.model.last_fc_0.bias, 0)
        if len(net.num_class) >= 2:
            normal_(net.model.last_fc_1.weight, 0, 0.001)
            constant_(net.model.last_fc_1.bias, 0)
        if len(self.num_class) >= 3:
            normal_(net.model.last_fc_2.weight, 0, 0.001)
            constant_(net.model.last_fc_2.bias, 0)


if __name__ == '__main__':

    print('Test passed.')
