"""
backbone.py

Overview:
This script defines the BackBone class, which is used as the feature extraction component in a neural network model.
The BackBone class is responsible for setting up the base model architecture, applying any necessary modifications
like temporal shifts, and preparing it for integration with the rest of the model. It typically works with
pre-trained models or custom architectures specified in the parameters.

The script also includes utility functions for loading model weights and defining identity layers used in the model.

Classes:
- Identity: A placeholder layer that passes its input unchanged.
- BackBone: The main class for setting up the feature extraction part of the model.

Utility Functions:
- timm_load_model_weights: Loads weights into a model using the TIMM library.

Usage:
The BackBone class is instantiated with specific parameters and then integrated into a larger model architecture.
It is not typically run as a standalone script but imported and used in other model training or evaluation scripts.
"""


import torch
from torch import nn
from torch.nn.init import normal_, constant_
import torchvision

import sys
import timm


from lib.model.temporal_fusion import make_Shift


def timm_load_model_weights(model, model_path):
    """
    Load weights into a model using the TIMM library.

    Parameters
    ----------
    model : torch.nn.Module
        The model into which the weights will be loaded.
    model_path : str
        The path to the model weights file.

    Returns
    -------
    torch.nn.Module
        The model with loaded weights.

    Notes
    -----
    This function is specifically designed to work with models from the TIMM library. It loads the weights from the
    specified file into the model, handling any mismatches or missing layers gracefully by reporting them.
    """
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(
                    key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


class Identity(nn.Module):
    """
    A placeholder layer that passes its input unchanged.

    This layer is typically used as a replacement for the final layer of a pre-trained model during transfer learning
    or when modifying an existing architecture.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BackBone(nn.Module):

    """
    The main class for setting up the feature extraction part of the model.

    This class initializes the base model architecture, applies any necessary modifications like temporal shifts,
    and prepares it for integration with the rest of the model. It is typically used with pre-trained models or
    custom architectures specified in the parameters.

    Parameters
    ----------
    param_TSM : dict
        A dictionary containing parameters for the model, including architecture details and any specific
        modifications like temporal shift parameters.
    ATT : bool, optional
        A flag indicating whether attention mechanisms should be applied. Default is False.

    Attributes
    ----------
    base_model : torch.nn.Module
        The base model architecture, potentially modified with temporal shifts or other changes.
    features_dim_out : int
        The dimensionality of the output features from the base model.
    dropout_last : torch.nn.Dropout
        The dropout layer applied to the output features.
    lastpool : torch.nn.MaxPool1d
        The pooling layer applied to the output features if specified in the parameters.

    Methods
    -------
    prepare_base_model(param)
        Initializes the base model architecture based on the specified parameters.
    insert_shift_temporal(base_model, param)
        Applies temporal shift modifications to the base model if specified.
    insert_shift_temporal_modality(base_model, param)
        Applies modality-specific temporal shift modifications to the base model if specified.
    forward(x)
        Defines the forward pass of the model.
    """

    def __init__(self, param_TSM, ATT=False):
        super(BackBone, self).__init__()

        self.input_mode = param_TSM["main"]["input_mode"]
        self.n_video_segments = param_TSM["video_segments"]  # [8, 8, 1]
        self.n_audio_segments = param_TSM["audio_segments"]  # [8, 8, 1]
        self.n_motion_segments = 0
        if param_TSM["motion"]:
            self.n_motion_segments = self.n_video_segments

        self.param = param_TSM
        print("BackBone param:\n", self.param)

        self.base_model = self.prepare_base_model(param_TSM["main"])
        self.base_model = self.insert_shift_temporal(
            self.base_model, param_TSM["shift_temporal"])
        self.base_model = self.insert_shift_temporal_modality(
            self.base_model, param_TSM["shift_temporal_modality"])

        self.features_dim_out = getattr(
            self.base_model, self.base_model.last_layer_name).in_features

        print("BackBone output_dim:", self.features_dim_out)

        setattr(self.base_model, self.base_model.last_layer_name, Identity())

        self.dropout_last = nn.Dropout(p=param_TSM['main']["dropout"])

        if param_TSM['main']["last_pool"] > 1:
            self.lastpool = torch.nn.MaxPool1d(param_TSM['main']["last_pool"])
            self.features_dim_out = self.features_dim_out // param_TSM['main']["last_pool"]
            print("BackBone output_dim last_pool adjusted:", self.features_dim_out)

        self.ATT = ATT
        if ATT:
            self.conv_ATT = nn.Conv2d(
                2048, 1, kernel_size=1, padding=0, bias=True)
            self.conv_ATT.weight.data.normal_(0.01, 0.001)
            self.conv_ATT.bias.data.normal_(0.01, 0.001)
            self.relu = torch.nn.ReLU()

    def prepare_base_model(self, param):

        if "timm" in param["arch"]:
            base_model = timm.create_model('resnet50', pretrained=False)
            base_model = timm_load_model_weights(
                base_model, "net_weigths/resnet50_miil_21k.pth")
            print("prepare_base_model timm")
        else:
            base_model = getattr(torchvision.models, param["arch"])(
                True if param["pretrain"] == 'imagenet' else False)

        if 'resnet' in param["arch"]:
            base_model.last_layer_name = 'fc'
        elif 'densenet' in param["arch"]:
            base_model.last_layer_name = 'classifier'
        else:
            raise ValueError(f'Unknown BackBone base model: {param["arch"]}')

        return base_model

    def insert_shift_temporal(self, base_model, param):
        status = param["status"]
        f_div = param["f_div"]
        shift_depth = param["shift_depth"]
        n_insert = param["n_insert"]
        m_insert = param["m_insert"]

        n_video_segments, n_motion_segments, n_audio_segments = self.n_video_segments, self.n_motion_segments, self.n_audio_segments
        input_mode = self.input_mode
        mode = "shift_temporal"

        # print("insert_temporal_shift", param)

        if status:

            print("insert_temporal_shift n_video_segments, n_motion_segments n_audio_segments",
                  n_video_segments, n_motion_segments, n_audio_segments)
            print(
                f"make_temporal_shift n_insert={n_insert} m_insert={m_insert} f_div={f_div} input_mode={input_mode}")
            make_Shift(base_model, n_video_segments, n_motion_segments, n_audio_segments,
                       input_mode, f_div, shift_depth, mode, n_insert, m_insert)

        return base_model

    def insert_shift_temporal_modality(self, base_model, param):

        status = param["status"]
        f_div = param["f_div"]
        n_insert = param["n_insert"]
        m_insert = param["m_insert"]
        n_video_segments, n_motion_segments, n_audio_segments = self.n_video_segments, self.n_motion_segments, self.n_audio_segments
        input_mode = self.input_mode
        mode = "shift_temporal_modality"

        if n_video_segments < 1:
            status = False
        if n_motion_segments < 1:
            status = False

        if status:
            print("insert_modality_shift status", status)
            print(
                f"n_insert={n_insert} m_insert={m_insert} f_div={f_div} input_mode={input_mode}")
            shift_depth = 1  # not involved
            make_Shift(base_model,  n_video_segments, n_motion_segments, n_audio_segments,
                       input_mode, f_div, shift_depth, mode, n_insert, m_insert)

        return base_model

    def forward(self, x):
        # print("GG1 ", x.size())
        # exit()
        n_batches = x.size()[0]
        n_samples = x.size()[1]
        n_segs = x.size()[2]
        x = x.reshape((-1,) + x.size()[-3:])
        # x = x.view((-1,) + x.size()[-3:])
        # print("GG2 ", x.size())
        #

        if not self.ATT:
            x = self.base_model(x)
            # xA = 1
        """    
        else:
            blocks = list(self.base_model.children())
            for i, b in enumerate(blocks):
                # print("blocks", i,b)
                x = b(x)
                #print("blocks", i, x.size())
                if i == 7:
                    xA = self.conv_ATT(x)
                    #print("xA", xA.size(), xA.sum())
                    xA = self.relu(xA)
                    #print("xA", xA.size(), xA.sum())
                    #print("x1:", x.size(), x.sum() )
                    x = torch.einsum('ijkl,imkl->ijkl', x, xA  )
                    #print("x2:", x.size(), x.sum())

                #print("x.sum()", i, x.size(), x.sum())

            x = torch.squeeze(x, -1)
            x = torch.squeeze(x, -1)
        """

        # print("GG3 ", x.size())
        x = self.dropout_last(x)

        if self.param['main']["last_pool"] > 1:
            x = torch.unsqueeze(x, 1)
            x = self.lastpool(x)
            x = torch.squeeze(x, 1)

        x = x.view((-1,) + (n_samples, n_segs) + x.size()[-1:])

        # print("backbone x_out ", x.size())
        # exit()

        return x  # , xA
