"""
temporal_fusion.py

Overview:
This script defines classes and functions related to temporal fusion or shifting in neural network models. Temporal fusion
is a technique often used in video or sequence processing tasks to enhance the model's understanding of temporal dynamics
by shifting features across time or modality. The script includes the TempoModalShift class for applying temporal shifts
to layers and utility functions for integrating these shifts into existing network architectures.

Classes:
- TempoModalShift: A class for applying temporal shifts to model layers.

Functions:
- shift_block: Applies the TempoModalShift to blocks within a stage of a network.
- make_Shift: Integrates temporal shifting into an entire network architecture.

Usage:
The classes and functions in this script are typically used to modify existing network architectures, such as ResNet or
DenseNet, to include temporal shifting capabilities. They are used in the construction or modification of models for
tasks that involve temporal or sequential data, such as video classification or activity recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import Counter
import timm


class TempoModalShift(nn.Module):
    """
    TempoModalShift is a module for applying temporal shifts to model layers. It is designed to enhance the model's
    understanding of temporal dynamics by shifting features across time or modality.

    Parameters
    ----------
    layer : torch.nn.Module
        The layer to which the temporal shift will be applied.
    n_video_segments : int
        The number of video segments to consider for shifting.
    n_motion_segments : int
        The number of motion segments to consider for shifting.
    n_audio_segments : int
        The number of audio segments to consider for shifting.
    input_mode : int
        Specifies how different input types should be combined or processed.
    f_div : int
        The rate of features to fuse/shift.
    shift_depth : int
        The depth of shifting to apply.
    mode : str
        The mode of shifting (e.g., "shift_temporal", "shift_temporal_modality").

    Methods
    -------
    forward(x):
        Applies the temporal shift to the input tensor and passes it through the layer.
    shift_temporal(x):
        Applies a temporal shift to the input tensor.
    shift_temporal_modality(x):
        Applies a temporal shift across modalities to the input tensor.

    Note from Alexei - Patameters:
        f_div=8 rate of features to fuse/shift
        self.n_video_segments = param_TSM["video_segments"] #[8, 8, 1]
        self.n_audio_segments = param_TSM["audio_segments"]  # [8, 8, 1]
        self.n_motion_segments = 0
    """

    def __init__(self, layer, n_video_segments, n_motion_segments, n_audio_segments,  input_mode,  f_div,  shift_depth, mode):
        super(TempoModalShift, self).__init__()
        self.layer = layer
        self.mode = mode
        self.n_segment = n_video_segments + n_motion_segments + n_audio_segments

        self.shift_config = []
        if mode == "shift_temporal":
            if input_mode == 1:

                s1, s2 = 0, n_video_segments + n_motion_segments
                self.shift_config.append([s1, s2])
            else:
                s1, s2 = 0, n_video_segments + n_motion_segments
                self.shift_config.append([s1, s2])

        if mode == "shift_temporal_modality":
            s = n_video_segments
            self.shift_config.append([0, s, s])

        self.in_channels = self.layer.in_channels
        self.f_n = int(self.in_channels // f_div)
        self.shift_depth = shift_depth

    def forward(self, x):

        if self.mode == "shift_temporal":
            x = self.shift_temporal(x)
        elif self.mode == "shift_temporal_modality":
            x = self.shift_temporal_modality(x)
        elif self.mode == "shift_spatial":
            x = self.shift_spatial(x)
        else:
            raise ValueError(
                f'TempoModalShift Unknown mode: {self.mode} (must be "shift_temporal", "shift_single_modality" or "shift_temporal_modality")')

        x = self.layer(x)
        return x

    def shift_temporal(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        f = self.f_n // self.shift_depth
        out = x.clone()

        for (s1, s2) in self.shift_config:
            # print("shift_temporal", s1, s2)
            # continue
            for d in range(self.shift_depth):
                l = d + 1
                f1 = 2*d*f
                f2 = (2*d+1)*f
                # print("shift_temporal d", d, f1, f2)
                out[:, s1:s2-l, f1:f1+f] = x[:, s1+l:s2, f1:f1+f]
                out[:, s1+l:s2, f2:f2+f] = x[:, s1:s2-l, f2:f2+f]
        # exit()
        return out.view(nt, c, h, w)

    def shift_temporal_modality(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        f = self.f_n
        out = x.clone()
        (s1, s2, s3) = self.shift_config[0]

        out[:, s1:s1+s3, 0:f] = x[:, s2:s2+s3, 0:f]
        out[:, s2:s2+s3, 0:f] = x[:, s1:s1+s3, 0:f]
        # for i in range(s3):
        #    out[:, s1 + i, 0:f] = x[:, s2 + i, 0:f]
        #    out[:, s2 + i, 0:f] = x[:, s1 + i, 0:f]
        return out.view(nt, c, h, w)


def shift_block(stage, n_video_segments, n_motion_segments, n_audio_segments, input_mode,  f_div,  shift_depth, mode, n_insert, m_insert):
    """
    Applies the TempoModalShift to blocks within a stage of a network.

    Parameters
    ----------
    stage : torch.nn.Module
        The stage of the network containing multiple blocks.
    n_video_segments : int
        The number of video segments to consider for shifting.
    n_motion_segments : int
        The number of motion segments to consider for shifting.
    n_audio_segments : int
        The number of audio segments to consider for shifting.
    input_mode : int
        Specifies how different input types should be combined or processed.
    f_div : int
        The rate of features to fuse/shift.
    shift_depth : int
        The depth of shifting to apply.
    mode : str
        The mode of shifting (e.g., "shift_temporal", "shift_temporal_modality").
    n_insert : int
        The interval at which to insert shifts within the stage.
    m_insert : int
        The specific blocks within the interval to apply the shift.

    Returns
    -------
    torch.nn.Module
        The modified stage with TempoModalShift applied to specific blocks.
    """
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i % n_insert == m_insert:
            print(
                f'shift_block:  stage with {len(blocks)} blocks ,mode:{mode}')
            blocks[i].conv1 = TempoModalShift(
                b.conv1, n_video_segments, n_motion_segments, n_audio_segments, input_mode, f_div, shift_depth, mode)

    return stage


def make_Shift(net, n_video_segments, n_motion_segments, n_audio_segments,  input_mode,  f_div,  shift_depth, mode, n_insert, m_insert):
    """
    Integrates temporal shifting into an entire network architecture.

    Parameters
    ----------
    net : torch.nn.Module
        The network to which the temporal shifts will be applied.
    n_video_segments : int
        The number of video segments to consider for shifting.
    n_motion_segments : int
        The number of motion segments to consider for shifting.
    n_audio_segments : int
        The number of audio segments to consider for shifting.
    input_mode : int
        Specifies how different input types should be combined or processed.
    f_div : int
        The rate of features to fuse/shift.
    shift_depth : int
        The depth of shifting to apply.
    mode : str
        The mode of shifting (e.g., "shift_temporal", "shift_temporal_modality").
    n_insert : int
        The interval at which to insert shifts within the network.
    m_insert : int
        The specific blocks within the interval to apply the shift.

    Notes
    -----
    This function modifies the provided network by integrating temporal shifts into its architecture. It is typically
    used to enhance models for tasks involving temporal or sequential data.
    """

    if isinstance(net, (torchvision.models.ResNet, timm.models.ResNet)):

        net.layer1 = shift_block(net.layer1, n_video_segments, n_motion_segments,
                                 n_audio_segments,  input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)
        net.layer2 = shift_block(net.layer2, n_video_segments, n_motion_segments,
                                 n_audio_segments,  input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)
        net.layer3 = shift_block(net.layer3, n_video_segments, n_motion_segments,
                                 n_audio_segments,  input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)
        net.layer4 = shift_block(net.layer4, n_video_segments, n_motion_segments,
                                 n_audio_segments,  input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)

    elif isinstance(net, torchvision.models.DenseNet):

        net.features.denseblock1 = shift_block(net.features.denseblock1, n_video_segments, n_motion_segments,
                                               n_audio_segments, input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)
        net.features.denseblock2 = shift_block(net.features.denseblock2, n_video_segments, n_motion_segments,
                                               n_audio_segments, input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)
        net.features.denseblock3 = shift_block(net.features.denseblock3, n_video_segments, n_motion_segments,
                                               n_audio_segments, input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)
        net.features.denseblock4 = shift_block(net.features.denseblock4, n_video_segments, n_motion_segments,
                                               n_audio_segments, input_mode,  f_div,  shift_depth, mode, n_insert, m_insert)

    else:
        raise NotImplementedError("UnKnown Arch (not ResNet or DenceNet)")
