
"""
prepare_input.py

Overview:
This script defines the X_input class, which is responsible for processing input data before it's fed into the model.
It handles different types of inputs like video, audio, and potentially motion, and applies necessary transformations
or preprocessing steps.

Classes:
- X_input: The main class for input data processing.

Utility Functions:
- vis_MDM2: Visualizes motion data.
- vis_MDM: Another function for visualizing motion data.

Usage:
The X_input class is instantiated with specific parameters and then used in a model pipeline to process input data
before it's passed to the feature extraction and classification components of the model.
"""


import torch
from torch import nn
from torch.nn.init import normal_, constant_
import torchvision
import torchvision.transforms as T
from lib.model.motion import MDFM


def vis_MDM2(X,  save_folder, id):
    """
    Visualize and save motion data matrices (MDM) as images.

    This function iterates through batches, modalities, and segments of the input tensor X, converting each segment
    into an image and saving it to the specified folder. It's typically used for debugging or understanding the motion
    data processed by the model.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor containing motion data. Expected to have dimensions [batch, modality, segment, channel, height, width].
    save_folder : str
        The path to the folder where the images should be saved.
    id : str
        A base identifier for the saved images, which will be augmented with indices for each image.

    Notes
    -----
    The function saves images with filenames based on the 'id' and indices for batch, modality, and segment.
    Each image represents a segment of the motion data.
    """

    [b, m, seg, c, h, w] = X.size()
    for jb in range(b):
        for jm in range(m):
            for jn in range(seg):

                file_ID = f"{id}_b{jb}_m{jm}_n{jn}"
                T.ToPILImage()(X[jb, jm, jn]).save(
                    f'{save_folder}/{file_ID}_MDM.png', mode='png')


def vis_MDM(X, X_MOT, save_folder, id):
    """
    Visualize and save motion data matrices (MDM) and corresponding original data as images.

    This function iterates through batches, modalities, and segments of the input tensors X and X_MOT, converting each
    segment into an image and saving it to the specified folder. It's typically used for debugging or understanding
    the motion and original data processed by the model.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor containing original data. Expected to have dimensions [batch, modality, segment, time, channel, height, width].
    X_MOT : torch.Tensor
        The input tensor containing motion data. Expected to have dimensions [batch, modality, segment, channel, height, width].
    save_folder : str
        The path to the folder where the images should be saved.
    id : str
        A base identifier for the saved images, which will be augmented with indices for each image.

    Notes
    -----
    The function saves images with filenames based on the 'id' and indices for batch, modality, and segment.
    Each image represents a segment of the original or motion data. The function saves multiple versions of the motion data
    and one version of the original data for each segment.
    """

    [b, m, n, t, c, h, w] = X.size()
    [b, m, n, c, h, w] = X_MOT.size()
    for jb in range(b):
        for jm in range(m):
            for jn in range(n):

                file_ID = f"{id}_b{jb}_m{jm}_n{jn}"
                T.ToPILImage()(X_MOT[jb, jm, jn]).save(
                    f'{save_folder}/{file_ID}_MDM.png', mode='png')
                T.ToPILImage()(
                    X[jb, jm, jn, t//2]).save(f'{save_folder}/{file_ID}_X.png', mode='png')

                T.ToPILImage()(X_MOT[jb, jm, jn, 0:1]).save(
                    f'{save_folder}/{file_ID}_MDM_0.png', mode='png')
                T.ToPILImage()(X_MOT[jb, jm, jn, 1:2]).save(
                    f'{save_folder}/{file_ID}_MDM_1.png', mode='png')
                T.ToPILImage()(X_MOT[jb, jm, jn, 2:3]).save(
                    f'{save_folder}/{file_ID}_MDM_2.png', mode='png')


class X_input(torch.nn.Module):

    """
    X_input is responsible for processing and preparing input data for the model. It handles different types of inputs
    like video, audio, and motion, applying necessary transformations or preprocessing steps.

    Parameters
    ----------
    param_TSM : dict
        A dictionary containing parameters for the model, specifically related to how input data should be handled.

    Attributes
    ----------
    input_mode : int
        Specifies how different input types should be combined or processed.
    n_video_segments : int
        The number of segments for video input.
    n_audio_segments : int
        The number of segments for audio input.
    n_motion_segments : int
        The number of segments for motion input, if applicable.

    Methods
    -------
    forward(x_video, x_audio)
        Defines the forward pass of the input processing, combining and transforming video, audio, and motion inputs.
    """

    def __init__(self, param_TSM):
        super(X_input, self).__init__()

        self.input_mode = param_TSM["main"]["input_mode"]
        self.n_video_segments = param_TSM["video_segments"]  # [8, 8, 1]
        self.n_audio_segments = param_TSM["audio_segments"]  # [8, 8, 1]
        self.n_motion_segments = 0

        if param_TSM["motion"]:
            self.MDF = MDFM(param_TSM["motion_param"])
            self.n_motion_segments = self.n_video_segments

        print("self.n_motion_segments", self.n_motion_segments)

    def forward(self, x_video, x_audio):
        # print("X_input:", x_video.size(),  x_audio.size())
        # exit()

        with torch.no_grad():

            if self.n_motion_segments > 0:
                xMOT = self.MDF(x_video)
                # print("x_video", x_video.size())
                # print("xMOT", xMOT.size())
                # vis_MDM(x_video, xMOT, "logs/vis_MDM", "xMOT")
                # exit()

                if self.n_video_segments > 0:
                    k_frames = x_video.size()[3]
                    if k_frames > 1:
                        xRGB = x_video[:, :, :, k_frames // 2]
                    else:
                        xRGB = x_video
                    # print("xRGB", xRGB.size())

                    if self.input_mode == 1:
                        x = torch.cat((xRGB, xMOT), dim=2)
                        # vis_MDM2(x, "logs/vis_MDM", "xMODE1")
                        # exit()
                    elif self.input_mode == 2:
                        x = torch.stack([torch.stack([xRGB[:, :, i], xMOT[:, :, i]])
                                        for i in range(self.n_video_segments)])
                        x = x.view(
                            (-1,) + (x.size()[-5:])).permute((1, 2, 0, 3, 4, 5))
                        # vis_MDM2(x, "logs/vis_MDM", "xMODE2")
                        # exit()
                    else:
                        sys.exit(
                            f'VCM_input: incorrect mode: {self.input_mode}')
                else:
                    x = xMOT

            elif self.n_video_segments > 0:
                x = torch.squeeze(x_video, -4)

            if self.n_audio_segments > 0:
                if self.n_video_segments > 0:
                    x = torch.cat((x, x_audio), dim=2)
                else:
                    x = x_audio

        # print("x", x.size())
        # exit()
        return x
