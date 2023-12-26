import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torch
import torchaudio
import torchvision
import torchvision.transforms as T
import torchaudio.transforms as AT
import torch.nn.functional as F
from collections import Counter
import random


def load_image(directory, idx, image_tmpl):
    """
    Load an image given a directory and index.

    Parameters:
    ----------
    directory : str
        Directory where images are located.
    idx : int
        Index of the image to load.
    image_tmpl : str
        Template string for image file naming.

    Returns:
    -------
    torch.Tensor
        The loaded image as a tensor.
    """
    try:
        return T.ToTensor()(Image.open(f"{directory}/{image_tmpl.format(idx)}").convert('RGB'))
    except Exception as e:
        print(f'Error loading image: {directory}/{image_tmpl.format(idx)}')
        raise e


def get_video_x(record, args,  mode_train_val):
    """
    Get video data for a given record and processing mode.

    Parameters:
    ----------
    record : dict
        A dictionary containing information about the video record.
    args : dict
        Configuration arguments including paths and processing parameters.
    mode_train_val : str
        Mode of operation, can be 'training', 'validation', or 'test'.

    Returns:
    -------
    torch.Tensor
        The processed video data as a tensor.
    """

    video_img_param = args["dataset"]["video_img_param"]
    video_augmentation_param = args["dataset"]["video_augmentation"]

    num_segments, k_frames = args["video_segments"], args["segment_frames"]
    clip_length = args["emotion_jumps"]["clip_length"]
    fps = args["dataset"]["fps"]
    # print("get_video_x", clip_length, fps)

    size, i_add = size_i_add_from_record(record, fps, clip_length)
    # print("get_video_x", size, i_add)

    # size = record["imagefolder_size"]

    if mode_train_val == "training":
        indices = sample_indices_training(size, num_segments, k_frames, i_add)
        # indices = sample_indices_stride(size, num_segments, k_frames, frame_stride)
        # print("size, num_segments, k_frames", size, num_segments, k_frames)
        # print("indices", indices)
        # exit()
    elif mode_train_val == "validation":
        # indices = sample_indices_stride(size, num_segments, k_frames, frame_stride)
        indices = sample_indices_validation(
            size, num_segments, k_frames, i_add)
    else:
        indices = sample_indices_validation(
            size, num_segments, k_frames, i_add)

    image_tmpl = video_img_param["image_tmpl"]
    X = torch.stack([torch.stack([load_image(record["imagefolder"], indices[s] + i -
                    k_frames // 2, image_tmpl) for i in range(k_frames)]) for s in range(num_segments)])

    if mode_train_val == "training":
        X = augmentation_video(
            X, video_augmentation_param["scales"], video_img_param["img_output_size"], video_augmentation_param)
    elif mode_train_val == "validation":
        X = validation_transform(
            X, video_img_param["img_input_size"], video_img_param["img_output_size"])
    else:
        X = validation_transform(
            X, video_img_param["img_input_size"], video_img_param["img_output_size"])
    # print("X", X.size())
    X = torch.unsqueeze(X, 0)
    return X


def vis_MDM(X, X_MOT, save_folder, id):

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


def vis_X(X,  save_folder, id):
    print("vis_X", X.size())
    X = X.view((-1, ) + X.size()[-3:])
    (n, c, h, w) = X.size()
    print("(n, c, h, w):", n, c, h, w)

    for jb in range(n):
        file_ID = f"{id}_b{jb}"
        T.ToPILImage()(X[jb]).save(
            f'{save_folder}/{file_ID}_MDM.png', mode='png')


def scales_pairs(scales):
    pairs = []
    for i1 in range(len(scales)):
        for i2 in range(len(scales)):
            if (abs(i1 - i2) > 1):
                continue
            pairs.append((scales[i1], scales[i2]))
    return pairs


def augmentation_video(X, scales, img_output_size, param):
    pairs = scales_pairs(scales)
    (C, H, W) = X.size()[-3:]
    # print("X.size()", X.size())
    S = min(H, W)
    # print("pairs", pairs)

    new_crop_size = random.choice(pairs)
    # print("new_crop_size", new_crop_size)

    new_crop_size = (int(S * new_crop_size[0]),  int(S * new_crop_size[1]))
    transform_list = [T.RandomCrop(new_crop_size), T.Resize(
        (img_output_size, img_output_size))]

    if param["RandomHorizontalFlip"]:
        transform_list.append(T.RandomHorizontalFlip(p=0.5))
    if param["ColorJitter"]:
        transform_list.append(T.ColorJitter(brightness=(
            0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.5))
    if param["RandomGrayscale"] > 0:
        transform_list.append(T.RandomGrayscale(
            p=float(param["RandomGrayscale"])))
    if param["GaussianBlur"]:
        transform_list.append(T.GaussianBlur(9, (0.5, 2)))
    transforms = torch.nn.Sequential(*transform_list)
    X = transforms(X.view((-1, C, H, W))).view(X.size()
                                               [:-3] + (C, img_output_size, img_output_size))
    return X


def validation_transform(X, img_input_size, img_output_size):

    (C, H, W) = X.size()[-3:]
    s = img_input_size  # int(img_output_size * 1.20)#
    if W >= H:
        new_size = (s, int((img_input_size * W / H)))
    else:
        new_size = (int((s * H / W)), s)
    transform_list = [T.Resize(new_size), T.CenterCrop(img_output_size)]
    transforms = torch.nn.Sequential(*transform_list)
    X = transforms(X.view((-1, C, H, W))).view(X.size()
                                               [:-3] + (C, img_output_size, img_output_size))

    return X


def size_i_add_from_record(record, fps, t_length):
    size = record["imagefolder_size"]
    t_start = record["t"]

    i_add = 0
    if t_length > 0:
        i_add = t_start * fps
        size = t_length * fps
        last_i = t_start * fps + t_length * fps
        if last_i > record["imagefolder_size"]:
            size = record["imagefolder_size"] - t_start * fps
            if size < 10:
                print("size_i_add_from_record:", size,
                      record["imagefolder"], t_start, t_length, record["imagefolder_size"])
                exit()
    return size, i_add


def sample_indices_training(size, num_segments, k_frames, i_add):

    average_duration = size // (2*num_segments)
    tick = size / float(num_segments)
    rv = random.uniform(-0.5, 0.5)
    indices = np.array([int(tick / 2.0 + tick * x + rv) +
                       randint(-average_duration, average_duration+1) for x in range(num_segments)])
    indices[indices < k_frames // 2] = k_frames // 2
    indices[indices > size - 1 - k_frames // 2] = size - (k_frames // 2) - 1
    return indices + 1 + i_add


def sample_indices_validation(size, num_segments, k_frames, i_add):
    tick = size / float(num_segments)
    indices = np.array([int(tick / 2.0 + tick * x)
                       for x in range(num_segments)])
    indices[indices < k_frames//2] = k_frames//2
    indices[indices > size - 1 - k_frames // 2] = size - (k_frames // 2) - 1
    return indices + 1 + i_add
