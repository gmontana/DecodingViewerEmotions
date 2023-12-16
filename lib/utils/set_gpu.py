import shutil
import json
import os
import torch
import torch.nn as nn
from collections import Counter
from re import search





def set_model_DataParallel( args, model ):

    DataParallel = False
    cuda_ids = args["cuda_ids"]
    if len(cuda_ids) > 1:

        print(f"Let's use {cuda_ids} GPUs out of {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=cuda_ids , output_device=cuda_ids[0], dim=0)
        DataParallel = True

    return model, DataParallel


def set_cuda_device(args_in, args ):

    if args_in.cuda_ids:
       cuda_device_ids = args["cuda_ids"]  # gpu to be used
       #args["cuda_ids"] = cuda_device_ids
    else:
       cuda_device_ids = [0] #set here default GPU ids , normally [0,1,2,..,16]

    device = torch.device(f"cuda:{cuda_device_ids[0]}" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    device_id =cuda_device_ids[0]
    return device , device_id, args


