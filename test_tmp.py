import math

import torch
from torch import nn
from torch.nn.init import normal_, constant_
import torchvision
import os
from re import search

import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

from numpy.random import randint
import random



from math import e, pi


import torch
import torch.nn as nn

import timm



indexes = random.sample(range(10), 4)
print("indexes", indexes)

import sys

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", type=str)
args = parser.parse_args()
input_file = args.f
lines = [x.strip().split(' ') for x in open(input_file)]

f1 = open(f"{input_file}_p1", 'w')
f2 = open(f"{input_file}_p2", 'w')
for i, element in enumerate(lines):
    if i < len(lines) // 2:
        f1.write(element[0] + "\n")
    else:
        f2.write(element[0] + "\n")
f1.close()
f2.close()




















