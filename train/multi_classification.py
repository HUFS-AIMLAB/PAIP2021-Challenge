import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
import timm

import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomResizedCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,Normalize,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, GaussNoise,
    RGBShift, RandomRain, RandomSnow, RandomShadow, RandomFog, ElasticTransform,Normalize,Sharpen,Sequential
)
from albumentations.pytorch import ToTensorV2

# In Progress