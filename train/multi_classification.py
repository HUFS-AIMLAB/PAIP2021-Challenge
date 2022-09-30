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
import random
import os
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

def parse_args():
    parser = argparse.ArgumentParser(description = "Train Model Organ Specific for Probability Map")
    parser.add_argument('--root_dir', type = str, help = "Patch(Random Extract) Directory")
    parser.add_argument('--result_dir', type = str, help = "Save Model Parameter & Loss")
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--level', type = int, help = "level 0 : 20X, level 1 : 5X")
    parser.add_argument('--organ', type = str, help = "'all, col', 'pan', 'pros'")
    parser.add_argument('--seed', type = int, default = 42)
    return parser.parse_args()


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars\n
        torch.backends.cudnn.deterministic = True  #needed\n
        torch.backends.cudnn.benchmark = False


def path_list(root_dir):
    #/nfs/paip2021/paip2021/docker_test/Col_0001/random/level_1/class1/img
    data_list = list()
    level_dim = str(args.level)
    label_value = {'class0': 0, 'class1': 1, 'class2': 2, 'class3': 3}
    for patient in sorted(os.listdir(os.path.join(root_dir))):
        if patient.split('_')[0] not in ['Col', 'Pan', 'Pros']:
            continue
        if args.organ != 'all' and patient.split('_')[0] != args.organ.title():
            continue
        for label in sorted(os.listdir(os.path.join(root_dir, patient, 'random', f" level_{level_dim}"))):
            if 'class' in label:
                for image in sorted(os.listdir(os.path.join(root_dir, patient, 'random', f" level_{level_dim}", label, "img"))):
                    if image.split('.')[-1] != 'png':
                        continue
                    else:
                        case = {
                            'image' : os.path.join(root_dir, patient, 'random', f" level_{level_dim}", label, "img", image),
                            'label' : label_value[label]
                        }
                        data_list.append(case)
            
    return data_list


if __name__ == '__main__':
    global args
    args = parse_args()
    random_seed(args.seed, True)
    main()