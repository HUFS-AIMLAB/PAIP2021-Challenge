import os
import glob
import re
import time
import copy
import random
import math
import sys

import cv2
import numpy as np
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse

import openslide
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description = 'Make Patches PAIP 2021 Patch Dataset')
    parser.add_argument('--svs_load_dir', type = str, help = 'Challenge Whole Slide Images dir')
    parser.add_argument('--xml_load_dir', type = str, help = "Challenge Annotation dir")
    parser.add_argument('--save_dir', type = str, help = 'Patch Save dir')
    parser.add_argument('--psize', type = int, default = 224, help = 'Patch Size')
    parser.add_argument('--max_patches', type = int, default = 2000, help = 'maximum patches for each class')
    parser.add_argument('--seed', type = int, default = 42)
    return parser.parse_args()

def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

class Preprocessor():
    def __init__(self):
        # ...in process