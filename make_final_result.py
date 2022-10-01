import os
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import argparse

import timm
import segmentation_models_pytorch as smp
import openslide

from inference.pni_segmentation import SegInferer

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars\n
        torch.backends.cudnn.deterministic = True  #needed\n
        torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description = "Train Model Organ Specific for Probability Map")
    parser.add_argument('--root_dir', type = str, help = "Whole Slide Images Directory")
    parser.add_argument('--result_dir', type = str, help = "Model & Result Directory")

    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--organ', type = str, help = "col, pros, pan")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--use_gpu', type = str2bool, default = True)

    return parser.parse_args()

def main():
    args = parse_args()
    random_seed(args.seed, True)
    if args.use_gpu:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
    
    model_path = defaultdict(str)
    for organ in ['col', 'pros', 'pan']:
        key_ = organ + '_' + '1'
        model_path[key_] = os.path.join(args.result_dir, 'seg', organ, f"level_1", 'checkpoint.pt')
    
    wsi_fns = []
    hmaps = []
    key_ = args.organ + '_' + '1'
    model = smp.Unet(encoder_name = "timm-efficientnet-b0", encoder_weights = "noisy-student",in_channels = 3, classes = 2)
    model.load_state_dict(torch.load(model_path[key_]))
    model = model.to(device).eval()


    inferer = SegInferer(args, model, device)

    for wsi in sorted(os.listdir(args.root_dir)):
        if wsi.split('.')[-1] == 'svs':
            if args.organ.title() == wsi:
                wsi_fns.append(os.path.join(args.root_dir, wsi))
    
    for hmap in sorted(os.listdir(os.path.join(args.result_dir, 'probmap', args.organ))):
        if hmap.split('.')[-1] == 'npy':
            hmaps.append(os.path.join(args.result_dir, 'probmap', args.organ, hmap))
    
    for wsi_fn, hmap in zip(wsi_fns, hmaps):
        name = wsi_fn.split('/')[-1].split('.')[0]
        slide = openslide.OpenSlide(wsi_fn)
        overlay = np.load(hmap)
        overlay = np.where(overlay >= 0.5, 1, 0)
        overlay = overlay.astype(np.uint8)
        result = inferer.read_wsi_seg(slide, overlay)
        ppath = Path(os.path.join(args.result_dir, 'final_result', args.organ, f"{name}.npy"))
        ppath.parent.mkdir(parents = True, exist_ok = True)
        np.save(str(ppath), result)


if __name__ == '__main__':
    main()


