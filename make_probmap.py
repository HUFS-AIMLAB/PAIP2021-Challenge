import os
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import argparse

import timm
import openslide

from inference.pni_probmap import Inferer

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
    for l in [0, 1]:
        for organ in ['col', 'pros', 'pan']:
            key_ = organ + '_' + l
            model_path[key_] = os.path.join(args.result_dir, 'clf', organ, f"level_{l}", 'checkpoint.pt')
    
    wsi_fns = []
    models = {}
    for l in [0, 1]:
        key_ = args.organ + '_' + l
        model = timm.create_model("tf_efficientnet_b0_ns", pretrained = False, num_classes = 4)
        model.load_state_dict(torch.load(model_path[f"{args.organ}_{l}"]))
        model = model.to(device).eval()
        models[key_] = model

    inferer_0 = Inferer(args, 0, models[f"{args.organ}_0"], device)
    inferer_1 = Inferer(args, 1, models[f"{args.organ}_1"], device)

    for wsi in sorted(os.listdir(args.root_dir)):
        if wsi.split('.')[-1] == 'svs':
            if args.organ.title() == wsi:
                wsi_fns.append(os.path.join(args.root_dir, wsi))
    
    for wsi_fn in wsi_fns:
        slide = openslide.OpenSlide(wsi_fn)
        probmap_0 = inferer_0.read_wsi_clf(slide)
        probmap_1 = inferer_1.read_wsi_clf(slide)
        probmap = (probmap_0 + probmap_1) / 2
        ppath = Path(os.path.join(args.result_dir, "probmap", args.organ, "probmap.npy"))
        ppath.parent.mkdir(parents = True, exist_ok = True)
        np.save(str(ppath), probmap)


if __name__ == '__main__':
    main()


