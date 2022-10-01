import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import random
import numpy as np
import argparse

import monai

from trainer.trainer_encoder import EncoderTrainer, ProbmapTrainer
from trainer.trainer_unet import UNetTrainer
from model.model import EfficientNetB0, UNetEfficientNet



def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars\n
        torch.backends.cudnn.deterministic = True  #needed\n
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description = "Train Model Organ Specific for Probability Map")
    parser.add_argument('--root_dir', type = str, help = "Patch(Random Extract) Directory")
    parser.add_argument('--model_dir', type = str, help = "save model directory")
    parser.add_argument('--train_mode', type = str, help = "clf: classification, seg: segmentation")
    parser.add_argument('--train_type', type = str, help = "encoder, [for probmap]: col, pan, pros")

    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--level', type = int, help = "level 0 : 20X, level 1 : 5X")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--use_gpu', type = bool, default = True)
    return parser.parse_args()


def main():
    args = parse_args()
    random_seed(args.seed, True)
    if args.use_gpu:
        device = torch.devcie("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
        
    if args.train_mode == 'clf':
        model = EfficientNetB0(pre_trained = True, num_classes = 4)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.train_mode == 'seg':
        model = UNetEfficientNet(num_classes = 1, encoder_path = os.path.join(args.model_dir, 'clf/encoder', f"level_{args.level}/checkpoint.pt"))
        criterion = monai.losses.DiceLoss(sigmoid = True)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    if args.train_mode == 'clf':
        if args.train_type == 'encoder':
            trainer = EncoderTrainer(args, model, optimizer, criterion, device)
        else:
            trainer = ProbmapTrainer(args, model, optimizer, criterion, device)
    elif args.train_mode == 'seg':
        trainer = UNetTrainer(args, model, optimizer, criterion, device)


    trainer.training()


if __name__ == '__main__':
    main()
    