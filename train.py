import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import random
import numpy as np
import argparse

import timm
import monai
import segmentation_models_pytorch as smp

from trainer.trainer_encoder import EncoderTrainer
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
    parser.add_argument('--root_dir', type = str, help = "Patch(Random Extract) Directory")
    parser.add_argument('--model_dir', type = str, help = "save model directory")
    parser.add_argument('--train_mode', type = str, help = "clf: classification, seg: segmentation")
    parser.add_argument('--train_type', type = str, help = "encoder, [for probmap]: col, pan, pros")

    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--train_epochs', type = int, default = 100)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--level', type = int, help = "level 0 : 20X, level 1 : 5X")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--use_gpu', type = str2bool, default = True)

    parser.add_argument('--patience', type = int, default = 3, help = "Early Stop patience")
    parser.add_argument('--random_sampling', type = str2bool, default = True)
    return parser.parse_args()


def main():
    args = parse_args()
    random_seed(args.seed, True)
    if args.use_gpu:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    if args.train_mode == 'clf':
        # model = EfficientNetB0(pre_trained = True, num_classes = 4)
        model = timm.create_model("tf_efficientnet_b0_ns", pretrained = True, num_classes = 4)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.train_mode == 'seg':
        encoder_path = os.path.join(args.model_dir, 'clf/all', f"level_{args.level}/checkpoint.pt")
        model = smp.Unet(encoder_name = "timm-efficientnet-b0", encoder_weights = "noisy-student", in_channels = 3, classes = 2)
        model.encoder.load_state_dict(torch.load(encoder_path))
        # criterion = monai.losses.DiceLoss(sigmoid = True, include_background = False)
        criterion = monai.losses.DiceLoss(softmax = True, to_onehot = True, include_background = True)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    if args.train_mode == 'clf':
        trainer = EncoderTrainer(args, model, optimizer, criterion, device)
    elif args.train_mode == 'seg':
        trainer = UNetTrainer(args, model, optimizer, criterion, device)


    trainer.training()


if __name__ == '__main__':
    main()
    