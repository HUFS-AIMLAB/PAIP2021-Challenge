import os
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import monai

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.tools import EarlyStopping

def strong_aug(p=0.5):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.RandomBrightnessContrast(brightness_limit = [-0.2, 0.2]),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit = [1000,1050]),
        ], p=0.7),
        A.HueSaturationValue(p=0.7),
    ], p=p)


class MyDataset(Dataset):
    def __init__(self, path_list, transform = None):
        self.path_list = path_list
        self.transform = transform


    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index]['image'])
        mask = cv2.imread(self.path_list[index]['mask'], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask'][np.newaxis, :, :]

        return image, mask


    def __len__(self):
        return len(self.path_list)


class UNetTrainer():
    def __init__(self, args, model, optimizer, criterion, device):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
    

    def path_list(self):
        data_list = list()
        root_dir = self.args.root_dir
        level_dim = str(self.args.level)

        for patient in sorted(os.listdir(os.path.join(root_dir))):
            if patient.split('_')[0] not in ['Col', 'Pan', 'Pros']:
                continue
            if self.args.train_type != 'all' and patient.split('_')[0] != self.args.train_type.title():
                continue
            for image in sorted(os.listdir(os.path.join(root_dir, patient, 'sw', f"level_{level_dim}", 'pni', "img_sw"))):
                if image.split('.')[-1] != 'png':
                    continue
                else:
                    if self.args.random_sampling and random.randint(0, 9) < 9:
                        continue
                    case = {
                        'image' : os.path.join(root_dir, patient, 'sw', f"level_{level_dim}", 'pni', "img_sw", image),
                        'mask' : os.path.join(root_dir, patient, 'sw', f"level_{level_dim}", 'pni', "mask_sw", image)
                    }
                    data_list.append(case)
                
        return data_list


    def validation(self, valid_loader, dice_metric, activate, discrete, inferer):
        self.model.eval()
        valid_iterator = tqdm(
        valid_loader, desc="VALIDATION (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        valid_loss = []
        valid_dice = []
        iter_count = 0
        correct = 0
        max_iterations = len(valid_loader)
        with torch.no_grad():
            for item in tqdm(valid_iterator):
                iter_count += 1
                image, label = item[0].to(self.device), item[1].type(torch.long).to(self.device)
                pred = inferer(inputs = image, network = self.model)
                loss = self.criterion(pred, label)
                valid_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (iter_count, max_iterations, loss.item())
                )
                pred = activate(pred)
                pred = discrete(pred)
                dice_value = dice_metric(y_pred = pred, y = label)
                valid_loss.append(loss.item())
                valid_dice.append(dice_value.squeeze().mean().detach().cpu().numpy())
        valid_loss = np.average(valid_loss).item()
        valid_dice = np.average(valid_dice).item()
        return valid_loss, valid_dice


    def training(self):
        data_list = self.path_list()
        train_list, valid_list = train_test_split(data_list, test_size = 0.1, shuffle=True)
        print(f"[INFO] train_list: {len(train_list)}, valid_list: {len(valid_list)}")
        
        albu_aug = strong_aug(p = 0.8)
        train_aug = A.Compose([
            albu_aug,
            A.Normalize(),
            ToTensorV2(),
        ])
        valid_aug = A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ])

        trainset = MyDataset(train_list, transform = train_aug)
        validset = MyDataset(valid_list, transform = valid_aug)

        train_loader = DataLoader(
            trainset,batch_size = self.args.batch_size, shuffle = True, 
            num_workers = self.args.num_workers, pin_memory = True)
        valid_loader = DataLoader(
            validset,batch_size = self.args.batch_size, shuffle = False, 
            num_workers = self.args.num_workers, pin_memory = True)

        self.model = self.model.to(self.device)

        dice_metric = monai.metrics.DiceMetric(include_background = False, reduction = 'mean')
        activate = monai.transforms.Activations(softmax = True)
        discrete = monai.transforms.AsDiscrete(threshold = 0.5)
        inferer = monai.inferers.SimpleInferer()
        
        print(f"[INFO] training start")
        train_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        early_stopping = EarlyStopping(args = self.args, verbose = True)

        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            train_loss = []
            iter_count = 0
            correct = 0
            max_iterations = len(train_loader)
            for item in tqdm(train_iterator):
                iter_count += 1
                self.optimizer.zero_grad()
                image, label = item[0].to(self.device), item[1].type(torch.long).to(self.device)
                pred = self.model(image)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                train_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (iter_count, max_iterations, loss.item())
                )
                train_loss.append(loss.item())
                
            train_loss = np.average(train_loss).item()
            valid_loss, valid_dice = self.validation(valid_loader, dice_metric, activate, discrete, inferer)

            print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f}")
            print(f"Epoch: {epoch} | Valid Loss: {valid_loss:.3f}, Valid Dice Score: {valid_dice:.3f}")
            
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            


                
