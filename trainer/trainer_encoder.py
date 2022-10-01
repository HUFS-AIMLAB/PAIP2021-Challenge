import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
        image = Image.open(self.path_list[index]['image'])
        image = image.convert("RGB")
        image = np.array(image)
        label = torch.tensor(self.path_list[index]['label']).type(torch.uint8)

        if self.transform:
            print(self.transform)
            print(type(image))
            print(image.shape)
            augmented = self.transform(image = image)
            image = augmented['image']

        data = {'image' : image, 'label' : label.item()}

        return data

    def __len__(self):
        return len(self.path_list)


class EncoderTrainer():
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
        label_value = {'class0': 0, 'class1': 1, 'class2': 2, 'class3': 3}
        for patient in sorted(os.listdir(os.path.join(root_dir))):
            if patient.split('_')[0] not in ['Col', 'Pan', 'Pros']:
                continue
            if self.args.train_type != 'all' and patient.split('_')[0] != self.args.train_type.title():
                continue
            for label in sorted(os.listdir(os.path.join(root_dir, patient, 'random', f"level_{level_dim}"))):
                if 'class' in label:
                    for image in sorted(os.listdir(os.path.join(root_dir, patient, 'random', f"level_{level_dim}", label, "img"))):
                        if image.split('.')[-1] != 'png':
                            continue
                        else:
                            case = {
                                'image' : os.path.join(root_dir, patient, 'random', f"level_{level_dim}", label, "img", image),
                                'label' : label_value[label]
                            }
                            data_list.append(case)
                
        return data_list

    def validation(self, valid_loader, length):
        self.model.eval()
        valid_iterator = tqdm(
        valid_loader, desc="VALIDATION (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        valid_loss = []
        iter_count = 0
        correct = 0
        max_iterations = len(valid_loader)
        for item in tqdm(valid_iterator):
            iter_count += 1
            image, label = item['image'].to(self.device), item['label'].type(torch.long).to(self.device)
            pred = F.softmax(self.model(image), dim = 1)
            loss = self.criterion(pred, label)
            valid_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iter_count, max_iterations, loss.item())
            )
            correct += (pred.argmax(dim = 1) == label).sum().cpu()
            valid_loss.append(loss.item())
        valid_loss = np.average(valid_loss).item()
        valid_acc = correct / np.float32(length)
        return valid_loss, valid_acc

    def training(self):
        data_list = self.path_list()
        train_list, valid_list = train_test_split(data_list, test_size = 0.1, shuffle=True)
        print(f"[INFO] train_list: {len(train_list)}, valid_list: {len(valid_list)}")
        
        albu_aug = strong_aug(p = 0.8)
        train_aug = A.Compose([
            albu_aug,
            A.Normalize(),
            ToTensorV2
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
                image, label = item['image'].to(self.device), item['label'].type(torch.long).to(self.device)
                pred = F.softmax(self.model(image), dim = 1)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                train_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (iter_count, max_iterations, loss.item())
                )
                correct += (pred.argmax(dim = 1) == label).sum().cpu()
                train_loss.append(loss.item())
                
            train_loss = np.average(train_loss).item()
            train_acc = correct / np.float32(len(trainset))
            valid_loss, valid_acc = self.validation(valid_loader, len(validset))

            print(f"Epoch: {epoch} | Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
            print(f"Epoch: {epoch} | Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")
            
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            


                
