import os
import sys
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.model import EfficientNetB0

def parse_args():
    parser = argparse.ArgumentParser(description = "Train Model Organ Specific for Probability Map")
    parser.add_argument('--root_dir', type = str, help = "Patch(Random Extract) Directory")
    parser.add_argument('--result_dir', type = str, help = "Save Model Parameter & Loss")
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 1e-3)
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
    def __init__(self, path_list, normalization, transform = None):
        self.path_list = path_list
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.path_list[index]['image'])
        image = image.convert("RGB")
        image = np.array(image)
        label = torch.tensor(self.path_list[index]['label']).type(torch.uint8)

        if self.transform:
            augmented = self.transform(image = image)
            image = augmented['image']

        data = {'image' : image, 'label' : label.item()}

        return data

    def __len__(self):
        return len(self.path_list)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_list = path_list(args.root_dir)
    train_list, valid_list = train_test_split(data_list, test_size = 0.1, shuffle=True)
    print(f"train_list : {len(train_list)}, valid_list : {len(valid_list)}")

    aug_func = strong_aug(p=0.8)
    train_aug = A.Compose([
        aug_func,
        A.Normalize(),
        ToTensorV2
    ])
    valid_aug = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    trainset = MyDataset(train_list, transform = train_aug)
    validset = MyDataset(valid_list, transform = valid_aug)
    train_loader = torch.utils.data.DataLoader(
        trainset,batch_size = args.batch_size, shuffle = True, 
        num_workers = args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        validset,batch_size = args.batch_size, shuffle = False, 
        num_workers = args.num_workers)

    # model = timm.create_model("tf_efficientnet_b0_ns", pretrained = True, num_classes = 4)
    model = EfficientNetB0(pre_trained = True, num_classes = 4)
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_train = np.array([])
    loss_valid = np.array([])
    accs_train = np.array([])
    accs_valid = np.array([])
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(args.num_epochs):
        stime = time.time()
        model.train()
        i=0
        l_epoch = 0
        correct = 0
        l_epoch_val = 0
        for item in tqdm(train_loader):
            i=i+1
            image, y = item['image'].to(device), item['label'].type(torch.long).to(device)
            y_hat = model(image)
            y_hat= F.softmax(y_hat, dim = 1)
            l=loss(y_hat,y)
            correct += (y_hat.argmax(dim=1)==y).sum()
            l_epoch+=l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        loss_train = np.append(loss_train,l_epoch.cpu().detach().numpy()/i)
        accs_train = np.append(accs_train,correct.cpu()/np.float(len(trainset)))

        correct = 0
        i = 0
        model.eval()
        with torch.no_grad():
            for item in tqdm(valid_loader):
                i +=1
                image, y = item['image'].to(device), item['label'].to(device)
                y_hat=optimizer(image)
                y_hat= F.softmax(y_hat, dim = 1)
                l = loss(y_hat, y)
                correct += (y_hat.argmax(dim=1)==y).sum()
                l_epoch_val += l
        accs_valid = np.append(accs_valid,correct.cpu()/np.float(len(validset)))
        loss_valid = np.append(loss_valid, l_epoch_val.cpu().detach().numpy()/i)

        if (correct.cpu()/np.float(len(validset))) > best_metric:
            best_metric = correct.cpu()/np.float(len(validset))
            best_metric_epoch = epoch
            torch.save(model.state_dict(), f"{args.result_dir}/best_model_organ_{args.organ}_level_{str(args.level)}.pth") # pan, pros
            print("saved new best metric model")

        if True:
            fig = plt.figure(figsize = (12, 6))
            ax = fig.add_subplot(1,2,1)
            plt.plot(loss_train,label='train loss')
            plt.plot(loss_valid, label='valid loss')
            plt.legend(loc='lower left')
            plt.title('epoch: %d '%(epoch+1))

            ax = fig.add_subplot(1,2,2)
            plt.plot(accs_train,label='train accuracy')
            plt.plot(accs_valid,label='valid accuracy')
            plt.legend(loc='lower left')
            plt.pause(.0001)
            plt.show()
            fig.savefig(f"{args.result_dir}/loss_organ_{args.organ}_level_{str(args.level)}.png") #pan, pros

            print('train loss: ',loss_train[-1])
            print('valid loss: ', loss_valid[-1])
            print('train accuracy: ',accs_train[-1])
            print('valid accuracy: ',accs_valid[-1])
            print(f"best metric epoch : {best_metric_epoch}, best metric accuracy : {best_metric}")
        print(f"1 epoch time : {(time.time() - stime) / 60} min")


if __name__ == '__main__':
    global args
    args = parse_args()
    random_seed(args.seed, True)
    main()