import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2


class Inferer():
    def __init__(self, args, level, model, device):
        self.args = args
        self.psize = (224, 224)
        self.overlap_0 = (163, 163)
        self.overlap_1 = (208, 208)
        self.batch_size = self.args.batch_size
        self.device = device
        self.model = model
        self.level = level
        self.valid_aug = A.Compose([A.Normalize(), ToTensorV2(),])


    def read_wsi_clf(self, slide):
        max_x = slide.level_dimensions[self.level][0]
        max_y = slide.level_dimensions[self.level][1]
        num = 0
        
        out_fill_1 = np.zeros(self.psize)
        out_fill_0 = np.zeros((self.psize[0]//4, self.psize[1]//4))

        if self.level ==1:
            overlay = np.zeros([max_y, max_x])
            reference = np.zeros([max_y, max_x])
            overlap = self.overlap_1

        elif self.level==0:
            overlay = np.zeros([max_y//4, max_x//4])
            reference = np.zeros([max_y//4, max_x//4])
            overlap = self.overlap_0

        steps_x = int(math.ceil((max_x - overlap[0]) / float(self.psize[0] - overlap[0])))
        steps_y = int(math.ceil((max_y - overlap[1]) / float(self.psize[1] - overlap[1])))
        start_coords = list()
        
        for y in tqdm(range(0,steps_y)):
            for x in range(0, steps_x):
                x_start = x*(self.psize[0] - overlap[0])
                x_end = x_start + self.psize[0]
                y_start = y*(self.psize[1] - overlap[1])
                y_end = y_start + self.psize[1]
                
                if x_end > max_x:
                    x_start = max_x - self.psize[0]
                    x_end = max_x
                if y_end > max_y:
                    y_start = max_y - self.psize[1]
                    y_end = max_y
                    
                temp = np.array(slide.read_region((x_start*(4**(self.level)), y_start*(4**(self.level))), self.level, self.psize))
                temp = temp[:,:,:3].astype(np.uint8)
                temp = self.valid_aug(image = temp)['image'].unsqueeze(0).to(self.device).float()

                if self.level == 1:
                    start_coords.append((y_start,x_start))
                elif self.level == 0:
                    start_coords.append((y_start//4,x_start//4))
                
                cat = temp if num == 0 else torch.cat((cat, temp), dim = 0)
                num += 1
                
                if num == self.batch_size or (y == steps_y - 1 and x == steps_x - 1):
                    length = self.batch_size if num == self.batch_size else num
                    pred_fill = out_fill_1 if self.level == 1 else out_fill_0
                    pred = self.model(cat)
                    pred = F.softma(pred, dim = 1)
                    pred = pred.detech().cpu().numpy()
                    pred = pred[:, 1]
                    denom = 4 ** (1 - self.level)
                    for i in range(length):
                        pred_fill.fill(pred[i])
                        overlay[start_coords[i][0]:start_coords[i][0]+self.psize[0]//denom, start_coords[i][1]:start_coords[i][1]+self.psize[1]//denom] += pred_fill
                        reference[start_coords[i][0]:start_coords[i][0]+self.psize[0]//denom, start_coords[i][1]:start_coords[i][1]+self.psize[1]//denom] += 1
                    num = 0
                    start_coords = []

        output = overlay / reference
        return output

