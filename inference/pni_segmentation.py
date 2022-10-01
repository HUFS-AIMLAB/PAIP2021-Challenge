import numpy as np
import cv2
from tqdm import tqdm
import math

import torch
import monai

import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegInferer:
    def __init__(self, args, model, device):
        self.args = args
        self.psize = (224, 224)
        self.level = 1
        self.batch_size = 64
        self.overlap = (220, 220)
        self.model = model
        self.device = device
        self.valid_aug = A.Compose([A.Normalize(), ToTensorV2(),])
        self.post_transform = monai.transforms.Activations(softmax=True)

    def read_wsi_seg(self, slide, heatmap):
        max_x = slide.level_dimensions[self.level][0]
        max_y = slide.level_dimensions[self.level][1]

        num = 0

        overlay = np.zeros([max_y, max_x])
        reference = np.zeros([max_y, max_x])

        steps_x = int(
            math.ceil(
                (max_x - self.overlap[0]) / float(self.psize[0] - self.overlap[0])
            )
        )
        steps_y = int(
            math.ceil(
                (max_y - self.overlap[1]) / float(self.psize[1] - self.overlap[1])
            )
        )
        start_coords = list()

        for y in tqdm(range(0, steps_y)):
            for x in range(0, steps_x):
                x_start = x * (self.psize[0] - self.overlap[0])
                x_end = x_start + psize[0]
                y_start = y * (self.psize[1] - self.overlap[1])
                y_end = y_start + psize[1]

                if x_end > max_x:
                    x_start = max_x - self.psize[0]
                    x_end = max_x
                if y_end > max_y:
                    y_start = max_y - self.psize[1]
                    y_end = max_y

                heatmap_patch = heatmap[
                    y_start : y_start + self.psize[0], x_start : x_start + self.psize[1]
                ]

                if np.amax(heatmap_patch) == 1:

                    start_coords.append((y_start, x_start))

                    temp = np.array(
                        slide.read_region(
                            (
                                x_start * (4 ** (self.level)),
                                y_start * (4 ** (self.level)),
                            ),
                            self.level,
                            self.psize,
                        )
                    )
                    temp = temp[:, :, :3]
                    temp = temp.astype(np.uint8)
                    temp = temp[:, :, [2, 1, 0]]  # rgb2bgr
                    temp = (
                        self.valid_aug(image=temp)["image"]
                        .unsqueeze(0)
                        .to(self.device)
                        .float()
                    )

                    cat = temp if num == 0 else torch.cat((cat, temp), dim=0)
                    num += 1

                    if num == self.batch_size or (
                        y == steps_y - 1 and x == steps_x - 1
                    ):
                        length = self.batch_size if num == self.batch_size else num
                        pred = self.model(cat)
                        pred = self.post_transform(pred)[1].detach().cpu().numpy()
                        for i in range(length):
                            overlay[
                                start_coords[i][0] : start_coords[i][0] + self.psize[0],
                                start_coords[i][1] : start_coords[i][1] + self.psize[1],
                            ] += pred[i]
                            reference[
                                start_coords[i][0] : start_coords[i][0] + self.psize[0],
                                start_coords[i][1] : start_coords[i][1] + self.psize[1],
                            ] += 1
                            if i == length - 1:
                                num = 0
                                start_coords = []

        output = overlay / reference
        output = np.where(overlay >= 0.5, 1, 0)
        output = output.astype(np.uint8)
        output = cv2.bitwise_and(output, heatmap)
        output = output.astype(np.uint8)

        return output
