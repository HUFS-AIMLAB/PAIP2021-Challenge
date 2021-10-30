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
    parser = argparse.ArgumentParser(description = 'Make Patches PAIP 2021 Dataset')
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

def xml2mask(xml_fn, slide, level):
    """
    <XML Tree>
    Annotations (root)
    > Annotation
      > Regions
        > Region
          > Vertices
            > Vertex
              > X, Y
    <Label>
    nerve_without_tumor (contour): 1
    perineural_invasion_junction (line): 2
    nerve_without_tumor (bounding box): 11
    tumor_without_nerve (bounding box): 13
    nontumor_without_nerve (bounding box): 14
    """
    etree = ET.parse(xml_fn)

    # Height and Width Ratio
    src_w, src_h  = slide.level_dimensions[0]
    dest_w, dest_h = slide.level_dimensions[level]
    w_ratio = src_w / dest_w
    h_ratio = src_h / dest_h

    mask = np.zeros((dest_h, dest_w))

    annotations = etree.getroot()
    for annotation in annotations:
        label = int(annotation.get("Id"))

        cntr_pts = list()
        bbox_pts = list()

        regions = annotation.findall("Regions")[0]
        for region in regions.findall("Region"):
            pts = list()

            vertices = region.findall("Vertices")[0]
            for vertex in vertices.findall("Vertex"):
                x = round(float(vertex.get("X")))
                y = round(float(vertex.get("Y")))

                # Match target level coordinates
                x = np.clip(round(x / w_ratio), 0, dest_w)
                y = np.clip(round(y / h_ratio), 0, dest_h)

                pts.append((x, y))

            if len(pts) == 4:
                bbox_pts += [pts]
            else:
                cntr_pts += [pts]

        # Bounding box
        for pts in bbox_pts:
            pts = [np.array(pts, dtype=np.int32)]
            mask = cv2.drawContours(mask, pts, -1, label + 10, -1)
        for pts in cntr_pts:
            pts = [np.array(pts, dtype=np.int32)]
            # Curved line
            if label == 2:
                mask = cv2.polylines(mask, pts, isClosed=False, color=label, thickness=1)
            # Contour
            else:
                mask = cv2.drawContours(mask, pts, -1, label, -1)
    return mask

def make_patch(patch_save, slide, masks, name, level):
    max_x = slide.level_dimensions[level][0]
    max_y = slide.level_dimensions[level][1]

    idx_nerve= np.arange(len(np.where(masks == class_nerve)[0]))
    idx_pni = np.arange(len(np.where(masks == class_pni)[0]))
    idx_tumor = np.arange(len(np.where(masks == class_tumor)[0]))
    idx_benign = np.arange(len(np.where(masks == class_benign)[0]))
    
    np.random.shuffle(idx_nerve)
    np.random.shuffle(idx_pni)
    np.random.shuffle(idx_tumor)
    np.random.shuffle(idx_benign)

    y_nerve = np.where(masks==class_nerve)[0][idx_nerve]
    x_nerve = np.where(masks==class_nerve)[1][idx_nerve]
    y_pni = np.where(masks==class_pni)[0][idx_pni]
    x_pni = np.where(masks==class_pni)[1][idx_pni]
    y_tumor = np.where(masks==class_tumor)[0][idx_tumor]
    x_tumor = np.where(masks==class_tumor)[1][idx_tumor]
    y_benign = np.where(masks==class_benign)[0][idx_benign]
    x_benign = np.where(masks==class_benign)[1][idx_benign]
    
    num = 0
    idx_list = [idx_nerve, idx_pni, idx_tumor, idx_benign]
    coord_x = [x_nerve, x_pni, x_tumor, x_benign]
    coord_y = [y_nerve, y_pni, y_tumor, y_benign]
    for idx in idx_list:
        if num == 0:
            label = 'class0'
        elif num == 1:
            label = 'class1'
        elif num == 2:
            label = 'class2'
        elif num == 3:
            label = 'class3'
        cnt = 0
        if len(idx) > args.max_patches:
            for i in range(args.max_patches):
                x = coord_x[num][i]*(4**(level)) - (args.psize//2)*(4**(level))
                y = coord_y[num][i]*(4**(level)) - (args.psize//2)*(4**(level))
                img = np.array(slide.read_region((x,y), level, args.psize))
                img = img[:,:,:3]
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                mask_ = masks[coord_y[num][i] - (args.psize//2) : coord_y[num][i] + (args.psize//2), coord_x[num][i] - (args.psize//2) : coord_x[num][i] + (args.psize//2), np.newaxis]
                mask_ = mask_.astype(np.uint8)
                
                
                num2 = format(cnt, '07')
                cnt += 1
                ipath = Path(os.path.join(patch_save, name, str(level),f"{label}/img", f"{name}_{num2}.png"))
                mpath = Path(os.path.join(patch_save, name, str(level),f"{label}/mask", f"{name}_{num2}.png"))

                ipath.parent.mkdir(parents=True, exist_ok=True)
                mpath.parent.mkdir(parents=True, exist_ok=True)

                try:
                    cv2.imwrite(str(ipath), img)
                    cv2.imwrite(str(mpath), mask_)
                except:
                    print(img.shape)
                    print(mask_.shape)
                    print(label)
                    print(f"mask coord : {coord_y[num][i]}, {coord_x[num][i]}")
                    continue
        else:
            for i in range(len(idx)):
                x = coord_x[num][i]*(4**(level)) - (args.psize//2)*(4**(level))
                y = coord_y[num][i]*(4**(level)) - (args.psize//2)*(4**(level))
                img = np.array(slide.read_region((x,y), level, args.psize))
                img = img[:,:,:3]
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                mask_ = masks[coord_y[num][i] - (args.psize//2) : coord_y[num][i] + (args.psize//2), coord_x[num][i] - (args.psize//2) : coord_x[num][i] + (args.psize//2), np.newaxis]
                mask_ = mask_.astype(np.uint8)
                
                
                num2 = format(cnt, '07')
                cnt += 1
                ipath = Path(os.path.join(patch_save, name, str(level),f"{label}/img", f"{name}_{num2}.png"))
                mpath = Path(os.path.join(patch_save, name, str(level),f"{label}/mask", f"{name}_{num2}.png"))

                ipath.parent.mkdir(parents=True, exist_ok=True)
                mpath.parent.mkdir(parents=True, exist_ok=True)

                try:
                    cv2.imwrite(str(ipath), img)
                    cv2.imwrite(str(mpath), mask_)
                except:
                    print(img.shape)
                    print(mask_.shape)
                    print(label)
                    print(f"mask coord : {coord_y[num][i]}, {coord_x[num][i]}")
                    continue
        num+=1

def main():
    global class_nerve, class_pni, class_tumor, class_benign

    wsi_dir = Path(args.svs_load_dir)
    xml_dir = Path(args.xml_load_dir)
    patch_save = Path(args.save_dir)
    levels = [0, 1, 2]

    svs_paths = sorted(wsi_dir.glob("*.svs"))
    xml_paths = sorted(xml_dir.glob("*.xml"))
    assert len(svs_paths) == len(xml_paths)

    paths = list(zip(svs_paths, xml_paths))
    class_nerve = 1
    class_pni = 2
    class_tumor = 13
    class_benign = 14

    stime = time.time()
    for (slide_path, xml_path) in tqdm(paths):
        name = '_'.join(xml_path.stem.split('_')[i] for i in (0,-1))
        slide = openslide.OpenSlide(str(slide_path))
        for level in levels:
            mask = xml2mask(xml_path, slide, level)
            make_patch(patch_save, slide, mask, name, level)
    print(f"* Time : {(time.time() - stime) / 60}min")




if __name__ == '__main__':
    global args
    args = parse_args()
    random_seed(args.seed)
    main()