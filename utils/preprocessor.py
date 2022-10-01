import os

import cv2
import numpy as np
from pathlib import Path
import scipy.ndimage

import openslide
import xml.etree.ElementTree as ET

CLASS_NERVE = 1
CLASS_PNI = 2
CLASS_TUMOR = 13
CLASS_BENIGN = 14


class Preprocessor:
    def __init__(self, args):
        self.args = args

    def xml2mask(self, xml_fn, slide, level):
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
        src_w, src_h = slide.level_dimensions[0]
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
                    mask = cv2.polylines(
                        mask, pts, isClosed=False, color=label, thickness=1
                    )
                # Contour
                else:
                    mask = cv2.drawContours(mask, pts, -1, label, -1)
        return mask

    def shuffle_coord(self, mask, label_class):
        idx = np.arange(len(np.where(mask == label_class)[0]))
        np.random.shuffle(idx)
        y = np.where(mask == label_class)[0][idx]
        x = np.where(mask == label_class)[1][idx]

        return idx, y, x

    def make_patch(self, slide, mask, name, level):
        max_x, max_y = slide.level_dimensions[level]

        idx_nerve, y_nerve, x_nerve = self.shuffle_coord(mask, CLASS_NERVE)
        idx_pni, y_pni, x_pni = self.shuffle_coord(mask, CLASS_PNI)
        idx_tumor, y_tumor, x_tumor = self.shuffle_coord(mask, CLASS_TUMOR)
        idx_benign, y_benign, x_benign = self.shuffle_coord(mask, CLASS_BENIGN)

        num = 0
        idx_list = [idx_nerve, idx_pni, idx_tumor, idx_benign]
        coord_x = [x_nerve, x_pni, x_tumor, x_benign]
        coord_y = [y_nerve, y_pni, y_tumor, y_benign]

        for idx in idx_list:
            if num == 0:
                label = "class0"
            elif num == 1:
                label = "class1"
            elif num == 2:
                label = "class2"
            elif num == 3:
                label = "class3"

            cnt = 0
            range_max = self.args.max_patches if len(idx) > self.args.max_patches else len(idx)
            for i in range(range_max):
                x = coord_x[num][i] * (4 ** (level)) - (self.args.psize // 2) * (
                    4 ** (level)
                )
                y = coord_y[num][i] * (4 ** (level)) - (self.args.psize // 2) * (
                    4 ** (level)
                )
                img = np.array(
                    slide.read_region((x, y), level, (self.args.psize, self.args.psize))
                )
                img = img[:, :, :3]
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                mask_ = mask[
                    coord_y[num][i]
                    - (self.args.psize // 2) : coord_y[num][i]
                    + (self.args.psize // 2),
                    coord_x[num][i]
                    - (self.args.psize // 2) : coord_x[num][i]
                    + (self.args.psize // 2),
                    np.newaxis,
                ]
                mask_ = mask_.astype(np.uint8)

                num_form = format(cnt, "07")
                cnt += 1
                ipath = Path(
                    os.path.join(
                        self.patch_save,
                        name,
                        self.args.mode,
                        "level_" + str(level),
                        f"{label}/img",
                        f"{name}_{num_form}.png",
                    )
                )
                mpath = Path(
                    os.path.join(
                        self.patch_save,
                        name,
                        self.args.mode,
                        "level_" + str(level),
                        f"{label}/mask",
                        f"{name}_{num_form}.png",
                    )
                )

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
            num += 1

    # def sliding_window_extract(patch_save, paths, label, level):
    def make_patch_sw(self, slide, mask, name, level, target=1, stride=25, label="pni"):
        py = self.args.psize // 2
        px = self.args.psize // 2

        m = np.where(mask == target, 1, 0)
        if len(m) == 0:
            return
        m = m.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(m)

        c_list = []
        for i in range(1, num_labels):
            bblist = np.where(labels == i)
            bblist_ = list(zip(bblist[0], bblist[1]))
            topL = np.array(bblist_).min(axis=0)
            botR = np.array(bblist_).max(axis=0)
            start_x = topL[1]
            start_y = topL[0]
            end_x = botR[1]
            end_y = botR[0]
            output = [
                (x, y)
                for x in range(start_x, end_x + 1, stride)
                for y in range(start_y, end_y + 1, stride)
            ]
            c_list.extend(output)

        final_list = c_list

        for idx, (x, y) in enumerate(final_list):
            mask_ = mask[y - (py // 2) : y + (py // 2), x - (px // 2) : x + (px // 2)]
            if target in np.unique(mask_):
                ## img
                img = np.array(
                    slide.read_region(
                        ((x - px) * (4 ** (level)), (y - py) * (4 ** (level))),
                        level,
                        (self.args.psize, self.args.psize),
                    )
                )
                img = img[:, :, :3]
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                ## mask
                mask_ = mask[y - py : y + py, x - px : x + px, np.newaxis]
                mask_ = np.where(mask_ == target, 1, 0)
                mask_ = scipy.ndimage.morphology.binary_dilation(mask_)
                mask_ = mask_.astype(np.uint8)

                num = format(idx, "07")
                idx += 1
                ipath = Path(
                    os.path.join(
                        self.patch_save,
                        name,
                        self.args.mode,
                        "level_" + str(level),
                        label,
                        "img_sw",
                        f"{name}_{num}.png",
                    )
                )
                mpath = Path(
                    os.path.join(
                        self.patch_save,
                        name,
                        self.args.mode,
                        "level_" + str(level),
                        label,
                        "mask_sw",
                        f"{name}_{num}.png",
                    )
                )

                ipath.parent.mkdir(parents=True, exist_ok=True)
                mpath.parent.mkdir(parents=True, exist_ok=True)

                try:
                    cv2.imwrite(str(ipath), img)
                    cv2.imwrite(str(mpath), mask_)
                except:
                    print(img.shape)
                    print(mask_.shape)
                    continue
