import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import openslide

from utils.preprocessor import Preprocessor


def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Make Patches PAIP 2021 Patch Dataset")
    parser.add_argument(
        "--svs_load_dir", type=str, help="Challenge Whole Slide Images dir"
    )
    parser.add_argument("--xml_load_dir", type=str, help="Challenge Annotation dir")
    parser.add_argument("--save_dir", type=str, help="Patch Save dir")
    parser.add_argument("--psize", type=int, default=224, help="Patch Size")
    parser.add_argument(
        "--max_patches", type=int, default=2000, help="maximum patches for each class"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, help="random, sw: sliding window")
    return parser.parse_args()


def main(args):
    wsi_dir = Path(args.svs_load_dir)
    xml_dir = Path(args.xml_load_dir)
    levels = [0, 1, 2]

    svs_paths = sorted(wsi_dir.glob("*.svs"))
    xml_paths = sorted(xml_dir.glob("*.xml"))
    assert len(svs_paths) == len(xml_paths)

    preprocessor = Preprocessor(args)
    paths = list(zip(svs_paths, xml_paths))
    stime = time.time()

    for (slide_path, xml_path) in tqdm(paths):
        name = "_".join(xml_path.stem.split("_")[i] for i in (0, -1))
        slide = openslide.OpenSlide(str(slide_path))
        for level in levels:
            mask = preprocessor.xml2mask(xml_path, slide, level)
            if args.mode == "random":
                preprocessor.make_patch(slide, mask, name, level)
            else:
                preprocessor.make_patch_sw(slide, mask, name, level)
    print(f"* Time : {(time.time() - stime) / 60}min")


if __name__ == "__main__":
    args = parse_args()
    random_seed(args.seed)
    main(args)
