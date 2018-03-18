#!/bin/python3

import argparse
import os

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_adaptive, threshold_local, rank
from skimage.morphology import disk
from tqdm import tqdm

from utils.image import apply_rotation_from_original


def convert_to_bw(original, threshold):
    gray = original.convert('L')
    return gray.point(lambda x: 0 if x < threshold else 255, '1')


def convert_to_bw_otsu(original, local=False):
    gray = original.convert('L')
    threshold = rank.otsu(np.array(gray), disk(15)) if local else threshold_otsu(np.array(gray))
    return Image.fromarray(np.uint8(gray >= threshold) * 255)

def convert_to_bw_adaptive(original):
    gray = original.convert('L')
    binary_adaptive = threshold_adaptive(np.array(gray), 99, offset=5)
    # return Image.fromarray(binary_adaptive)
    return Image.fromarray(np.uint8(binary_adaptive) * 255)


def convert_and_save(file_name, threshold=128, otsu=False, local_otsu=False, adaptive=False):
    try:
        original = Image.open(file_name)
        new_name = file_name.replace('jpg', 'png')
        if otsu:
            bw = convert_to_bw_otsu(original, local_otsu)
        elif adaptive:
            bw = convert_to_bw_adaptive(original)
        else:
            bw = convert_to_bw(original, threshold)
        bw = apply_rotation_from_original(original, bw)
        bw.save(new_name)
        original.close()
    except IOError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Converts all images in directory to monochrome")
    parser.add_argument("dir", help="Target directory", type=str)
    parser.add_argument("-t", "--threshold", help="(default 128)", type=int, default=128)
    parser.add_argument("--otsu", action="store_true", help="otsu threshold")
    parser.add_argument("--local", action="store_true", help="works only with --otsu (slower)")
    parser.add_argument("--adaptive", action="store_true", help="adaptive threshold")
    args = parser.parse_args()

    print("Converting to monochrome...")
    listdir = os.listdir(args.dir)
    progress = tqdm(range(len(listdir)), unit="file")
    for file_name in listdir:
        convert_and_save(f"{args.dir}/{file_name}", args.threshold, args.otsu, args.local, args.adaptive)
        progress.update()
    progress.close()


if __name__ == "__main__":
    main()
