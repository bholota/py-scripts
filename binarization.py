#!/bin/python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_local, rank
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

def convert_to_bw_adaptive(original, block_size, offset):
    gray = original.convert('L')
    image = np.array(gray)
    binary_adaptive = image > threshold_local(image, block_size, offset=offset)
    return Image.fromarray(np.uint8(binary_adaptive) * 255)


def convert_and_save(file_path, threshold=128, block_size=99, offset=5, otsu=False, local_otsu=False, adaptive=False):
    try:
        file_name = str(file_path)
        original = Image.open(file_name)
        new_name = file_name.replace('jpg', 'png')
        if otsu:
            bw = convert_to_bw_otsu(original, local_otsu)
        elif adaptive:
            bw = convert_to_bw_adaptive(original, block_size, offset)
        else:
            bw = convert_to_bw(original, threshold)
        bw = apply_rotation_from_original(original, bw)
        bw.save(new_name)
        original.close()
    except IOError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Converts all images in directory to monochrome")
    parser.add_argument("path", help="Target directory or file", type=str)
    parser.add_argument("-t", "--threshold", help="(default 128)", type=int, default=128)
    parser.add_argument("-b", "--block_size", help="(default 99)", type=int, default=99)
    parser.add_argument("-o", "--offset", help="(default 5)", type=int, default=5)
    parser.add_argument("--otsu", action="store_true", help="otsu threshold")
    parser.add_argument("--local", action="store_true", help="works only with --otsu (slower)")
    parser.add_argument("--adaptive", action="store_true", help="adaptive threshold")
    args = parser.parse_args()

    print("Converting to monochrome...")
    p = Path(args.path)
    if p.is_dir():
        listdir = list(p.iterdir())
        progress = tqdm(range(len(listdir)), unit="file")
        for file_path in listdir:
            convert_and_save(file_path, threshold=args.threshold, block_size=args.block_size, offset=args.offset, otsu=args.otsu, local_otsu=args.local, adaptive=args.adaptive)
            progress.update()
        progress.close()
    else:
        progress = tqdm(range(1), unit="file")
        convert_and_save(p, threshold=args.threshold, block_size=args.block_size, offset=args.offset, otsu=args.otsu, local_otsu=args.local, adaptive=args.adaptive)
        progress.update()
        progress.close()


if __name__ == "__main__":
    main()
