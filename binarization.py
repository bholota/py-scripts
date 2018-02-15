#!/bin/python3

import argparse
import os

from PIL import Image
from tqdm import tqdm


def convert_and_save(file_name, threshold=128):
    try:
        original = Image.open(file_name)
        gray = original.convert('L')
        bw = gray.point(lambda x: 0 if x < threshold else 255, '1')
        new_name = file_name.replace('jpg', 'png')
        if hasattr(original, '_getexif'):
            orientation = 0x0112
            exif = original._getexif()
            if exif is not None:
                orientation = exif[orientation]
                rotations = {
                    3: Image.ROTATE_180,
                    6: Image.ROTATE_270,
                    8: Image.ROTATE_90
                }
                if orientation in rotations:
                    bw = bw.transpose(rotations[orientation])

        bw.save(new_name)
        original.close()
    except IOError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Converts all images in directory to monochrome")
    parser.add_argument("dir", help="Target directory", type=str)
    parser.add_argument("-t", "--threshold", help="(default 128)", type=int)
    args = parser.parse_args()

    print("Converting to monochrome...")
    listdir = os.listdir(args.dir)
    progress = tqdm(range(len(listdir)), unit="file")
    for file_name in listdir:
        convert_and_save(f"{args.dir}/{file_name}", args.threshold)
        progress.update()
    progress.close()


if __name__ == "__main__":
    main()
