#!/bin/python3

import argparse
import os

from skimage import io, img_as_ubyte
from skimage.morphology import convex_hull_image
from tqdm import tqdm

def convert_and_save(file_name, new_file_name):
    try:
        original = io.imread(file_name, as_grey=True)
        convex_hull = convex_hull_image(original == 0)
        io.imsave(new_file_name, img_as_ubyte(convex_hull))
    except IOError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Converts all images to convex hull images")
    parser.add_argument("dir", help="Source directory", type=str)
    parser.add_argument("target_dir", nargs='?', help="Target directory", type=str)

    args = parser.parse_args()

    print("Converting to convex hull...")
    listdir = os.listdir(args.dir)
    progress = tqdm(range(len(listdir)), unit="file")

    target_dir = (args.dir, args.target_dir)[args.target_dir != None]

    for file_name in listdir:
        new_file_name = file_name.replace('.png', '_convexhull.png')
        convert_and_save(f"{args.dir}/{file_name}", f"{target_dir}/{new_file_name}")
        progress.update()
    progress.close()

if __name__ == "__main__":
    main()
