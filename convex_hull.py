#!/bin/python3

import argparse
import os
import fnmatch
import warnings
from multiprocessing.dummy import Pool as ThreadPool

from skimage import io, img_as_ubyte
from skimage.morphology import convex_hull_image
from tqdm import tqdm


def convert_and_save(file_name, new_file_name):
    try:
        original = io.imread(file_name, as_grey=True)
        convex_hull = convex_hull_image(original == 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(new_file_name, img_as_ubyte(convex_hull))
    except IOError:
        pass


def generate_convex_hull(source_dir, target_dir):
    listdir = fnmatch.filter(os.listdir(source_dir), '*_label*[!lp].png')
    progress = tqdm(range(len(listdir)), unit="file")

    target_dir = (source_dir, target_dir)[target_dir != None]

    def convert(file_name):
        new_file_name = file_name.replace('.png', '_convexhull.png')
        convert_and_save(f"{source_dir}/{file_name}", f"{target_dir}/{new_file_name}")
        progress.update()

    pool = ThreadPool(4)
    results = pool.map(convert, listdir)
    pool.close()
    pool.join()

    progress.close()


def main():
    parser = argparse.ArgumentParser(description="Converts all images to convex hull images")
    parser.add_argument("dir", help="Source directory", type=str)
    parser.add_argument("target_dir", nargs='?', help="Target directory", type=str)
    args = parser.parse_args()

    print("Converting to convex hull...")
    generate_convex_hull(args.dir, args.target_dir)


if __name__ == "__main__":
    main()
