#!/bin/python3

import argparse
import fnmatch
import os
import warnings
from multiprocessing.dummy import Pool as ThreadPool

from skimage import io, img_as_ubyte
from tqdm import tqdm


def merge_and_save(convex_hull_file_name, heatmap_file_name):
    try:
        convex_hull_image = io.imread(convex_hull_file_name, as_grey=True)
        merged_image = convex_hull_image != 0
        try:
            heatmap_image = io.imread(heatmap_file_name, as_grey=True)
            merged_image = merged_image | (heatmap_image != 0)
        except IOError:
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(heatmap_file_name, img_as_ubyte(merged_image))
    except IOError:
        pass


def merge_heatmap(source_path, target_path):
    listdir = fnmatch.filter(os.listdir(source_path), '*_convexhull.png')
    progress = tqdm(range(len(listdir)), unit="file")

    target_dir = (source_path, target_path)[target_path != None]

    def convert(file_name):
        heatmap_file_name = file_name.split('_', 1)[0] + '_heatmap.png'
        merge_and_save(f"{source_path}/{file_name}", f"{target_dir}/{heatmap_file_name}")
        progress.update()

    pool = ThreadPool(4)
    results = pool.map(convert, listdir)
    pool.close()
    pool.join()

    progress.close()


def main():
    parser = argparse.ArgumentParser(description="Merge convex hull images into heatmap")
    parser.add_argument("dir", help="Source directory", type=str)
    parser.add_argument("target_dir", nargs='?', help="Target directory", type=str)

    args = parser.parse_args()

    print("Merging convex hulls into heatmaps...")
    merge_heatmap(args.dir, args.target_dir)


if __name__ == "__main__":
    main()
