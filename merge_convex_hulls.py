#!/bin/python3

import argparse
import os, fnmatch

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

        io.imsave(heatmap_file_name, img_as_ubyte(merged_image))
    except IOError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Merge convex hull images into heatmap")
    parser.add_argument("dir", help="Source directory", type=str)
    parser.add_argument("target_dir", nargs='?', help="Target directory", type=str)

    args = parser.parse_args()

    print("Merging convex hulls into heatmaps...")
    listdir = fnmatch.filter(os.listdir(args.dir),'*_convexhull.png')
    progress = tqdm(range(len(listdir)), unit="file")

    target_dir = (args.dir, args.target_dir)[args.target_dir != None]

    for file_name in listdir:
        heatmap_file_name = file_name.split('_', 1)[0] + '_heatmap.png'
        merge_and_save(f"{args.dir}/{file_name}", f"{target_dir}/{heatmap_file_name}")
        progress.update()
    progress.close()

if __name__ == "__main__":
    main()
