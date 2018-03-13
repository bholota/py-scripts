#!/bin/python3

import argparse
import imagehash
import os
import fnmatch
import glob
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image
from tqdm import tqdm

from utils.image import apply_rotation_from_original

def rename(path, labels_path = None):
    progress = tqdm(range(len(os.listdir(path))), unit="file")

    def convert(file_name):
        full_path = f"{path}/{file_name}"
        extension = full_path.split(".")[-1]
        file_name_without_extension = os.path.basename(file_name).split(".")[0]

        # normalization and exif cleanup
        try:
            with Image.open(full_path) as img:
                normalized = apply_rotation_from_original(img, img)

                if img != normalized:
                    print (f"{full_path} was normalized")
                    exif = list(normalized.getdata())
                    clean_image = Image.new(normalized.mode, normalized.size)
                    clean_image.putdata(exif)
                    clean_image.save(full_path)

                perceptual_hash = imagehash.phash(normalized)
                new_path = f"{path}/{perceptual_hash}.{extension.lower()}"

                if new_path != full_path:
                    print(f"renaming {full_path} to {new_path}")
                    os.rename(full_path, new_path)

                    if labels_path != None:
                        for matching_filename in glob.iglob(f"{labels_path}/**/*{file_name_without_extension}*", recursive=True):
                            renamed_file_name = matching_filename.replace(file_name_without_extension, f"{perceptual_hash}")

                            if renamed_file_name != matching_filename:
                                print(f"renaming match {matching_filename} to {renamed_file_name}")
                                os.rename(matching_filename, renamed_file_name)

        except IOError:
            pass
        progress.update()

    pool = ThreadPool(4)
    results = pool.map(convert, fnmatch.filter(os.listdir(path),'*.*'))
    pool.close()
    pool.join()

    progress.close()


def main():
    parser = argparse.ArgumentParser(description="Rename all files in directory with hash method")
    parser.add_argument("dir", help="Target directory", type=str)
    parser.add_argument("labels_dir", nargs='?', help="Labels directory", type=str)
    args = parser.parse_args()
    rename(args.dir, args.labels_dir)


if __name__ == "__main__":
    main()
