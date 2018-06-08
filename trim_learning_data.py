# -*- encoding: utf-8 -*-
import argparse
import fnmatch
import os
import pathlib

import cv2



def files_recursively(dir_path, pattern="*.png"):
    for path, dir_name, file_names in os.walk(dir_path):
        for file_name in fnmatch.filter(file_names, pattern):
            yield os.path.join(path, file_name)


def trim_orginal_image(hash, x, y, w, h):
    return cv2.imread(hashed_images_map[hash])[y:y + h, x:x + w]


def generate_thumbnails(dir, images, out_dir=None):
    global hashed_images_map
    hashed_images_map = {image.split('/')[-1].split('.')[0]: image for image in files_recursively(images, "*.jpg")}

    for file in files_recursively(dir):
        img = cv2.imread(file)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        x, y, w, h = cv2.boundingRect(cnt)

        image = trim_orginal_image(file.split('/')[-1].split('_')[0], x, y, w, h)
        out = "" if out_dir is None else out_dir
        out = os.getcwd() + "/" + out + "/" + file
        pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out, image)

# usage:
# "trim_learning_data.py cloth_labels/categories cloth_labels/training_set/hashed_images"
def main():
    parser = argparse.ArgumentParser(description="Generate thumbnails to verify labels")
    parser.add_argument("labels", help="Label directory", type=str)
    parser.add_argument("images", help="Hashed images directory", type=str)
    args = parser.parse_args()
    generate_thumbnails(args.labels, args.images, "out")


if __name__ == '__main__':
    main()
