# -*- encoding: utf-8 -*-
import argparse
import fnmatch
import os
import pathlib

import cv2
import numpy as np
from PIL import Image


def _debug_resize(img):
    height, width = img.shape[:2]
    max_height = 800
    max_width = 800
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def files_recursively(dir_path, pattern="*.png"):
    for path, dir_name, file_names in os.walk(dir_path):
        for file_name in fnmatch.filter(file_names, pattern):
            yield os.path.join(path, file_name)


def getMinAreaRect(image):
    img = np.asarray(image)
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    cnt = cv2.findNonZero(thresh)
    return cv2.minAreaRect(cnt)


def rotate_with_bounding_rect(img, bounding_rect):
    center, dimensions, angle = bounding_rect

    box = cv2.boxPoints(bounding_rect)
    box = np.int0(box)

    W = bounding_rect[1][0]
    H = bounding_rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    mult = 1.2

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    img = cv2.warpAffine(img, M, img.size, cv2.INTER_CUBIC)
    # debug
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.circle(img, center, 10, (0, 255, 0), -1)



    return img


def generate_thumbnails(dir, images, heatmaps, out_dir=None):
    global hashed_images_map
    hashed_images_map = {image.split('/')[-1].split('.')[0]: image for image in files_recursively(images, "*.jpg")}
    hashed_heatmap_map = {image.split('/')[-1].split('_heatmap')[0]: image for image in
                          files_recursively(heatmaps, "*.png")}

    for file in files_recursively(dir):
        img = cv2.imread(file)

        if img is None:
            continue

        hash = file.split('/')[-1].split('_')[0]

        # normalize rotation according to heatmap min area rect
        heatmap = Image.open(hashed_heatmap_map[hash]).convert('L')
        heatmap_bounding_rect = getMinAreaRect(heatmap)
        # angle = getMinAreaRect(heatmap)[2]

        # load and normalize label
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        # trim label and get boundary coordinates
        _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        x, y, w, h = cv2.boundingRect(cnt)

        normalized = np.array(heatmap.convert('RGB'))
        normalized = rotate_with_bounding_rect(normalized, heatmap_bounding_rect)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', _debug_resize(normalized))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        original_image = cv2.imread(hashed_images_map[hash])

        # trim original image with label boundaries
        image = original_image[y:y + h, x:x + w]

        # save new original image
        out = "" if out_dir is None else out_dir
        out = os.getcwd() + "/" + out + "/" + file
        pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out, image)
        break


# usage:
# "trim_learning_data.py cloth_labels/categories cloth_labels/training_set/hashed_images"
def main():
    parser = argparse.ArgumentParser(description="Generate thumbnails to verify labels")
    parser.add_argument("labels", help="Label directory", type=str)
    parser.add_argument("images", help="Hashed images directory", type=str)
    parser.add_argument("heatmaps", help="Hashed heatmap directory", type=str)
    args = parser.parse_args()
    generate_thumbnails(args.labels, args.images, args.heatmaps, "out")


if __name__ == '__main__':
    main()
