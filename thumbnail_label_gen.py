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


def generate_thumbnails(dir, out_dir=None):
    for file in files_recursively(dir):
        #print("File: " + file)
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
        dst = img[y:y+h, x:x+w]

        out = "" if out_dir is None else out_dir
        out = os.getcwd() + "/" + out + "/" + file
        print("Out: " + out)
        pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out, dst)

        #cv2.imshow('image', dst)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Generate thumbnails to verify labels")
    parser.add_argument("dir", help="Target directory", type=str)
    args = parser.parse_args()
    generate_thumbnails(args.dir, "out")


if __name__ == '__main__':
    main()
