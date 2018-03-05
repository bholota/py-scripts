import os
import time
from contextlib import suppress
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

import convex_hull
import merge_convex_hulls
from dectector_net import DetectorNet
from owncloud_downloader import Downloader
from utils.image import apply_rotation_from_original

ImageFile.LOAD_TRUNCATED_IMAGES = True

OWN_URL = "xxxxxxx"
OWN_PWD = "xxxxxx"
OWN_USER = "xxxxxx"
OWN_DIR_DATA = "cloth labels/training_set/hashed_images"
OWN_DIR_LABELS = "cloth labels/labels"

DATA_DIR = "data"
LABELS_DIR = "raw_labels"
CONVEX_DIR = "convex"
HEATMAP_DIR = "heatmap"

IMAGE_SIZE = 128


def dir_exists(path):
    if os.path.isdir(path) and len(os.listdir(path)) > 0:
        return True
    else:
        return False


def download(own_dir, local_dir):
    downloader = Downloader(OWN_URL, own_dir, local_dir, OWN_USER, OWN_PWD)

    with downloader as d:
        with suppress(FileExistsError):
            os.mkdir(local_dir)

        file_list = d.list()
        progress = tqdm(range(len(file_list)), unit="file")

        for file_info in file_list:
            d.download_file(file_info.path, file_info.name)
            progress.update()

        progress.close()


def data_vs_labels_validation():
    label_set = set()
    data_set = set()
    missing_set = set()
    valid_set = set()

    for file in os.listdir(LABELS_DIR):
        label_set.add(file.split('_')[0])
    for file in os.listdir(DATA_DIR):
        data_set.add(file.split('.')[0])

    print("+ Data vs label size same: %r" % (len(label_set) == len(data_set)))
    for data in data_set:
        if data not in label_set:
            missing_set.add(data)
        else:
            valid_set.add(data)

    if len(missing_set) == 0:
        print("+ Label for every data entry: True")
    else:
        print("* Label for every data entry: False")

    return valid_set


def generate_convex_hull():
    with suppress(FileExistsError):
        os.mkdir(CONVEX_DIR)
    convex_hull.generate_convex_hull(LABELS_DIR, CONVEX_DIR)


def generate_heatmap():
    with suppress(FileExistsError):
        os.mkdir(HEATMAP_DIR)
    merge_convex_hulls.merge_heatmap(CONVEX_DIR, HEATMAP_DIR)


def normalize_rotation(path):
    progress = tqdm(range(len(os.listdir(path))), unit="file")

    def normalize(file_name):
        full_path = f"{path}/{file_name}"
        try:
            with Image.open(full_path) as img:
                normalized = apply_rotation_from_original(img, img)
                exif = list(normalized.getdata())
                clean_image = Image.new(normalized.mode, normalized.size)
                clean_image.putdata(exif)
                clean_image.save(full_path)
        except IOError:
            pass
        progress.update()

    pool = ThreadPool(4)
    results = pool.map(normalize, os.listdir(path))
    pool.close()
    pool.join()
    progress.close()


def download_data():
    print("+ Performing data check")
    if dir_exists(DATA_DIR) is False:
        print(" * There is no data, downloading...")
        download(OWN_DIR_DATA, DATA_DIR)
        normalize_rotation(DATA_DIR)
    else:
        print("+ Data exists...")

    time.sleep(0.1)
    print("+ Performing label check")
    if dir_exists(LABELS_DIR) is False:
        print("* There is no label data, downloading...")
        download(OWN_DIR_LABELS, LABELS_DIR)
    else:
        print("+ Labels exists...")

    time.sleep(0.1)
    print("+ Performing convex hull check")
    if dir_exists(CONVEX_DIR) is False:
        print("* There is no convex data, generating...")
        generate_convex_hull()
    else:
        print("+ CONVEX exists...")

    time.sleep(0.1)
    print("+ Performing heatmap check")
    if dir_exists(HEATMAP_DIR) is False:
        print("* There is no heatmap data, generating...")
        generate_heatmap()
    else:
        print("+ Heatmap exists...")


def main():
    download_data()
    # print(len(data_vs_labels_validation()))
    data = [DATA_DIR + '/' + f + '.jpg' for f in data_vs_labels_validation()]
    labels = [HEATMAP_DIR + '/' + f for f in os.listdir(HEATMAP_DIR)]

    net = DetectorNet(data, labels)
    net.train()


if __name__ == "__main__":
    main()
