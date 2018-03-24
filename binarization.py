#!/bin/python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_local, rank
from skimage.morphology import disk
from tqdm import tqdm

from utils.image import apply_rotation_from_original

class Parameters:
    def __init__(self, type, threshold=128, block_size=99, offset=5, suffix=False):
        self.type = type
        self.threshold = threshold
        self.block_size = block_size
        self.offset = offset
        self._suffix = suffix

    @property
    def is_adaptive(self):
        return self.type == "adaptive"

    @property
    def is_otsu(self):
        return self.type == "otsu"

    @property
    def is_otsu_local(self):
        return self.type == "otsu_local"

    @property
    def is_threshold(self):
        return self.type == "threshold"


    @property
    def suffix(self):
        if self._suffix is True:
            if self.is_adaptive:
                return f"_a_{self.block_size}_{self.offset}"
            elif self.is_otsu:
                return "_o"
            elif self.is_otsu_local:
                return "_ol"
            else:
                return f"_t_{self.threshold}"
        elif self._suffix is False:
            return ""
        else:
            return self._suffix

    def __str__(self):
        return f"<Paremeters type: {self.type}, threshold: {self.threshold}, block_size: {self.block_size}, offset: {self.offset}, suffix: {self.suffix}>"

    @classmethod
    def from_args(cls, args):
        type = None
        if args.adaptive:
            type = "adaptive"
        elif args.local:
            type = "otsu_local"
        elif args.otsu:
            type = "otsu"
        else:
            type = "threshold"

        return cls(type=type, threshold=args.threshold, block_size=args.block_size, offset=args.offset)


TRY_PARAMETERS = [
    Parameters("threshold", threshold=128, suffix=True),
    Parameters("otsu", suffix=True),
    # Parameters("otsu_local", suffix=True),

    Parameters("adaptive", block_size=25, offset=5, suffix=True),
    Parameters("adaptive", block_size=25, offset=10, suffix=True),
    Parameters("adaptive", block_size=25, offset=15, suffix=True),
    Parameters("adaptive", block_size=25, offset=20, suffix=True),
    Parameters("adaptive", block_size=25, offset=25, suffix=True),
    Parameters("adaptive", block_size=25, offset=30, suffix=True),

    Parameters("adaptive", block_size=49, offset=5, suffix=True),
    Parameters("adaptive", block_size=49, offset=10, suffix=True),
    Parameters("adaptive", block_size=49, offset=15, suffix=True),
    Parameters("adaptive", block_size=49, offset=20, suffix=True),
    Parameters("adaptive", block_size=49, offset=25, suffix=True),
    Parameters("adaptive", block_size=49, offset=30, suffix=True),

    Parameters("adaptive", block_size=99, offset=5, suffix=True),
    Parameters("adaptive", block_size=99, offset=10, suffix=True),
    Parameters("adaptive", block_size=99, offset=15, suffix=True),
    Parameters("adaptive", block_size=99, offset=20, suffix=True),
    Parameters("adaptive", block_size=99, offset=25, suffix=True),
    Parameters("adaptive", block_size=99, offset=30, suffix=True),

    Parameters("adaptive", block_size=149, offset=5, suffix=True),
    Parameters("adaptive", block_size=149, offset=10, suffix=True),
    Parameters("adaptive", block_size=149, offset=15, suffix=True),
    Parameters("adaptive", block_size=149, offset=20, suffix=True),
    Parameters("adaptive", block_size=149, offset=25, suffix=True),
    Parameters("adaptive", block_size=149, offset=30, suffix=True),

    Parameters("adaptive", block_size=199, offset=5, suffix=True),
    Parameters("adaptive", block_size=199, offset=10, suffix=True),
    Parameters("adaptive", block_size=199, offset=15, suffix=True),
    Parameters("adaptive", block_size=199, offset=20, suffix=True),
    Parameters("adaptive", block_size=199, offset=25, suffix=True),
    Parameters("adaptive", block_size=199, offset=30, suffix=True),

    Parameters("adaptive", block_size=249, offset=5, suffix=True),
    Parameters("adaptive", block_size=249, offset=10, suffix=True),
    Parameters("adaptive", block_size=249, offset=15, suffix=True),
    Parameters("adaptive", block_size=249, offset=20, suffix=True),
    Parameters("adaptive", block_size=249, offset=25, suffix=True),
    Parameters("adaptive", block_size=249, offset=30, suffix=True),

    Parameters("adaptive", block_size=299, offset=5, suffix=True),
    Parameters("adaptive", block_size=299, offset=10, suffix=True),
    Parameters("adaptive", block_size=299, offset=15, suffix=True),
    Parameters("adaptive", block_size=299, offset=20, suffix=True),
    Parameters("adaptive", block_size=299, offset=25, suffix=True),
    Parameters("adaptive", block_size=299, offset=30, suffix=True),

    Parameters("adaptive", block_size=349, offset=5, suffix=True),
    Parameters("adaptive", block_size=349, offset=10, suffix=True),
    Parameters("adaptive", block_size=349, offset=15, suffix=True),
    Parameters("adaptive", block_size=349, offset=20, suffix=True),
    Parameters("adaptive", block_size=349, offset=25, suffix=True),
    Parameters("adaptive", block_size=349, offset=30, suffix=True),
]


def convert_to_bw(original, threshold):
    gray = original.convert('L')
    return gray.point(lambda x: 0 if x < threshold else 255, '1')


def convert_to_bw_otsu(original, local=False):
    gray = original.convert('L')
    threshold = rank.otsu(np.array(gray), disk(15)) if local else threshold_otsu(np.array(gray))
    return Image.fromarray(np.uint8(gray >= threshold) * 255)

def convert_to_bw_adaptive(original, block_size, offset):
    gray = original.convert('L')
    image = np.array(gray)
    binary_adaptive = image > threshold_local(image, block_size, offset=offset)
    return Image.fromarray(np.uint8(binary_adaptive) * 255)


def convert_and_save(file_path, parameter):
    try:
        name = "".join([file_path.stem, parameter.suffix])
        new_file_path = file_path.with_name(f"{name}.png")

        file_name = str(file_path)
        original = Image.open(file_name)
        if parameter.is_otsu or parameter.is_otsu_local:
            bw = convert_to_bw_otsu(original, parameter.is_otsu_local)
        elif parameter.is_adaptive:
            bw = convert_to_bw_adaptive(original, parameter.block_size, parameter.offset)
        else:
            bw = convert_to_bw(original, parameter.threshold)
        bw = apply_rotation_from_original(original, bw)
        bw.save(str(new_file_path))
        original.close()
    except IOError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Converts all images in directory to monochrome")
    parser.add_argument("path", help="Target directory or file", type=str)
    parser.add_argument("-t", "--threshold", help="(default 128)", type=int, default=128)
    parser.add_argument("-b", "--block_size", help="(default 99)", type=int, default=99)
    parser.add_argument("-o", "--offset", help="(default 5)", type=int, default=5)
    parser.add_argument("--otsu", action="store_true", help="otsu threshold")
    parser.add_argument("--local", action="store_true", help="works only with --otsu (slower)")
    parser.add_argument("--adaptive", action="store_true", help="adaptive threshold")
    parser.add_argument("--try", action="store_true", help="try different parameters")
    args = parser.parse_args()

    parameters = []
    if vars(args)["try"] is True:
        parameters = TRY_PARAMETERS
    else:
        parameters = [Parameters.from_args(args)]

    print("Converting to monochrome...")
    p = Path(args.path)
    if p.is_dir():
        listdir = list(p.iterdir())
        progress = tqdm(range(len(listdir) * len(parameters)), unit="file")
        for file_path in listdir:
            for parameter in parameters:
                convert_and_save(file_path, parameter)
                progress.update()
        progress.close()
    else:
        progress = tqdm(range(len(parameters)), unit="file")
        for parameter in parameters:
            convert_and_save(p, parameter)
            progress.update()
        progress.close()


if __name__ == "__main__":
    main()
