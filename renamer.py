#!/bin/python3

import argparse
import os
import hashlib


def hash_from_file(file_name):
    sha256 = hashlib.sha256()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Rename all files in directory with hash method")
    parser.add_argument("dir", help="Target directory", type=str)
    args = parser.parse_args()

    for file_name in os.listdir(args.dir):
        full_path = f"{args.dir}/{file_name}"
        extension = full_path.split(".")[-1]
        new_path = f"{args.dir}/{hash_from_file(full_path)}.{extension.lower()}"
        os.rename(full_path, new_path)


if __name__ == "__main__":
    main()
