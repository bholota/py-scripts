#!/bin/python3

import argparse
from contextlib import suppress

import owncloud
import os
from tqdm import tqdm


class Downloader:

    def __init__(self, own_url, own_dir, out_dir, own_user=None, own_pass=None):
        self.own_url = own_url
        self.own_dir = own_dir
        self.out_dir = out_dir
        self.own_user = own_user
        self.own_pass = own_pass
        self.oc = None

    def connect(self):
        self.oc = owncloud.Client(self.own_url)
        self.oc.login(self.own_user, self.own_pass)

    def disconnect(self):
        if self.oc is not None:
            self.oc.logout()

    def list(self):
        if self.oc is not None:
            return self.oc.list(self.own_dir)
        else:
            raise ConnectionError()

    def download_zip(self):
        self.oc.get_directory_as_zip(self.own_dir, self.out_dir)

    def download_file(self, remote_path, file_name):
        self.oc.get_file(remote_path, f"{self.out_dir}/{file_name}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Download and parse owncloud images")
    parser.add_argument("ownurl", help="OwnCloud url", type=str)
    parser.add_argument("owndir", help="OwnCloud dir path", type=str)
    parser.add_argument("output", help="output directory", type=str)
    parser.add_argument("-u", "--user", help="OwnCloud user", type=str)
    parser.add_argument("-p", "--passwd", help="OwnCloud user password", type=str)
    args = parser.parse_args()

    downloader = Downloader(args.ownurl, args.owndir, args.output, args.user, args.passwd)

    with downloader as d:
        with suppress(FileExistsError):
            os.mkdir(args.output)

        file_list = d.list()
        progress = tqdm(range(len(file_list)), unit="file")

        for file_info in file_list:
            d.download_file(file_info.path, file_info.name)
            progress.update()

        progress.close()


if __name__ == "__main__":
    main()
