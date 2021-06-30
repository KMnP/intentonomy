#!/usr/bin/env python3
"""
Downloads images from unsplash using multiple threads.
Images that already exist will not be downloaded again, so the script can
resume a partially completed download. All images will be saved in the JPG
format with 90% compression quality.
"""
import argparse
import glob
import json
import multiprocessing
import os
import requests

from contextlib import contextmanager
from io import BytesIO
from typing import Union
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "rb") as fin:
        data = json.load(fin, encoding="utf-8")
    return data


def download_image(url: str, file_name: str, image_root: str) -> None:
    image_path = os.path.join(image_root, file_name)

    if os.path.exists(image_path):
        print(f"Image {image_path} already exists. Skipping download.")
        return

    try:
        response = requests.get(url)
        image_data = response.content
    except Exception as e:
        print(f"Warning: Could not download image {file_name} from {url} with message {e}")  # noqa
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"Warning: Failed to parse image {url} with message {e}")  # noqa
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to convert image {url} to RGB with message {e}")  # noqa
        return

    try:
        pil_image_rgb.save(image_path, format='JPEG', quality=90)
    except Exception as e:
        print(f"Warning: Failed to save image {url} to {image_path} with message {e}")  # noqa
        return


def download_image_unpack(args):
    return download_image(*args)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def main(anno_root: str, image_root: str) -> None:
    if not os.path.exists(image_root):
        os.makedirs(image_root)

    ann_paths = glob.glob(anno_root + "/*.json")
    for ann_path in ann_paths:
        print("=" * 80)
        print(f"Start downloading images from {ann_path}")
        print("=" * 80)
        # read the json file
        ann = read_json(ann_path)

        arg_list = [(
            image_ann["unsplash_url"],
            image_ann["filename"],
            image_root
        ) for image_ann in ann["images"]]

        # make sure the image dir exists
        img_dir, _ = os.path.split(os.path.join(image_root, arg_list[0][1]))
        if img_dir and not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # spawn process
        num_processes = multiprocessing.cpu_count() // 2
        print(f"\tgoing to spawn {num_processes} processes")
        with poolcontext(processes=num_processes) as pool:
            for _ in tqdm(pool.imap_unordered(download_image_unpack, iter(arg_list)), total=len(arg_list)):  # noqa
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--anno-root',
        default="/Users/menglin/Workplace/2020intent/public_release/annotations")
    parser.add_argument(
        '--image-root',
        default="/Users/menglin/Workplace/2020intent/public_release")
    args = parser.parse_args()

    main(args.anno_root, args.image_root)
