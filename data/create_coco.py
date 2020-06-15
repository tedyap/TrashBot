"""
Take in COCO style annotations to create a COCO format dataset
"""

import os
import argparse
import json
import random
import logging
import math

from utils import annotation_split

def main():

    logger = logging.getLogger('logger')
    random.seed(42)
    
    parser = argument_parser()
    args = parser.parse_args()

    datapath = args.data
    filename = args.cocofile
    train_name = args.train_name
    valid_name = args.valid_name
    test_name = args.test_name
    output = args.output
    train = args.train_split
    valid = args.valid_split
    test = 1 - train - valid

    cocofile = os.path.join(datapath, filename)

    try:
        with open(cocofile, 'r') as fp:
            data = json.load(fp)
    except (FileNotFoundError, FileExistsError) as e:
        logger.error("JSON file not found")
        logger.error(e)
        exit()

    img = data["images"]
    categories = data["categories"]
    annotations = data["annotations"]
    info = data["info"]
    licenses = data["licenses"]

    images = img.copy()

    random.shuffle(img)
    num_images = len(images)
    val_idx = math.floor(train * num_images)
    test_idx = num_images - math.floor(test * num_images)

    train_ds = img[: val_idx]
    valid_ds = img[val_idx: test_idx]
    test_ds = img[test_idx:]

    train_annotations, valid_annotations, test_annotations = annotation_split(
        annotations, train_ds, valid_ds, test_ds
    )


    
def argument_parser(epilog: str = None) -> argparse.ArgumentParser:
    """
    Create an argument parser for initiating the conversion process.

    Args:
        epilog (str): epilog passed to ArgumentParser describing usage
    
    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(epilog=epilog or f"""
    Example:
        python create_coco.py --data /path/to/data/root --train_name TRAIN_NAME --valid_name VALID_NAME --test_name TEST_NAME --output /path/to/output/dir  # noqa: E501, F541
    """)

    parser.add_argument("--data", "-d", help="Path to data root directory", required=True)
    parser.add_argument("--cocofile", type=str, default="coco.json", help="Name of json file generated from the supervisely2coco script")
    parser.add_argument("--train_name", default="train", help="Name for the train dataset")
    parser.add_argument("--valid_name", default="valid", help="Name for the train dataset")
    parser.add_argument("--test_name", default="test", help="Name for the train dataset")
    parser.add_argument("--output", "-o", default="dataset", help="Output data dir", required=True)
    parser.add_argument("--train-split", type=int, default=0.6, help="Percentage split for train dataset between 0 and 1")
    parser.add_argument("--valid-split", type=int, default=0.15, help="Percentage split for valid dataset between 0 and 1")
    return parser

if __name__ == "__main__":
    main()