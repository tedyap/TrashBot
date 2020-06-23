"""
Take in COCO style annotations to create a COCO format dataset
"""

import os
import argparse
import json
import logging
import shutil

from utils import annotation_split, dataset_split, create_json

def makedirs(datapath: str, dirname: str) -> None:
    """
    Create directory structure if not exists.

    Args:
        datapath (str): Absolute path of current directory
        dirname (str): Name of new directory to create
    
    Returns:
        None
    """
    dirname = os.path.join(datapath, dirname)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def main():
    """
    Parse command line inputs and generate COCO style dataset
    with respective json annotation files.
    """
    logger = logging.getLogger('logger')
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
        logger.error("JSON file not found. Ensure paths are entered properly")
        logger.error(e)
        exit()

    img = data["images"]
    annotations = data["annotations"]

    images = img.copy()
    train_ds, valid_ds, test_ds = dataset_split(images, train, valid, test)

    train_annotations, valid_annotations, test_annotations = annotation_split(
        annotations, train_ds, valid_ds, test_ds
    )

    makedirs(datapath, train_name)
    makedirs(datapath, valid_name)
    makedirs(datapath, test_name)

    train_images = [image['file_name'] for image in train_ds]
    valid_images = [image['file_name'] for image in valid_ds]
    test_images = [image['file_name'] for image in test_ds]

    for image in train_images:
        shutil.move(os.path.join(datapath, image), os.path.join(datapath, train_name, image))
    for image in valid_images:
        shutil.move(os.path.join(datapath, image), os.path.join(datapath, valid_name, image))
    for image in test_images:
        shutil.move(os.path.join(datapath, image), os.path.join(datapath, test_name, image))

    create_json(data, train_ds, train_annotations, output, 'instances_train.json')
    create_json(data, valid_ds, valid_annotations, output, 'instances_valid.json')
    create_json(data, test_ds, test_annotations, output, 'instances_test.json')

    logger.info("COCO Dataset created")

def argument_parser(epilog: str = None) -> argparse.ArgumentParser:
    """
    Create an argument parser for converting data COCO style split datasets.

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
    parser.add_argument("--cocofile", type=str, default="coco.json",
                        help="Name of json file generated from the supervisely2coco script")
    parser.add_argument("--train_name", default="train", help="Name for the train dataset")
    parser.add_argument("--valid_name", default="valid", help="Name for the train dataset")
    parser.add_argument("--test_name", default="test", help="Name for the train dataset")
    parser.add_argument("--output", "-o", default="dataset", help="Output data dir", required=True)
    parser.add_argument("--train-split", type=int, default=0.6,
                        help="Percentage split for train dataset between 0 and 1")
    parser.add_argument("--valid-split", type=int, default=0.15,
                        help="Percentage split for valid dataset between 0 and 1")
    return parser


if __name__ == "__main__":
    main()