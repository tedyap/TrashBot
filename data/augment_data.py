"""
"""

import os
import shutil
import argparse
import logging
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from collections import defaultdict

from generate_json import get_json
from augmentation.augment import Pipeline
from augmentation.horizontal import RandomHorizontalFlip
from augmentation.scale import RandomScale, Scale
from augmentation.translate import RandomTranslate, Translate
from augmentation.rotate import RandomRotate
from augmentation.utils import draw_rectangle

def main():
    logger = logging.getLogger('logger')
    parser = argument_parser()
    args = parser.parse_args()

    jsonfile = args.json
    datapath = args.data
    outputdir = args.output

    with open(jsonfile) as fp:
        data = json.load(fp)
    
    image2annotations = defaultdict(list)
    filenames, paths = [], []
    augmented_images, augmented_boxes = [], defaultdict(list)

    images = data['images']
    annotations = data['annotations']

    for image in images:
        filename = image['file_name'].split('/')[-1]
        filenames.append(filename)

    last_image_idx = image['id']

    for annotation in annotations:
        bbox2cat = annotation['bbox'] + [annotation['category_id']]
        bbox = [float(b) for b in bbox2cat]
        image2annotations[annotation['image_id']].append(bbox)
    
    last_annotation_idx = annotation['id']

    image_folder = datapath + '/img/'
    for filename in filenames:
        path = os.path.join(image_folder, filename)
        paths.append(path)

    image_output = os.path.join(outputdir, 'img')

    for idx in range(len(paths)):
        image, bbox = augment_image(paths[idx], np.array(image2annotations[idx]), viz=True)
        image = image[:, :, ::-1]
        name = 'aug_' + filenames[idx]
        augmented_image = os.path.join(image_output, name)
        try:
            cv.imwrite(augmented_image, image)
            augmented_images.append(augmented_image)
            augmented_boxes[idx].append(bbox)
        except Exception as e:
            logger.info(e)
            continue

    for idx in range(len(paths)):
        last_annotation_idx = get_json(last_image_idx, augmented_images[idx], last_annotation_idx, augmented_boxes[idx], jsonfile)  # noqa:#E501
        last_image_idx += 1
    
def augment_image(path, bbox, transforms, random: bool = True, viz: bool = False):
    """
    """

    image = cv.imread(path)[:, :, ::-1]
    if viz:
        plt.imshow(draw_rectangle(image, bbox))
        plt.show()
    
    transforms = Pipeline([RandomHorizontalFlip(1), Scale(0.2, 0.2), RandomRotate((2, 2))])
    new_image, new_bbox = transforms(image, bbox)
    
    if viz:
        plt.imshow(draw_rectangle(new_image, new_bbox))
        plt.show()
    
    return new_image, new_bbox

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
        python augment_data.py --json /path/to/category.json --data /path/to/supervisely/categoty/root --output /path/to/output # noqa: E501, F541
    """)

    parser.add_argument("--json", "-m", help="Path to category json file")
    parser.add_argument("--data", "-a", help="Data base dir")
    parser.add_argument("--output", "-o", help="Output directory")
    return parser


if __name__ == "__main__":
    main()