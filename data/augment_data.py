"""
"""

import os
import argparse
import json
import numpy as np
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from augmentation.augment import Pipeline
from augmentation.horizontal import RandomHorizontalFlip
from augmentation.scale import RandomScale, Scale
from augmentation.translate import RandomTranslate, Translate
from augmentation.rotate import RandomRotate
from augmentation.utils import draw_rectangle

def main():
    parser = argument_parser()
    args = parser.parse_args()

    jsonfile = args.json
    datapath = args.data
    # outputdir = args.output

    with open(jsonfile) as fp:
        data = json.load(fp)
    
    cat2idx, idx2cat = {}, {}
    idx2image, image2idx = {}, {}
    image2annotations = defaultdict(list)
    filenames, paths = [], []

    images = data['images']
    categories = data['categories']
    annotations = data['annotations']

    for image in images:
        idx2image[image['id']] = image['file_name']
        image2idx[image['file_name']] = image['id']
        filename = image['file_name'].split('/')[-1]
        filenames.append(filename)

    for category in categories:
        idx2cat[category['id']] = category['name']
        cat2idx[category['name']] = category['id']

    for annotation in annotations:
        image_id = annotation['image_id']
        bbox2cat = annotation['bbox'] + [annotation['category_id']]
        bbox = [float(b) for b in bbox2cat]
        image2annotations[image_id].append(bbox)
        
    image_folder = datapath + '/img/'
    for filename in filenames:
        path = os.path.join(image_folder, filename)
        paths.append(path)

    for idx in range(3):
        augment_image(paths[idx], np.array(image2annotations[idx]))

def augment_image(path, bbox):
    image = cv.imread(path)[:, :, ::-1]
    # bbox = np.array([[402, 0, 799, 270, 1]]).astype('float64')
    print(bbox, path)
    plt.imshow(draw_rectangle(image, bbox))
    plt.show()

    transforms = Pipeline([RandomHorizontalFlip(0.7), Scale(0.2, 0.2), RandomRotate((2, 2))])
    image, bbox = transforms(image, bbox)
    
    print(bbox, path)
    plt.imshow(draw_rectangle(image, bbox))
    plt.show()

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
    parser.add_argument("-image-name", '-n', action="store_true",
                        help="Save only filename(without absolute path")
    return parser


if __name__ == "__main__":
    main()