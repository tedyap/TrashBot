##
# Modified version of scripts written by Caio Marcellos and Sai Peri
# Refer to README for details
##

"""
Convert from Supervisely to MS-COCO format
"""

import json
from datetime import datetime
import logging
import argparse

from .utils import get_all_annotation_files, get_categories, convert_image
from .utils import NpEncoder

def convert(meta: str, base_dir: str, output: str, image_name: bool):
    """
    Convert from supervisely to COCO

    Args:
        meta (str): path to meta.json file
        base_dir (str): path to annotations base directory
        output (str): output filename
        iamge_name (bool): boolean indicating path to save
    
    Returns:
        json describing input dataset in MS-COCO format
    """

    filenames, jsons = get_all_annotation_files(base_dir)
    catmap = get_categories(meta)

    categories = [{
        "id": v,
        "name": k,
        "supercategory": "trash"
    } for k, v in catmap.items()]

    out_images = [
        convert_image(imgId, filenames[imgId], jsons[imgId],\
            catmap, base_dir, image_name) for imgId in \
            range(len(filenames))]
    
    images = [out[0] for out in out_images]
    annotations = [out[1] for out in out_images]

    annotations_flattened = [inner for lst in annotations for inner in lst]

    for idx, ann in enumerate(annotations_flattened):
        ann['id'] = idx
    
    coco = {
        "info": {
            "year": datetime.now().strftime("%Y"),
            "version": "0.1",
            "description": "Converted from supervisely to coco",
            "contributor": "caiofcm + speri203 + jsaurabh",
            "url": "",
            "date_created": datetime.now().strftime("%Y/%m/%d%H:%M:%S")
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "categories": categories,
        "images": images,
        "annotations": annotations_flattened
    }

    with open(output, 'w') as f:
        logging.debug("Saving as JSON")
        json.dump(coco, f, cls = NpEncoder)
    
def argument_parser(epilog: str = None)-> argparse.ArgumentParser:
    """
    Create an argument parser for initiating the conversion process.

    Args:
        epilog (str): epilog passed to ArgumentParser describing usage
    
    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(epilog= epilog or f"""
    Example:
        python supervisely2coco.py --meta /path/to/meta.json --annotations /path/to/annotations/folder --output /path/to/output.json
    """)

    parser.add_argument("--meta", "-m", help = 'Path to meta.json file')
    parser.add_argument("--annotations", "-a", help = "Annotations base dir")
    parser.add_argument("--output", "-o", help = "Output json filename")
    parser.add_argument("-image-name", '-n', action = 'store_true',
                        help = 'Save only filename(without absolute path')


    return parser

def main():
    logger = logging.getLogger('logger')
    parser = argument_parser()
    args = parser.parse_args()
    print(args)

    meta = args.meta
    base_dir = args.annotations
    savefile = args.output
    flag = args.image_name

    logger.info("Conversion started")
    try:
        convert(meta = meta, base_dir = base_dir, output = savefile, image_name = flag)
        logger.info('Finished converting. Check the output file for results.')
    except:
        logger.error('Could not convert. Please refer to the logs for more details')
    

if __name__ == "__main__":
    main()