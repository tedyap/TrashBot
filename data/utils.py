"""
Utils for converting from Supervisely to MS-COCO
"""

import os
import json
import numpy as np
import glob
from pathlib import Path
from typing import Tuple, Dict, Any

class NpEncoder(json.JSONEncoder):
    """Helper class for JSON dumping
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float):
            return float(obj)
        else:
            return super(NpEncoder, self).default(obj)
        
def get_categories(meta: str) -> dict:
    """
    Get all categories for the given dataset

    Args:
        meta (str): path to meta.json file
    
    Returns:
        dict with categories as key, order index as value
    """
    with open(meta, 'r') as f:
        meta_json = json.load(f)
    
    categories = [c['title'] for c in meta_json['classes']]
    catmap = {c:idx for idx, c in enumerate(categories)}
    return catmap

def get_all_annotation_files(base_dir: str) -> Tuple[list, list]:
    """
    Get all annotation filenames and corresponding jsons 

    Args:
        base_dir (str): path to base directory for annotations
    
    Returns:
        tuple containing filenames(sans extension) and associated annotation json files
    """
    ann_path = os.path.join(base_dir, "*.json")
    annotation_files = glob.glob(ann_path)

    image_files = [".".join(name.split(".")[:-1]) for name in annotation_files]
    jsons = []
    for files in annotation_files:
        with open(files, 'r') as f:
            annotation = json.load(f)
        jsons += [annotation]
    
    return image_files, jsons

def convert_image(id: int, name: str, jsons, category: dict, base_dir: str, image_name = False, start_idx = 0) -> Tuple[dict, list]:
    """
    Convert single image annotations to COCO representation

    Args:
        id (int): image id
        name (str): image filename
        jsons (str): json object containing annotations in supervise.ly format
        base_dir (str): path to base directory for annotations
        image_name (bool): flag indicating whether to save filenames with full path or not
        start_idx (int): annotation index
    
    Returns:
        tuple containing annotation for image info, objects
    """
    fname = name if not image_name else Path(name)
    base_coco = {
        "id": id,
        "width": jsons['size']['width'],
        "height": jsons['size']['height'],
        "file_name": fname,
        "license": 1,
        "date_captured": ""
    }

    objects = [obj for obj in jsons['objects']]
    exteriors = [np.array(obj['points']['exterior']) for obj in objects]

    bbox = [[
        exterior.min(axis = 0)[0],
        exterior.min(axis = 0)[1],
        exterior.max(axis = 0)[0],
        exterior.max(axis = 0)[1]
    ] for exterior in exteriors]

    annotations = [{
        "id": start_idx + 1,
        "image_id": id,
        "segmentation": [],
        "area": bbox[2] * bbox[3],
        "bbox": bbox,
        "category_id": category[obj['classTitle']],
        "iscrowd": 0
    } for idx, (obj, bbox) in enumerate(zip(objects, bbox))]

    return base_coco, annotations