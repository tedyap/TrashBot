"""
Utils for converting from Supervisely to MS-COCO
"""

import json
import numpy as np

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
        
def get_categories(meta):
    raise NotImplementedError

def get_all_annotation_files(base_dir):
    raise NotImplementedError

def convert_image(id, name, json, category, base_dir, image_name = False, start_idx = 0):
    raise NotImplementedError