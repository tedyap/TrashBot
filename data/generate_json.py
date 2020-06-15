"""
"""

import json

def get_json(current_idx, image, annotation_idx, annotations, output):
    """
    """

    with open(output, 'r') as fp:
        data = json.load(fp)

    base_image = {
        "id": current_idx + 1,
        "width": 800,
        "height": 450,
        "file_name": image,
        "license": 1,
        "date_captured": ""
    }
    data['images'].append(base_image)

    count = len(annotations[0])
    for cnt in range(count):
        bbox = annotations[0][cnt][:-1]
        category = int(annotations[0][cnt][-1])
        bbox = [int(box) for box in bbox]
        
        annotation = {
            "id": annotation_idx + 1,
            "image_id": current_idx + 1,
            "segmentation": [],
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "category_id": category,
            "iscrowd": 0
        }
        annotation_idx += 1
        data['annotations'].append(annotation)

    with open(output, 'w') as fp:
        json.dump(data, fp)
        
    return annotation_idx