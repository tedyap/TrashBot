"""
"""

import os
import cv2 as cv
import numpy as np

import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class Coco(Dataset):
    """
    """
    def __init__(self, root, data='train2017', transforms=None):
        self.root_dir = root
        self.set = data
        self.transforms = transforms
        self.path = os.path.join(self.root_dir, 'annotations', 'instances_' + self.set + '.json')
        self.num_classes = 80
        self.classes = {}
        self.labels = {}
        self.labels_inv = {}
        self.labels_rev = {}

        self.coco = COCO(self.path)
        self.imageIds = self.coco.getImgIds()

        self.load_classes()
    
    def __getitem__(self, idx):
        """
        """

        image = self.load_image(idx)
        annotation = self.load_annotations(idx)
        out = {
            'img': image,
            'annot': annotation
        }
        if self.transforms:
            out = self.transforms(out)
        
        return out

    def __len__(self):
        return len(self.imageIds)

    def load_classes(self):
        """
        """

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        for c in categories:
            self.labels[len(self.classes)] = c['id']
            self.labels_inv[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        for k, v in self.classes.items():
            self.labels_rev[v] = k

    def get_num_classes(self):
        """
        """
        return self.num_classes
    
    def label2coco(self, label):
        """
        """
        return self.labels[label]

    def coco2label(self, coco_label):
        """
        """
        return self.labels_inv[coco_label]

    def load_annotations(self, idx):
        """
        """
        annotationIds = self.coco.getAnnIds(imgIds=self.imageIds[idx], iscrowd=False)
        default = np.zeros((0, 5))

        if not annotationIds or len(annotationIds) == 0:
            return default

        cocoAnnotations = self.coco.loadAnns(annotationIds)

        for idx, ann in enumerate(cocoAnnotations):
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = ann['bbox']
            annotation[0, 4] = self.coco2label(ann['category_id'])
            annotations = np.append(default, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def load_image(self, idx):
        """
        """

        info = self.coco.loadImgs(self.imageIds[idx])[0]
        path = os.path.join(self.root_dir, 'images', self.set, info['file_name'])
        image = cv.imread(path)
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        return img

def collate(data):
    """
    """
    images = [d['img'] for d in data]
    annotations = [d['annot'] for d in data]
    scales = [d['scale'] for d in data]

    images = torch.from_numpy(np.stack(images, axis=0))
    max_annotations = max(annotation.shape[0] for annotation in annotations)
    if max_annotations > 0:
        annotations_padded = torch.ones((len(annotations), max_annotations, 5)) * -1
        if max_annotations > 0:
            for idx, annotation in enumerate(annotations):
                if annotation.shape[0] > 0:
                    annotations_padded[idx, :annotation.shape[0], :] = annotation
    else:
        annotations_padded = torch.ones((len(annotations), 1, 5)) * -1

    images = images.permute(0, 3, 1, 2)
    return {
        'img': images,
        'annot': annotations_padded,
        'scale': scales
    }
