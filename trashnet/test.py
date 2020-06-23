"""
Runner script to test the trained model.
"""

import argparse
import os
import shutil
import cv2 as cv

from effdet.data.dataset import Coco
from effdet.data.transforms import Normalize
from effdet.data.transforms import Resize
from constants import classes as CLASSES
from constants import colors
import torch
from torchvision import transforms

def argument_parser(epilog: str = None) -> argparse.ArgumentParser:
    """
    Create an argument parser for accepting testing loop thresholds and paths to output and pretrained model directories.

    Args:
        epilog (str): epilog passed to ArgumentParser describing usage.
    
    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(epilog=epilog or f"""
    Example:
        python test.py --path /path/to/data/folder/root --pretrained /path/to/pretrained/model/dir --output /path/to/output/dir  # noqa: F541
    """)

    parser.add_argument("--image_size", type=int, default=512, help="The height and width for images passed to the network")
    parser.add_argument("--cls_threshold", type=float, default=0.2, help="Threshold for classification score")
    parser.add_argument("--nms_threshold", type=float, default=0.2, help="Threshold for regressor boxes")
    parser.add_argument("--path", "-p", type=str, help="Path to root folder of data in MS-COCO format")
    parser.add_argument("--pretrained", type=str, default="models/trashnet.pth", help="Path to trained model")
    parser.add_argument("--output", type=str, default="predictions")

    arg = parser.parse_args()
    return arg

def test(args: argparse.Namespace) -> None:
    """
    Testing loop to generate testset dataloaders and predictions on the testset.

    Args:
        args (arparse.NameSpace): argparse object containing parsed user command line input.

    Returns:
        None
    """

    model = torch.load(args.pretrained).module
    if torch.cuda.is_available():
        model = model.cuda()

    test_ds = Coco(args.path, data="valid", transforms=transforms.Compose([Normalize(), Resize()]))

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    for idx in range(len(test_ds)):
        data = test_ds[idx]
        scale = data['scale']
        with torch.no_grad():
            if torch.cuda.is_available():
                score, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            else:
                score, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale

            if boxes.shape[0] > 0:
                info = test_ds.coco.loadImgs(test_ds.imageIds[idx])[0]
                path = os.path.join(test_ds.root_dir, 'images', test_ds.set, info['file_name'])
                output_image = cv.imread(path)

                for box_id in range(boxes.shape[0]):
                    prob = float(score[box_id])
                    if prob < args.cls_threshold:
                        break
                    label = int(labels[box_id])
                    xmin, ymin, xmax, ymax = boxes[box_id, :]
                    color = colors[label]
                    cv.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                    text = cv.getTextSize(CLASSES[label] + ' : %.2f' % prob, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]

                    cv.rectangle(output_image, (xmin, ymin), (xmin + text[0] + 3, ymin + text[1] + 4), color, -1)

                    cv.putText(
                        output_image, CLASSES[label] + ' : %.2f' % prob,
                        (xmin, ymin + text[1] + 4), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1
                    )
                
                cv.imwrite("{}/{}_pred.jpg".format(args.output, info["file_name"][:-4]), output_image)


if __name__ == "__main__":
    arg = argument_parser()
    test(arg)