"""
"""

import argparse
import os
import logging
import numpy as np
import torch
import cv2 as cv

from constants import classes, colors

logger = logging.getLogger('logger')
mean = np.array([[[0.485, 0.456, 0.406]]])
std = np.array([[[0.229, 0.224, 0.225]]])

def main():
    
    parser = argument_parser()
    args = parser.parse_args()
    image_folder = args.folder
    image_path = args.image_path
    image_size = args.image_size
    threshold = args.threshold
    model_path = args.model_path
    model_name = args.model_name
    output = args.output

    path = os.path.join(model_path, model_name)

    model = torch.load(path).module

    if torch.cuda.is_available():
        model = model.cuda()
    
    if args.batch:
        infer_batch(model, image_folder, image_size, classes, threshold, output)
    else:
        infer(model, image_path, image_size, classes, threshold, output)

def infer_batch(model, folder, image_size, class_list, threshold, output):
    """
    """

    filenames = os.listdir(folder)
    cnt = 0
    for filename in filenames:
        path = os.path.join(folder, filename)

        try:
            infer(model, path, class_list, threshold, output)
            cnt += 1
        except (FileNotFoundError) as e:
            logger.error("Failed to infer on " + path)
            print(e)
            continue

    print("Done with inference. Images saved in output dir")

def infer(model, path, size, class_list, threshold, output):
    """
    """
    if not os.path.exists(output):
        os.makedirs(output)

    filename = os.path.basename(path)
    img = load_image(path)
    h, w, c = img.shape
    
    if h > w:
        scale = size / h
        resized_h = size
        resized_w = int(w * scale)
    else:
        scale = size / w
        resized_h = int(h * scale)
        resized_w = size

    image = cv.resize(img, (resized_w, resized_h))
    new_image = np.zeros((size, size, 3))
    new_image[0:resized_h, 0:resized_w] = image
    
    sample = torch.from_numpy(new_image)

    if torch.cuda.is_available():
        sample = sample.cuda()

    with torch.no_grad():
        scores, labels, boxes = model(sample.permute(2, 0, 1).float().unsqueeze(dim=0))
        boxes /= scale
    try:
        if boxes.shape[0] > 0:
            output_image = cv.imread(path)

            for box in range(boxes.shape[0]):
                prob = float(scores[box])
                if prob < threshold:
                    break
                label = int(labels[box])
                xmin, xmax, ymin, ymax = boxes[box, :]
                color = colors[label]

                cv.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text = cv.getTextSize(class_list[label] + " : %.2f" % prob,
                                      cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv.rectangle(output_image, (xmin, ymin), (xmin + text[0] + 3, ymin + text[1] + 4),
                             color, -1)

                cv.putText(
                    output_image, class_list[label] + " : %.2f" % prob,
                    (xmin, ymin + text[1] + 4), cv.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1
                )

        cv.imwrite(os.path.join(output, filename), output_image)
        cv.imwrite("output.jpg", output_image)
        return scores, labels, boxes

    except Exception as e:
        logger.debug(e)
        return None

def load_image(image):
    img = cv.imread(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    image = img.astype(np.float32) - mean / std

    return image

def argument_parser(epilog: str = None):
    """
    """

    parser = argparse.ArgumentParser(epilog=epilog or f"""
    Example:
        python infer.py --image_path /path/to/dat/folder/root --model_path models --model_name MODELNAME # noqa: F541
    """)

    parser.add_argument("--image_size", type=int, default=512, help="The height and width for images passed to the network")
    parser.add_argument("--image_path", "-p", type=str, help="Path to images to be used for inference")
    parser.add_argument("--folder", type=str, help="Path to image folder for batch inference")
    parser.add_argument("--model_path", type=str, default="models", help="Path to save model directory")
    parser.add_argument("--model_name", type=str, help="Name of checkpoint model to use fo inference", required=True)
    parser.add_argument("--batch", action="store_true", help="Do batch infernce?")
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    main()