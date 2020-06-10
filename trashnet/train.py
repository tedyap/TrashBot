"""
"""

import argparse
import os
import shutil
import logging
import numpy as np

from effdet.data.dataset import collate
from effdet.data.dataset import Coco
from effdet.data.transforms import Normalize
from effdet.data.transforms import Augment
from effdet.data.transforms import Resize
from effdet.models import EfficientDet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

def argument_parser(epilog: str = None):
    """
    """

    parser = argparse.ArgumentParser(epilog=epilog or f"""
    Example:
        python train.py --meta /path/to/meta.json --annotations /path/to/annotations/folder --output /path/to/output.json # noqa: E501, F541
    """)

    parser.add_argument("--image_size", type=int, default=512, help="The height and width for images passed to the network")
    parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Number of images per batch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--num_epochs", "-n", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epochs between testing")
    parser.add_argument("--es_min_delta", type=float, default=0.0, help="Early stopping: Minimum change in loss to qualify as improvement")
    parser.add_argument("--es_patience", type=float, default=0.0, help="Number of epochs without improvement to stop training")
    parser.add_argument("--path", "-p", type=str, help="Path to root folder of data in MS-COCO format")
    parser.add_argument("--log", type=str, default="tensorboard/", help="Path to tensorboard log directory")
    parser.add_argument("--save_path", type=str, default="models", help="Path to model save directory")

    arg = parser.parse_args()
    return arg

def train(args):
    """
    """

    logger = logging.getLogger('logger')
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        torch.cuda.manuel_seed(42)
    else:
        torch.manuel_seed(42)
    
    train_dict = {
        'bs': args.batch_size * n_gpus,
        'shuffle': True,
        'drop_last': True,
        'collate_fn': collate,
        'num_workers': 4
    }

    test_dict = {
        'bs': args.batch_size,
        'shuffle': False,
        'drop_last': False,
        'collate_fn': collate,
        'num_workers': 4
    }

    train_ds = Coco(root=args.path, data="train", transforms=transforms.Compose(
        [Normalize(), Augment(), Resize()]
    ))
    train_dl = DataLoader(train_ds, **train_dict)

    test_ds = Coco(root=args.path, data="valid", transforms=transforms.Compose(
        [Normalize(), Resize()]
    ))
    test_dl = DataLoader(test_ds, **test_dict)

    model = EfficientDet(n_classes=train_ds.num_classes())

    if os.path.isdir(args.log):
        shutil.rmtree(args.log)
    os.makedirs(args.log)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    writer = SummaryWriter(args.log)

    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    iters_per_epoch = len(train_dl)

    for epoch in range(args.num_epochs):
        model.train()
        losses = []

        prog_bar = tqdm(train_dl)

        for iteration, data in enumerate(prog_bar):
            try:
                optim.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = model([data['img'].float(), data['annot']])
                
                classification_loss, regression_loss = classification_loss.mean(), regression_loss.mean()
                loss = classification_loss + regression_loss
                if loss == 0:
                    continue

                loss.backward()

                nn.utils.clip_grad_norm(model.parameters(), 0.1)

                optim.step()
                losses.append(float(loss))
                total_loss = np.mean(losses)

                writer.add_scalar('Train/Total Loss', total_loss, epoch * iters_per_epoch + iteration)
                writer.add_scalar('Train/Regression Loss', regression_loss, epoch * iters_per_epoch + iteration)
                writer.add_scalar('Train/Focal Loss', classification_loss, epoch * iters_per_epoch + iteration)
            
            except Exception:
                logger.error(Exception)
                continue
        
        schedule.step(np.mean(losses))

        if epoch % args.test_interval == 0:
            model.eval()
            regression_losses, classification_losses = [], []

            for iteration, data in enumerate(test_dl):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        classification_loss, regression_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        classification_loss, regression_loss = model([data['img'].float(), data['annot']])
                    
                    classification_loss, regression_loss = classification_loss.mean(), regression_loss.mean()

                    classification_losses.append(float(classification_loss))
                    regression_losses.append(float(regression_loss))
            
            cls_loss = np.mean(classification_losses)
            reg_loss = np.mean(regression_losses)

            loss = cls_loss + reg_loss
                        
            writer.add_scalar('Val/Total_loss', loss, epoch)
            writer.add_scalar('Val/Regression Loss', reg_loss, epoch)
            writer.add_scalar('Val/Classfication Loss (focal loss)', cls_loss, epoch)

            if loss + args.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(args.save_path, "trashnet_" + str(epoch) + ".pth"))

            if epoch - best_epoch > args.es_patience > 0:
                logger.debug("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break

    writer.close()


if __name__ == "__main__":
    args = argument_parser()
    train(args)