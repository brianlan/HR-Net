import pathlib
import functools

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.hrnet import HRNet
from src.heads import SegHead
from src.loss import dice_loss


def extract_mask(raw_png):
    return (raw_png > 0).astype(np.float32)[..., 0]


def read_seglabel():
    car_mask = extract_mask(cv2.imread('data/car-mask.png'))
    lane_mask = extract_mask(cv2.imread('data/lane-mask.png'))
    mask = np.vstack((car_mask[None, ...], lane_mask[None, ...]))
    return torch.tensor(mask[:, 16:, :])[None, ...]


def read_im():
    im = Image.open('data/night.jpg')
    im = TF.crop(im, 16, 0, 704, 1280)
    im = TF.to_tensor(im)
    im = im - 0.5
    return im[None, ...]


def save_model(model, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def calc_loss(epoch, pred, label):
    car_pred, lane_pred = pred[:, 0:1], pred[:, 1:2]
    car_label, lane_label = label[:, 0:1], label[:, 1:2]
    bce_loss = torch.nn.BCEWithLogitsLoss()
    losses = {
        "car_bce_loss": bce_loss(car_pred, car_label) * 5.0,
        "car_dice_loss": dice_loss(car_pred, car_label) * 0.1,
        "lane_bce_loss": bce_loss(lane_pred, lane_label) * 5.0,
        "lane_dice_loss": dice_loss(lane_pred, lane_label) * 0.1,
    }
    total_loss = functools.reduce(lambda x, y: x + y, [l for n, l in losses.items()])
    print(f"[epoch {epoch}] total_loss: {total_loss:.4f}")
    for n, l in losses.items():
        print(f"{n:15}: {l.item():.4f}")
    return total_loss


def train():
    im = read_im().cuda()
    label = read_seglabel().cuda()
    model = torch.nn.Sequential(
        HRNet(upsample_mode='bilinear'),
        SegHead(360, 2),
        torch.nn.Upsample(scale_factor=4, mode='bilinear')
    )
    model.train()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 800])
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(im)
        total_loss = calc_loss(epoch, out, label)
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)

    save_model(model, 'checkpoints/hrnet/final_model.pth')


if __name__ == '__main__':
    train()
