import pathlib

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.hrnet import HRNet
from src.heads import SegHead


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
    for i in range(1000):
        optimizer.zero_grad()
        out = model(im)
        car_loss = torch.nn.BCEWithLogitsLoss()(out[:, 0], label[:, 0])
        lane_loss = torch.nn.BCEWithLogitsLoss()(out[:, 1], label[:, 1])
        loss = car_loss + lane_loss
        print(f"[{i:04}] {'car_loss':10}: {car_loss.item():.4f}, {'lane_loss':10}: {lane_loss.item():.4f}, total_loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        lr_scheduler.step(i)

    save_model(model, 'checkpoints/hrnet/final_model.pth')


if __name__ == '__main__':
    train()
