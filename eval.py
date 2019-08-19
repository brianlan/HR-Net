import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.hrnet import HRNet
from src.heads import SegHead


def read_im():
    im = Image.open('data/night.jpg')
    im = TF.crop(im, 16, 0, 704, 1280)
    im = TF.to_tensor(im)
    im = im - 0.5
    return im[None, ...]


def evaluate():
    im = read_im().cuda()

    model = torch.nn.Sequential(
        HRNet(upsample_mode='bilinear'),
        SegHead(360, 2),
        torch.nn.Upsample(scale_factor=4, mode='bilinear')
    )

    model.load_state_dict(torch.load('checkpoints/hrnet/final_model.pth'))
    model.cuda()

    with torch.no_grad():
        out = model(im)
        seg_mask = (out > 0).cpu().numpy().astype(np.uint8)
        # _im = ((im[0].permute(1, 2, 0) + 0.5) * 255).cpu().numpy().astype(np.uint8)
        # _im = cv2.addWeighted(_im, 1, seg_mask, 0.5, 0)
        # plt.imshow(seg_mask[0, 0] * 255)
        # plt.show()
        cv2.imwrite('results/car.png', seg_mask[0, 0] * 255)
        cv2.imwrite('results/lane.png', seg_mask[0, 1] * 255)


if __name__ == '__main__':
    evaluate()
