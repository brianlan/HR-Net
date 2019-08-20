import torch
import numpy as np

from ..hrnet import HighResolutionNet


def test_model_definition():
    net = HighResolutionNet(upsample_mode='bilinear')
    im = torch.randn(1, 3, 640, 320)
    with torch.no_grad():
        out = net(im)
    print(out.shape)
