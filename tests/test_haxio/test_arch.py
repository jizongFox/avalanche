from unittest import TestCase

import torch

from haxio.models.multihead import resnet18


class TestResNet(TestCase):
    def test_resnet(self):
        resnet = resnet18()
        image = torch.randn(10, 3, 224, 224)
        labels = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 0, 0, 0]).long()
        pred = resnet(image, labels)
