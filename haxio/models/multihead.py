import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from avalanche.models import MultiTaskModule, MultiHeadClassifier
from .singlehead import ResNet as _ResNet, model_urls, BasicBlock, Bottleneck


class ResNet(MultiTaskModule, _ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.logits(out, task_labels)
        return out

    def logits(self, input_, task_labels):  # noqa
        return self.fc(input_, task_labels)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=True, progress=True, input_dim=3, num_classes=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    resnet = _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )
    if input_dim != 3:
        resnet.conv1 = nn.Conv2d(
            input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    resnet.fc = MultiHeadClassifier(512 * BasicBlock.expansion)
    return resnet


def resnet34(pretrained=True, progress=True, input_dim=3, num_classes=None, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    resnet = _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )
    if input_dim != 3:
        resnet.conv1 = nn.Conv2d(
            input_dim, resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
    resnet.fc = MultiHeadClassifier(512 * resnet.block.expansion)
    return resnet


def resnet50(pretrained=True, progress=True, input_dim=3, num_classes=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    resnet = _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )
    if input_dim != 3:
        resnet.conv1 = nn.Conv2d(
            input_dim, resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
    resnet.fc = MultiHeadClassifier(512 * resnet.block.expansion)
    return resnet


def resnext50_32x4d(pretrained=True, progress=True, input_dim=3, num_classes=None, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    resnet = _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )
    if input_dim != 3:
        resnet.conv1 = nn.Conv2d(
            input_dim, resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
    resnet.fc = MultiHeadClassifier(512 * resnet.block.expansion)
    return resnet


def wide_resnet50_2(pretrained=True, progress=True, input_dim=3, num_classes=None, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    resnet = _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )
    if input_dim != 3:
        resnet.conv1 = nn.Conv2d(
            input_dim, resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
    resnet.fc = MultiHeadClassifier(512 * resnet.block.expansion)
    return resnet
