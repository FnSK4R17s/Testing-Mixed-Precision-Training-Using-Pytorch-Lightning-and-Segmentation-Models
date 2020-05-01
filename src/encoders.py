import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.base_model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.base_model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        self.l0 = nn.Linear(512, 1024)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.base_model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        return l0


class SE_ResNeXt50_32x4d(nn.Module):
    def __init__(self, pretrained):
        super(SE_ResNeXt50_32x4d, self).__init__()
        if pretrained is True:
            self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        else:
            self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)

        self.l0 = nn.Linear(2048, 1024)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.base_model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        return l0

class SE_ResNeXt101_32x4d(nn.Module):
    def __init__(self, pretrained):
        super(SE_ResNeXt101_32x4d, self).__init__()
        if pretrained is True:
            self.base_model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained='imagenet')
        else:
            self.base_model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained=None)

        self.l0 = nn.Linear(2048, 1024)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.base_model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        return l0