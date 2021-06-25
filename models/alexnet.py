from __future__ import print_function

import torch
import torch.nn as nn


class MyAlexNetCMC(nn.Module):
    def __init__(self, feat_dim=128):
        super(MyAlexNetCMC, self).__init__()
        self.encoder = alexnet(feat_dim=feat_dim)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, sample, layer=8):
        return self.encoder(sample, layer)


class alexnet(nn.Module):
    def __init__(self, feat_dim=128):
        super(alexnet, self).__init__()
        # self.anchor_net = alexnet_half(in_channel=3, feat_dim=feat_dim)
        self.sample_net = alexnet_half(in_channel=2, feat_dim=feat_dim)

    def forward(self, sample, layer=8):
        feat_sample = self.sample_net(sample, layer)
        return feat_sample

class alexnet_half(nn.Module):
    def __init__(self, in_channel=1, feat_dim=128):
        super(alexnet_half, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96//2, 11, 4, 2, bias=False),
            nn.BatchNorm2d(96//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96//2, 256//2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256//2, 384//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384//2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384//2, 384//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384//2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384//2, 256//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.l2norm = Normalize(2)

    def forward(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':

    sample = torch.rand(10, 3, 224, 224)
    model = MyAlexNetCMC(128)
    feat = model(sample)
    print(feat.shape)
    pass

