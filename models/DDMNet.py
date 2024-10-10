# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import netConv2, netUp, netUp2
from .DFA import DFA
from .MEF import MEF
# from torchvision import models


class DDMNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,feature_scale=4,
                 is_deconv=False, is_batchnorm=True):
        super(DDMNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        # down sampling
        self.conv1 = netConv2(self.n_channels, 64, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = netConv2(64, 128, self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = netConv2(128, 256, self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = netConv2(256, 512, self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = netConv2(512, 512, self.is_batchnorm)

        # center
        self.center = DFA(in_channels=512)

        # upsampling
        self.up_concat4 = netUp(1024, 256, self.is_deconv)
        self.up_concat3 = netUp(512, 128, self.is_deconv)
        self.up_concat2 = netUp(256, 64, self.is_deconv)
        self.up_concat1 = netUp2(128, 64, self.is_deconv)

        # self.outconv1 = nn.Conv2d(64, self.n_classes, 3, padding=1)
        # self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)

        self.outconv1 = MEF(128, self.n_classes)
        self.outconv2 = MEF(128, 1)

    def forward(self, x):
        conv1 = self.conv1(x)  # 64*512*512
        maxpool1 = self.maxpool1(conv1)  # 64*256*256

        conv2 = self.conv2(maxpool1)  # 128*256*256
        maxpool2 = self.maxpool2(conv2)  # 128*128*128

        conv3 = self.conv3(maxpool2)  # 256*128*128
        maxpool3 = self.maxpool3(conv3)  # 256*64*64

        conv4 = self.conv4(maxpool3)  # 512*64*64
        maxpool4 = self.maxpool4(conv4)  # 512*32*32

        conv5 = self.conv5(maxpool4)  # 512*32*32

        center = self.center(conv5)  # 512*32*32

        up4 = self.up_concat4(center, conv4)  # 256*64*64
        up3 = self.up_concat3(up4, conv3)  # 128*128*128
        up2 = self.up_concat2(up3, conv2)  # 64*256*256
        up1 = self.up_concat1(up2, conv1)  # 64*512*512

        d1 = self.outconv1(up1)  # 1*512*512
        d2 = self.outconv2(up1)  # 1*512*512

        # return up1
        return d1, d2


if __name__ == '__main__':
    t = torch.randint(0, 255, (1, 3, 512, 512))

    net = DDMNet()
    t2 = net(t.to(torch.float32))

    # net = nn.Conv2d(3, 64, 3, 1, 1)

    # up = nn.UpsamplingBilinear2d(scale_factor=2)
    #
    # t2 = net(t.to(torch.float32))
    # t3 = up(t2)
    # t4 = up(t3)
    #
    # print(t.shape)
    # print(t2.shape)
    # print(t3.shape)
    # print(t4.shape)

