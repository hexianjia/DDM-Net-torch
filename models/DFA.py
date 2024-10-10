import torch
import torch.nn as nn
import torch.nn.functional as F
from .CAM import ChannelAttention


class DFA(nn.Module):
    def __init__(self, in_channels):
        super(DFA, self).__init__()

        self.CAM = ChannelAttention(in_channels)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                   nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                   nn.BatchNorm2d(in_channels), nn.ReLU())

        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 7, 1, 3),
                                     nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 7, 1, 3),
                                     nn.BatchNorm2d(in_channels), nn.ReLU())

        self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 5, 1, 2),
                                     nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv4_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 5, 1, 2),
                                     nn.BatchNorm2d(in_channels), nn.ReLU())

        self.conv5_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                     nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv5_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                     nn.BatchNorm2d(in_channels), nn.ReLU())

        self.down_sampling_1 = nn.AvgPool2d(kernel_size=2)
        self.down_sampling_2 = nn.AvgPool2d(kernel_size=2)
        self.down_sampling_3 = nn.AvgPool2d(kernel_size=2)

        self.up_sampling_1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.up_sampling_2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.up_sampling_3 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)

        # self.up_sampling_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampling_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampling_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        channels_weight = self.CAM(x)  # c*1*1
        cam_out = x * channels_weight  # c*32*32

        conv1 = self.conv1(cam_out)  # c*32*32
        conv2 = self.conv2(x)  # c*32*32

        down1 = self.down_sampling_1(x)  # c*16*16
        conv3_1 = self.conv3_1(down1)  # c*16*16
        conv3_2 = self.conv3_2(conv3_1)  # c*16*16

        down2 = self.down_sampling_2(conv3_1)  # c*8*8
        conv4_1 = self.conv4_1(down2)  # c*8*8
        conv4_2 = self.conv4_2(conv4_1)  # c*8*8

        down3 = self.down_sampling_3(conv4_1)  # c*4*4
        conv5_1 = self.conv5_1(down3)  # c*4*4
        conv5_2 = self.conv5_2(conv5_1)  # c*4*4

        up1 = self.up_sampling_1(conv5_2)
        up2 = self.up_sampling_2(conv4_2 + up1)
        up3 = self.up_sampling_3(conv3_2 + up2)

        return up3 * conv2 + conv1


if __name__ == '__main__':
    t = torch.rand(1, 3, 32, 32)
    net = DFA(in_channels=3)

    t2 = net(t)
    print(t2.shape)

    print(t)
    print(t2)
