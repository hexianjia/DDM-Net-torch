import torch
import torch.nn as nn
import torch.nn.functional as F


class MEF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MEF, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.upsample = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=False))

        self.conv5 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.maxpool(x) + self.avgpool(x)
        x2 = self.conv1(x1)
        x3 = self.upsample(x2)

        x4 = self.conv2(x)
        x5 = self.conv3(x4)

        x6 = self.conv5(torch.cat([x3, x5], dim=1))
        # x6 = self.conv5(x3 + x5)
        return x6


if __name__ == '__main__':
    x = torch.rand((2, 128, 512, 512))
    net = boundary_module(128, 1)

    y = net(x)

    print(x.shape)
    print(y.shape)
