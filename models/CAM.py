import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    # 初始化，in_planes参数指定了输入特征图的通道数，ratio参数用于控制通道注意力机制中特征降维和升维过程中的压缩比率。默认值为8
    def __init__(self, in_channels, ratio=16):
        # 继承父类初始化方法
        super(ChannelAttention, self).__init__()
        # 全局最大池化 [c,h,w]==>[c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # H*W
        # 全局平均池化 [c,h,w]==>[c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))  # H*W

        # 使用1x1x1卷积核代替全连接层进行特征降维
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        # 激活函数
        self.relu1 = nn.ReLU()
        # 使用1x1x1卷积核进行特征升维
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过平均池化和最大池化后的特征进行卷积、激活、再卷积操作
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # average pooling-->conv-->RELu-->conv
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # max pooling-->conv-->RELu-->conv

        out = avg_out + max_out

        return self.sigmoid(out)


if __name__ == '__main__':
    t = torch.rand(2, 3, 32, 32)
    net = ChannelAttention(in_channels=3)

    t2 = net(t)
    print(t2.shape)
