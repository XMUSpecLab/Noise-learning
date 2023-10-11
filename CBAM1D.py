"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-13
"""

from abc import ABC
import torch
import torch.nn as nn


class ChannelAttention1D(nn.Module, ABC):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.MaxPool1d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention1D(nn.Module, ABC):
    def __init__(self):
        super(SpatialAttention1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], 1)
        out = self.sigmoid(self.conv1d(out))
        return out


class CBAM(nn.Module, ABC):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention1D(channel)
        self.spatial_attention = SpatialAttention1D()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module, ABC):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(),
            nn.Conv1d(in_channels=places, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(),
            nn.Conv1d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places * self.expansion)
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(places * self.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    model = ResBlock_CBAM(in_places=16, places=4, downsampling=4)
    print(model)

    data = torch.randn(64, 16, 1328)  # Batch, Channel, Size of the data
    out = model(data)
    print(out.shape)
