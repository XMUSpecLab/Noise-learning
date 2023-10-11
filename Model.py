"""
The 1-D U-Net model with channel attention  for instrumental noise modeling
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date:2020-09-12
"""
from abc import ABC
import torch
from torch import nn
from CBAM1D import CBAM


class Conv_block(nn.Module, ABC):
    def __init__(self, ch_in, ch_out):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up_conv(nn.Module, ABC):
    def __init__(self, ch_in, ch_out):
        super(Up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


# Traditional U-net
class UNet(nn.Module, ABC):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = Conv_block(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block(ch_in=256, ch_out=512)
        self.Conv5 = Conv_block(ch_in=512, ch_out=1024)

        self.Up5 = Up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = Conv_block(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = Conv_block(ch_in=512, ch_out=256)

        self.Up3 = Up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = Conv_block(ch_in=256, ch_out=128)

        self.Up2 = Up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = Conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# U-Net with channel and spatial attention

class Conv_block_CA(nn.Module, ABC):
    def __init__(self, ch_in, ch_out):
        super(Conv_block_CA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )
        self.cbam = CBAM(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


class Up_conv_CA(nn.Module, ABC):
    def __init__(self, ch_in, ch_out):
        super(Up_conv_CA, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )
        self.cbam = CBAM(ch_out)

    def forward(self, x):
        x = self.up(x)
        x = self.cbam(x)
        return x


class UNet_CA(nn.Module, ABC):
    def __init__(self, ch_in=1, ch_out=1):
        super(UNet_CA, self).__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block_CA(ch_in=ch_in, ch_out=64)
        self.Conv2 = Conv_block_CA(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block_CA(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block_CA(ch_in=256, ch_out=512)
        self.Conv5 = Conv_block_CA(ch_in=512, ch_out=1024)

        self.Up5 = Up_conv_CA(ch_in=1024, ch_out=512)
        self.Up_conv5 = Conv_block_CA(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv_CA(ch_in=512, ch_out=256)
        self.Up_conv4 = Conv_block_CA(ch_in=512, ch_out=256)

        self.Up3 = Up_conv_CA(ch_in=256, ch_out=128)
        self.Up_conv3 = Conv_block_CA(ch_in=256, ch_out=128)

        self.Up2 = Up_conv_CA(ch_in=128, ch_out=64)
        self.Up_conv2 = Conv_block_CA(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNet_CA6(nn.Module, ABC):
    def __init__(self, ch_in=1, ch_out=1):
        super(UNet_CA6, self).__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block_CA(ch_in=ch_in, ch_out=64)
        self.Conv2 = Conv_block_CA(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block_CA(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block_CA(ch_in=256, ch_out=512)
        self.Conv5 = Conv_block_CA(ch_in=512, ch_out=1024)
        self.Conv6 = Conv_block_CA(ch_in=1024, ch_out=2048)

        self.Up6 = Up_conv_CA(ch_in=2048, ch_out=1024)
        self.Up_conv6 = Conv_block_CA(ch_in=2048, ch_out=1024)

        self.Up5 = Up_conv_CA(ch_in=1024, ch_out=512)
        self.Up_conv5 = Conv_block_CA(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv_CA(ch_in=512, ch_out=256)
        self.Up_conv4 = Conv_block_CA(ch_in=512, ch_out=256)

        self.Up3 = Up_conv_CA(ch_in=256, ch_out=128)
        self.Up_conv3 = Conv_block_CA(ch_in=256, ch_out=128)

        self.Up2 = Up_conv_CA(ch_in=128, ch_out=64)
        self.Up_conv2 = Conv_block_CA(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        # decoding + concat path
        d6 = self.Up6(x6)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNet_CA7(nn.Module, ABC):
    def __init__(self, ch_in=1, ch_out=1):
        super(UNet_CA7, self).__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block_CA(ch_in=ch_in, ch_out=64)
        self.Conv2 = Conv_block_CA(ch_in=64, ch_out=128)
        self.Conv3 = Conv_block_CA(ch_in=128, ch_out=256)
        self.Conv4 = Conv_block_CA(ch_in=256, ch_out=512)
        self.Conv5 = Conv_block_CA(ch_in=512, ch_out=1024)
        self.Conv6 = Conv_block_CA(ch_in=1024, ch_out=2048)
        self.Conv7 = Conv_block_CA(ch_in=2048, ch_out=4096)

        self.Up7 = Up_conv_CA(ch_in=4096, ch_out=2048)
        self.Up_conv7 = Conv_block_CA(ch_in=4096, ch_out=2048)

        self.Up6 = Up_conv_CA(ch_in=2048, ch_out=1024)
        self.Up_conv6 = Conv_block_CA(ch_in=2048, ch_out=1024)

        self.Up5 = Up_conv_CA(ch_in=1024, ch_out=512)
        self.Up_conv5 = Conv_block_CA(ch_in=1024, ch_out=512)

        self.Up4 = Up_conv_CA(ch_in=512, ch_out=256)
        self.Up_conv4 = Conv_block_CA(ch_in=512, ch_out=256)

        self.Up3 = Up_conv_CA(ch_in=256, ch_out=128)
        self.Up_conv3 = Conv_block_CA(ch_in=256, ch_out=128)

        self.Up2 = Up_conv_CA(ch_in=128, ch_out=64)
        self.Up_conv2 = Conv_block_CA(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        x7 = self.Maxpool(x6)
        x7 = self.Conv7(x7)

        # decoding + concat path
        d7 = self.Up7(x7)
        d7 = torch.cat((x6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(d7)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


if __name__ == '__main__':
    net1 = UNet(1, 1)
    print(net1)
    data = torch.randn(64, 1, 1600)  # Batch, Channel, Size of the data
    out1 = net1(data)
    print(out1.shape)
