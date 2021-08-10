#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   res_unet.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/18 18:46   Bot Zhao      1.0         None
"""

# import lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F


class Dilate_block(nn.Module):
    def __init__(self, channel):
        super(Dilate_block, self).__init__()
        self.dilate1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU()
        )
        self.dilate2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU()
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4),
            nn.LeakyReLU()
        )
        self.dilate4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        # dilate5_out = self.dilate5(dilate4_out)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel, drop_rate=0):
        super(DoubleConv, self).__init__()
        if drop_rate > 0:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.Dropout2d(drop_rate),
                nn.LeakyReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.Dropout2d(drop_rate),
                nn.LeakyReLU(),
            )

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.LeakyReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.LeakyReLU(),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Cls_layer(nn.Module):
    def __init__(self):
        super(Cls_layer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d()
        self.avg_pool = nn.AdaptiveAvgPool2d()
    def forward(self, x):
        pass


class Res_Unet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, drop_rate=0):
        super(Res_Unet34, self).__init__()

        filters = [64, 128, 256, 512, 512]
        resnet = models.resnet34(pretrained=False)
        self.inputs = nn.Sequential(
            nn.Conv2d(num_channels, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
                      resnet.bn1,
                      resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dilate_block(512)

        self.doubleConv5 = DoubleConv(filters[4], filters[3], drop_rate=drop_rate)
        self.doubleConv6 = DoubleConv(filters[3], filters[2], drop_rate=drop_rate)
        self.doubleConv7 = DoubleConv(filters[2], filters[1], drop_rate=drop_rate)
        self.doubleConv8 = DoubleConv(filters[1], filters[0], drop_rate=drop_rate)
        self.up1 = nn.ConvTranspose2d(filters[4], filters[4] // 2, 2, stride=(2, 2))
        self.up2 = nn.ConvTranspose2d(filters[3], filters[3] // 2, 2, stride=(2, 2))
        self.up3 = nn.ConvTranspose2d(filters[2], filters[2] // 2, 2, stride=(2, 2))
        self.up4 = nn.ConvTranspose2d(filters[1], filters[1] // 2, 2, stride=(2, 2))

        self.out = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1),
            # nn.Softmax()
        )

    def forward(self, x):
        # Encoder
        x = self.inputs(x)             # 64, w, h
        e1 = self.encoder1(x)          # 64, w//2, h//2
        e2 = self.encoder2(e1)         # 128, w//2, h//2
        e3 = self.encoder3(e2)         # 256, w//2, h//2
        e4 = self.encoder4(e3)         # 512, w//2, h//2

        # Center
        e4 = self.dblock(e4)

        # Decoder
        x = self.up1(e4)
        x = self.doubleConv5(torch.cat([e4, x], 1))
        x = self.up2(x)
        x = self.doubleConv6(torch.cat([e3, x], 1))
        x = self.up3(x)
        x = self.doubleConv7(torch.cat([e2, x], 1))
        x = self.up4(x)
        x = self.doubleConv8(torch.cat([e1, x], 1))
        x = self.out(x)
        x = self.out(x)

        return x


if __name__ == '__main__':
    model = Res_Unet34(num_classes=2, num_channels=1)
    x = torch.randn((1,1, 256, 256))
    y = model(x)
