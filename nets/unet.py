#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   unet.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/5 14:43   Bot Zhao      1.0         None
"""

# import lib
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
import segmentation_models_pytorch as smp


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


class Unet(nn.Module):
    def __init__(self, inchannel, n_class, drop_rate=0, filters=None):
        super(Unet, self).__init__()
        if not filters:
            filters = [64, 128, 256, 512, 1024]
        self.doubleConv1 = DoubleConv(inchannel, filters[0], drop_rate=drop_rate)
        self.doubleConv2 = DoubleConv(filters[0], filters[1], drop_rate=drop_rate)
        self.doubleConv3 = DoubleConv(filters[1], filters[2], drop_rate=drop_rate)
        self.doubleConv4 = DoubleConv(filters[2], filters[3], drop_rate=drop_rate)
        self.doubleConvBottom = DoubleConv(filters[3], filters[4], drop_rate=drop_rate)
        self.down = nn.MaxPool2d(2)
        self.doubleConv5 = DoubleConv(filters[4], filters[3], drop_rate=drop_rate)
        self.doubleConv6 = DoubleConv(filters[3], filters[2], drop_rate=drop_rate)
        self.doubleConv7 = DoubleConv(filters[2], filters[1], drop_rate=drop_rate)
        self.doubleConv8 = DoubleConv(filters[1], filters[0], drop_rate=drop_rate)
        self.up1 = nn.ConvTranspose2d(filters[4], filters[4] // 2, 2, stride=(2, 2))
        self.up2 = nn.ConvTranspose2d(filters[3], filters[3] // 2, 2, stride=(2, 2))
        self.up3 = nn.ConvTranspose2d(filters[2], filters[2] // 2, 2, stride=(2, 2))
        self.up4 = nn.ConvTranspose2d(filters[1], filters[1] // 2, 2, stride=(2, 2))
        self.out = nn.Sequential(
            nn.Conv2d(filters[0], n_class, 1),
            # nn.Softmax()
        )

    def forward(self, x):
        x1 = self.doubleConv1(x)
        x = self.down(x1)
        x2 = self.doubleConv2(x)
        x = self.down(x2)
        x3 = self.doubleConv3(x)
        x = self.down(x3)
        x4 = self.doubleConv4(x)
        x = self.down(x4)
        x = self.doubleConvBottom(x)
        x = self.up1(x)
        x = self.doubleConv5(torch.cat([x4, x], 1))
        x = self.up2(x)
        x = self.doubleConv6(torch.cat([x3, x], 1))
        x = self.up3(x)
        x = self.doubleConv7(torch.cat([x2, x], 1))
        x = self.up4(x)
        x = self.doubleConv8(torch.cat([x1, x], 1))
        x = self.out(x)
        return x


class Multitask_Unet(nn.Module):
    def __init__(self, inchannel, seg_n_class, cls_n_class, drop_rate=0, filters=None):
        super(Multitask_Unet, self).__init__()
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=drop_rate,  # dropout ratio, default is None
            activation='softmax',  # activation function, default is None
            classes=cls_n_class,  # define number of output labels
        )
        self.model = smp.Unet(
                        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=inchannel,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=seg_n_class,  # model output channels (number of classes in your dataset)
                        decoder_use_batchnorm=True,
                        aux_params=aux_params,
                    )

    def forward(self, x):
        mask, label = self.model(x)
        return mask, label

#
#
# class Unet(nn.Module):
#     def __init__(self, in_channels=1, n_classes=2,  feature_scale=2, filters=None, is_deconv=True, is_batchnorm=True):
#         super(Unet, self).__init__()
#         self.in_channels = in_channels
#         self.feature_scale = feature_scale
#         self.is_deconv = is_deconv
#         self.is_batchnorm = is_batchnorm
#
#         if not filters:
#             filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#         self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
#         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
#         # upsampling
#         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
#         self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
#         self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
#         self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
#         # final conv (without any concat)
#         self.final = nn.Conv2d(filters[0], n_classes, 1)
#
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, inputs):
#         conv1 = self.conv1(inputs)  # 16*512*512
#         maxpool1 = self.maxpool(conv1)  # 16*256*256
#
#         conv2 = self.conv2(maxpool1)  # 32*256*256
#         maxpool2 = self.maxpool(conv2)  # 32*128*128
#
#         conv3 = self.conv3(maxpool2)  # 64*128*128
#         maxpool3 = self.maxpool(conv3)  # 64*64*64
#
#         conv4 = self.conv4(maxpool3)  # 128*64*64
#         maxpool4 = self.maxpool(conv4)  # 128*32*32
#
#         center = self.center(maxpool4)  # 256*32*32
#         up4 = self.up_concat4(center, conv4)  # 128*64*64
#         up3 = self.up_concat3(up4, conv3)  # 64*128*128
#         up2 = self.up_concat2(up3, conv2)  # 32*256*256
#         up1 = self.up_concat1(up2, conv1)  # 16*512*512
#
#         final = self.final(up1)
#
#         return final
#
#
if __name__ == '__main__':
    import pdb
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # model = Unet(1, 2, drop_rate=0.3).cuda()
    model = Multitask_Unet(inchannel=1, seg_n_class=2, cls_n_class=2).cuda()
#     ###########################################
#     model = smp.Unet(
#         encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=2,                      # model output channels (number of classes in your dataset)
#     ).cuda()
    model.train()
    # summary(model, (1, 256, 256))

    x = torch.rand((8, 1, 144, 144, 144)).cuda()
    y = model(x)
    pdb.set_trace()
    print(y.cpu().detach().size())
#     pdb.set_trace()
#

