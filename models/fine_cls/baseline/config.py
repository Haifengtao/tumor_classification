#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fine_cls.py
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/18 16:37   Bot Zhao      1.0         None
"""

# import lib
# import lib
from easydict import EasyDict as edict
import torch
import numpy as np
from nets import Losses
import os

__C = edict()
cfg = __C

__C.general = {}
__C.general.log_path = "../cpts/fine_cls/baseline/model_2"
__C.general.model_path = "../cpts/fine_cls/baseline/model_2"


__C.data = {}
__C.data.train_txt = r"/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_labeled_fine_train_v2.h5"
__C.data.val_txt = r"/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_labeled_fine_val_v2.h5"
__C.data.test_txt = r"/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_labeled_fine_test_v2.h5"
__C.data.resize_shape = [256, 256]

__C.net = {}
__C.net.name = "Unet"
__C.net.inchannels = 1
__C.net.classes = 3
__C.net.enc_nf = None
__C.net.with_tumor = True


__C.train = {}
__C.train.load_model = None                # "../cpts/baseline/model/final_model.pth"
__C.train.start_epoch = 0
__C.train.epochs = 500
__C.train.batch_size = 20
__C.train.train_iteres = 500
__C.train.learning_rate = 0.001
__C.train.warmRestart = 1

__C.train.gpu = '0'
__C.train.lr = 0.001

__C.test = {}
__C.test.model = "../cpts/baseline/model/final_model.pth"

__C.loss = {}
os.environ['CUDA_VISIBLE_DEVICES'] = __C.train.gpu
weight = torch.tensor([0.05, 0.1, 1]).cuda()
__C.loss.losses = [Losses.DiceLoss(), torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")]

__C.logger = {}
__C.logger.title = []
__C.logger.save_path = "/home1/zhaobt/PycharmProjects/tumor_cls/models/fine_cls/baseline/"
__C.logger.save_name = "train_log_fine_cls.csv"
__C.logger.file_name = []

# __C.infer = {}
# __C.infer.data_log="/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_summary_v1.csv"
# __C.infer.save_suffix = "_pred_mask_baseling.nii.gz"