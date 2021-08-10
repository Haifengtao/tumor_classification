#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/2 19:14   Bot Zhao      1.0         None
"""

# import lib
from easydict import EasyDict as edict
import torch
import numpy as np
from nets import Losses
import os

__C = edict()
cfg = __C

__C.general = {}
__C.general.log_path = "../cpts/cls/mixup/model"
__C.general.model_path = "../cpts/cls/mixup/model"


__C.data = {}
__C.data.train_txt = r"../data/train_cls_dataset_1.txt"
__C.data.val_txt = r"../data/val_cls_dataset_1.txt"
__C.data.test_txt = r"../data/test_cls_dataset_1.txt"
__C.data.resize_shape = [256, 256]

__C.net = {}
__C.net.name = "resnet200_cls"
__C.net.pretrain = '../external/pretrain/resnet_200.pth'
__C.net.inchannels = 1
__C.net.classes = 2
__C.net.enc_nf = None
__C.net.with_tumor = None
__C.net.drop_rate = 0.5


__C.train = {}
__C.train.load_model = None  # "../cpts/cls/mixup/model/155_model.pth"
__C.train.start_epoch = 0
__C.train.epochs = 5000
__C.train.batch_size = 6
__C.train.train_iteres = 100
__C.train.save_epoch = 50
__C.train.learning_rate = 0.001
__C.train.warmRestart = 1
__C.train.mixup = True
__C.train.alpha = 0.5

__C.train.gpu = '1'
__C.train.lr = 0.001


__C.loss = {}
os.environ['CUDA_VISIBLE_DEVICES'] = __C.train.gpu
# weight = torch.tensor([0.3, 0.7]).cuda()
__C.loss.losses = [torch.nn.CrossEntropyLoss(reduction="mean")]

__C.logger = {}
__C.logger.title = []
__C.logger.save_path = "/home1/zhaobt/PycharmProjects/tumor_cls/models/cls/mixup/"
__C.logger.save_name = "train_log_cls.csv"
__C.logger.file_name = []

# __C.infer = {}
# __C.infer.data_log="/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_summary_v1.csv"
# __C.infer.save_suffix = "_pred_mask_baseling.nii.gz"

__C.test = {}
__C.test.model = "../cpts/cls/dropout/model/best_model.pth"
__C.test.save_dir = "../cpts/cls/mixup/res.json"