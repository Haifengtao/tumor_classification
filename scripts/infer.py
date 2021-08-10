#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   infer.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/14 16:25   Bot Zhao      1.0         None
"""

# import lib
import time
import torch
import numpy as np
import os
import h5py
import sys
from torch.utils.data import DataLoader, Dataset
import json
import pdb
from tqdm import tqdm
sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import misc
from nets import Losses
from utils import data_generator
from nets import unet
from utils import file_io
from utils import logger
from utils import model_io
from utils import img_utils


def infer(cfg, pathes):
    if cfg.net.name == "Unet":
        model = unet.Unet(cfg.net.inchannels, cfg.net.classes, drop_rate=0.2, filters=cfg.net.enc_nf)
    else:
        raise Exception("We have not implemented this model %s".format(cfg.net.name))

    checkpoint = torch.load(cfg.test.model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if torch.cuda.is_available():
        use_cuda = True
        model = model.cuda()
    else:
        use_cuda = False
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu
    print(pathes)
    with torch.no_grad():
        for idx, path in enumerate(pathes):
            print("==> Processing {}/{} cases:{}.".format(idx, len(pathes), path))
            img, img_arr= file_io.read_array_nii(path)
            if len(img_arr.shape) > 3:
                print("There is a error on data dim of {}!".format(path))
                continue
            proc_img_arr = img_utils.crop_resize(img, img_arr)
            proc_img_arr = proc_img_arr[:, np.newaxis, ...]
            input_img = torch.from_numpy(proc_img_arr.astype(np.float32))
            pred_mask = np.zeros([img_arr.shape[0], 512, 512])
            for z in range(img_arr.shape[0]):
                img_slice = input_img[[z], :, :, :]
                if use_cuda:
                    img_slice = img_slice.cuda()
                y = model(img_slice)
                sf = torch.nn.Softmax(dim=1)
                # pdb.set_trace()
                _, pred = torch.max(sf(y).cpu().detach(), dim=1)
                pred_mask[z, :, :] = pred.squeeze().numpy()
            pred_mask = img_utils.de_crop_resize(pred_mask, img)
            file_io.save_nii_array(pred_mask, path.replace(".nii.gz", cfg.infer.save_suffix), img)


def get_infer_list(data_dir):
    import pandas as pd
    data = pd.read_csv(data_dir)
    pathes = []
    with open("../data/all_dataset2_t1ce.txt", "w") as f:
        for i in data.iterrows():
            if i[1]["T1CE_num"] == 1:
                ref_path = i[1]["T1CE"][2:-2]
                ref_path = ref_path.replace("\\\\", "/")
                true_ref_path = ref_path.replace("D:/dataset/tumor_classification",
                                                 "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
                if not os.path.isfile(true_ref_path):
                    print("There is a file bug!!!")
                pathes.append(true_ref_path)
            if i[1]["T1CE_num"] == 2:
                ref_path_1, ref_path_2 = i[1]["T1CE"].split(",")[0][2:-1], i[1]["T1CE"].split(",")[1][2:-1]
                ref_path_1 = ref_path_1.replace("\\\\", "/")
                true_ref_path1 = ref_path_1.replace("D:/dataset/tumor_classification",
                                                 "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
                if not os.path.isfile(true_ref_path1):
                    print("There is a file bug!!!")
                else:
                    pathes.append(true_ref_path1)

                ref_path_2 = ref_path_2.replace("\\\\", "/")
                true_ref_path2 = ref_path_2.replace("D:/dataset/tumor_classification",
                                                 "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
                if not os.path.isfile(true_ref_path2):
                    print("There is a file bug!!!")
                else:
                    pathes.append(true_ref_path2)
        for path in pathes:
            f.write(path+"\n")
    return pathes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Brain Training')
    parser.add_argument('-i', '--config',  default="../models/baseline/config.py",
                        help='model config (default: Unet)')
    arguments = parser.parse_args()
    config = file_io.load_module_from_disk(arguments.config)
    cfg = config.cfg
    pathes = get_infer_list(cfg.infer.data_log)
    print(len(pathes))
    # infer(cfg, pathes=pathes)