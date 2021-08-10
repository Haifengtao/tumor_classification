#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/14 16:25   Bot Zhao      1.0         None
"""

# import lib
import time
# from progress.bar import Bar as Bar
import torch
import numpy as np
import os
import h5py
import sys
from torch.utils.data import DataLoader, Dataset
import json
import pdb

sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import misc
from nets import Losses
from utils import data_generator
from nets import unet, resnet
from utils import file_io
from utils import logger
from utils import model_io


def test(cfg):
    if cfg.net.name == "Unet":
        model = unet.Unet(cfg.net.inchannels, cfg.net.classes, drop_rate=0.2, filters=cfg.net.enc_nf)
    else:
        raise Exception("We have not implemented this model %s".format(cfg.net.name))

    checkpoint = torch.load(cfg.test.model)
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        if torch.cuda.is_available():
            use_cuda = True
            model = model.cuda()
        else:
            use_cuda = False
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu
        data_dir_labeled_test = cfg.data.test_txt
        labeled_f = h5py.File(data_dir_labeled_test, 'r')
        meta_info = labeled_f['meta_info']
        meta_info = json.loads(meta_info.value)
        test_dataset = data_generator.generator_2D_val(labeled_f["img"], labeled_f["mask"])
        test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model.eval()
        dices = [logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter()]
        print("==> There are {} cases in test dataset.".format(len(list(meta_info.keys()))))
        temp_size = meta_info[list(meta_info.keys())[0]]["raw_size"]
        temp_mask_arr = np.zeros((temp_size[2], 512, 512))
        temp_gold_arr = np.zeros((temp_size[2], 512, 512))
        temp_zid = 0
        temp_uid = 0
        for idx, data in enumerate(test_dataset):
            img, mask, _ = data
            if use_cuda:
                img = img.cuda()
                mask = mask.cuda()
            y = model(img)
            sf = torch.nn.Softmax(dim=1)
            # pdb.set_trace()
            _, pred = torch.max(sf(y).cpu().detach(), dim=1)


            # #
            if temp_zid == temp_size[2]-1:
                temp_zid = 0
                file_io.save_nii_fromH5(temp_mask_arr, meta_info[list(meta_info.keys())[temp_uid]])
                dice_values = Losses.labels_dice(temp_mask_arr, temp_gold_arr, 3)
                for i in range(len(dices)):
                    dices[i].update(dice_values[i])
                print("test dice background:{}, brain stem:{}, tumor:{}".format(dice_values[0], dice_values[1],
                                                                                dice_values[2]))

                temp_uid += 1
                try:
                    temp_size = meta_info[list(meta_info.keys())[temp_uid]]["raw_size"]
                    temp_mask_arr = np.zeros((temp_size[2], 512, 512))
                    temp_gold_arr = np.zeros((temp_size[2], 512, 512))
                    temp_mask_arr[temp_zid, :, :] = pred.squeeze().numpy()
                    temp_gold_arr[temp_zid, :, :] = mask.cpu().detach().squeeze().numpy()
                except IndexError:
                    continue
            else:
                temp_mask_arr[temp_zid, :, :] = pred.squeeze().numpy()
                temp_gold_arr[temp_zid, :, :] = mask.cpu().detach().squeeze().numpy()
                temp_zid += 1

        print("test dice background:{}, brain stem:{}, tumor:{}".format(dices[0].avg, dices[1].avg, dices[2].avg))
        labeled_f.close()
    # return dice.avg


def cal_res_ana(pred, y_true):
    """

    :param pred:
    :param y_true:
    :return:  acc, precision, recall
    """
    tp = 0
    p_n = 0
    fn = 0
    tn = 0
    for p, t in zip(pred, y_true):
        if p == 1:
            p_n += 1

        if p != 1 and t == 1:
            fn += 1

        if p == t and p == 1:
            tp += 1

        if p == t and p == 0:
            tn += 1

    return (tp+tn)/len(pred), tp/p_n, tp/(tp+fn)


def test_cls(cfg):
    if cfg.net.name == "Unet":
        model = unet.Unet(cfg.net.inchannels, cfg.net.classes, drop_rate=0.2, filters=cfg.net.enc_nf)
    elif cfg.net.name == "resnet200_cls":
        model = resnet.resnet200(sample_input_W=96,
                                 sample_input_H=112,
                                 sample_input_D=64,
                                 num_seg_classes=2,
                                 droprate=cfg.net.drop_rate)
    else:
        raise Exception("We have not implemented this model %s".format(cfg.net.name))
    # import pdb
    # pdb.set_trace()
    checkpoint = torch.load(cfg.test.model)
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        if torch.cuda.is_available():
            use_cuda = True
            model = model.cuda()
        else:
            use_cuda = False
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu
        test_dataset = DataLoader(data_generator.Cls_base(cfg.data.test_txt), batch_size=1, shuffle=False)
        model.eval()
        dices = [logger.AverageMeter(), logger.AverageMeter(), logger.AverageMeter()]
        print("==> There are {} cases in test dataset.".format(len(test_dataset)))
        y_true = []
        pred = []
        res = []
        for idx, data in enumerate(test_dataset):
            img, mask = data
            if use_cuda:
                img = img.cuda()
                mask = mask.cuda()
            y = model(img)
            soft = torch.nn.Softmax()
            y_p = soft(y)
            # pdb.set_trace()
            res.append(y_p.cpu().detach().numpy())
            y_true.append(mask.item())
            pred_y = torch.max(y, 1)[1].cpu().numpy()[0]
            pred.append(pred_y)
        with open(cfg.test.save_dir, "w") as f:
            json.dump({"name": cfg.net.name, "res": [i.tolist() for i in res], "y_true": y_true}, f)
        acc, precision, recall = cal_res_ana(pred, y_true)
        print("test acc: {}, precision: {}, recall: {}".format(acc, precision, recall))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Brain Training')
    parser.add_argument('-i', '--config',  default="../models/baseline/config.py",
                        help='model config (default: Unet)')
    arguments = parser.parse_args()
    config = file_io.load_module_from_disk(arguments.config)
    cfg = config.cfg
    test_cls(cfg)