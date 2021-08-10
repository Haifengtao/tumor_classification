#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model_io.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/14 16:31   Bot Zhao      1.0         None
"""

# import lib
import os
import torch


# TODO remove dependency to args
def reload_ckpt(path, model, optimizer, scheduler, use_cuda):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if use_cuda:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 因为optimizer加载参数时,tensor默认在CPU上
            # 故需将所有的tensor都放到cuda,
            # 否则: 在optimizer.step()处报错：
            # RuntimeError: expected device cpu but got device cuda:0
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            model.cuda()
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format("started with:", checkpoint['epoch']))
        return start_epoch
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(path))


def load_retrained_model(pth_dir, model):
    net_dict = model.state_dict()
    print('loading pretrained model {}'.format(pth_dir))
    pretrain = torch.load(pth_dir)
    print(list(pretrain['state_dict'].keys())[:10])
    print(list(net_dict.keys())[:10])
    pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict.keys()}
    print("updated: {}".format(pretrain_dict.keys()))
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    return model