#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_cls.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/2 19:16   Bot Zhao      1.0         None
"""

# import lib
import time
# from progress.bar import Bar as Bar
import torch
from torch import optim, nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, Dataset
import pdb

sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import misc
from nets import Losses
from utils import data_generator
from nets import unet, resnet
from utils import file_io
from utils import logger
from utils import model_io


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    """
    xy = [X, U]
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def train_mixmatch(cfg, labeled_trainloader, unlabeled_trainloader,
          model, optimizer, ema_optimizer, criterion, epoch, use_cuda):
    batch_time = misc.AverageMeter()
    data_time = misc.AverageMeter()
    losses = misc.AverageMeter()
    losses_x = misc.AverageMeter()
    losses_u = misc.AverageMeter()
    ws = misc.AverageMeter()
    end = time.time()

    # bar = Bar('Training', max=cfg.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(cfg.train_iteration):
        # 1. loading data
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples $q_{b}$:
            # # there are two augment methods.
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / cfg.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup:
        # NOTE: (inputs_u, inputs_u2) AND (targets_u, targets_u)
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        # get
        l = np.random.beta(cfg.alpha, cfg.alpha)
        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))
        # a is raw, b is shuffled
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        mixed_input = list(torch.split(mixed_input, batch_size))  # split mixed_input to [X, U]

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        # [batch_X, batch_U1, batch_U2,...]
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / cfg.train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
            batch=batch_idx + 1,
            size=cfg.train_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)


def train_baseline(cfg, epoch, model, optimizer, labeled_train_loader, use_cuda):
    global loss_value, losses
    model.train()
    timer, ec = logger.AverageMeter(), logger.AverageMeter()
    labeled_train_loader = iter(labeled_train_loader)
    for i in range(cfg.train.train_iteres):
        st = time.time()
        x, label = next(labeled_train_loader)
        # print(use_cuda)
        # pdb.set_trace()

        if cfg.train.mixup:
            lam = np.random.beta(cfg.train.alpha, cfg.train.alpha)
            x2, label2 = next(labeled_train_loader)
            x = lam*x + (1-lam)*x2
            if use_cuda:
                x = x.cuda()
                label, label2 = label.cuda(), label2.cuda()
            pred = model(x)
            criterion = nn.CrossEntropyLoss().cuda()
            loss_value = lam * criterion(pred, label) + (1 - lam) * criterion(pred, label2)
            losses = [loss_value]
        else:
            if use_cuda:
                x = x.cuda()
                label = label.cuda()
            # pdb.set_trace()
            y = model(x)
            for idx, loss_func in enumerate(cfg.loss.losses):
                if idx == 0:
                    loss_value = loss_func(y, label)
                    losses = [loss_value]
                else:
                    temp_loss = loss_func(y, label)
                    loss_value += temp_loss
                    losses.append(temp_loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_value.backward()
        # pdb.set_trace()
        optimizer.step()
        # pdb.set_trace()
        timer.update(time.time()-st)
        ec.update(losses[0].cpu().detach())
        msg = "epoch: {},  iter:{} / all:{}, etc:{:.2f}, ec_loss: {:.4f}" .\
            format(epoch, i, cfg.train.train_iteres, time.time()-st, ec.avg)
        print(msg)
    return ec.avg


def acc(pred, y_true):
    res = 0
    for p, t in zip(pred, y_true):
        if p == t:
            res += 1
    return res/len(pred)


def val_model(model, val_dataset, use_cuda):
    # implemented
    model.eval()
    ec = logger.AverageMeter()
    pred = []
    y_true = []
    for idx, data in enumerate(val_dataset):
        img, mask = data
        if use_cuda:
            img = img.cuda()
            mask = mask.cuda()
        y = model(img)
        # pdb.set_trace()
        loss = nn.CrossEntropyLoss()
        loss_value = loss(y, mask)
        ec.update(loss_value)
        # y_true.append(torch.max(mask, 1)[1].cpu().numpy()[0])
        y_true.append(mask.item())
        pred_y = torch.max(y, 1)[1].cpu().numpy()[0]
        pred.append(pred_y)

    print("val loss {}, acc: {}".format(ec.avg, acc(pred, y_true)))
    return ec.avg, acc(pred, y_true)


def main(cfg):
    if cfg.net.name == "Unet":
        model = unet.Unet(cfg.net.inchannels, cfg.net.classes, drop_rate=0.2, filters=cfg.net.enc_nf)
    elif cfg.net.name == "resnet200_cls":
        model = resnet.resnet200(sample_input_W=96,
                                 sample_input_H=112,
                                 sample_input_D=64,
                                 num_seg_classes=2,
                                 droprate=cfg.net.drop_rate)
        if cfg.net.pretrain:
            model_io.load_retrained_model(cfg.net.pretrain, model)
    else:
        raise Exception("We have not implemented this model %s".format(cfg.net.name))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu

    # train_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.train_txt), batch_size=cfg.train.batch_size,
    #                                shuffle=True)
    # val_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.val_txt), batch_size=1, shuffle=False)

    train_cls_dataset = data_generator.CLS_base(cfg.data.train_txt, cfg.train.batch_size)
    val_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.val_txt), batch_size=1, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0, )
    if torch.cuda.is_available():
        use_cuda = True
        model = model.cuda()
    else:
        use_cuda = False

    if cfg.train.warmRestart:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cur_iter: (1 + cur_iter) / (cfg.train.train_iteres * cfg.train.warm))

    if cfg.train.load_model is not None:
        start_epoch = model_io.reload_ckpt(cfg.train.load_model, model, optimizer, scheduler=scheduler, use_cuda=use_cuda)
        cfg.train.start_epoch = start_epoch

    value_save = ["epoch", "learning rate",  "train_ec", "val_ec", "val_acc"]
    log = logger.Logger_csv(value_save, cfg.logger.save_path, cfg.logger.save_name)

    temp = np.inf
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs+1):
        train_ec = train_baseline(cfg, epoch, model, optimizer,  train_cls_dataset, use_cuda)
        with torch.no_grad():
            val_dice, acc_value = val_model(model, val_cls_dataset, use_cuda)
        log.update({"epoch": epoch, "learning rate": optimizer.param_groups[0]['lr'],
                    "train_ec": train_ec.cpu().detach().numpy(),
                    "val_ec": val_dice.cpu().detach().numpy(),
                    "val_acc": acc_value})
        scheduler.step()
        print("learning rate", optimizer.param_groups[0]['lr'])
        if epoch % 5 == 0:
            if not os.path.isdir(cfg.general.model_path):
                os.makedirs(cfg.general.model_path)
            torch.save(dict(epoch=epoch,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict()),
                       f=os.path.join(cfg.general.model_path, str(epoch)+"_model.pth"))

        if val_dice < temp:
            torch.save(dict(epoch=epoch,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict()),
                       f=os.path.join(cfg.general.model_path, "best_model.pth"))
            temp = val_dice

    torch.save(dict(epoch=cfg.train.epochs+1,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict()),
               f=os.path.join(cfg.general.model_path, "final_model.pth"))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Brain Training')
    parser.add_argument('-i', '--config',  default=None,
                        help='model config (default: Unet)')
    arguments = parser.parse_args()
    config = file_io.load_module_from_disk(arguments.config)
    cfg = config.cfg
    print(cfg.data.train_txt)
    main(cfg)