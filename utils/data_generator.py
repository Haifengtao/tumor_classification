#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_generator.py
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/5 16:44   Bot Zhao      1.0         None
"""

# import lib
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torchvision import transforms
import h5py
from utils import file_io
import os
# import nibabel
# from scipy import ndimage


class data_generator_2D(Dataset):
    def __init__(self, labeled_img, labeled_mask, unlabeled):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.unlabeled = unlabeled

        pass

    def __len__(self):
        return {}
        pass

    def __getitem__(self, index):
        pass


def generator_2D_labeled(labeled_img, labeled_mask,
                         batch_size=1, with_tumor=False, resize=None, is_neg_sampling=False):
    labeled_number = labeled_img.shape[0]
    arr = np.arange(labeled_number)
    # print(arr)
    # np.random.shuffle(arr)
    # print(arr[:batch_size])
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        X = labeled_img[labeled_indices, ...]
        X = X[:, np.newaxis, ...]
        mask = labeled_mask[labeled_indices, ...]
        if not with_tumor:
            mask[mask != 0] = 1
        label = np.sum(mask, axis=(1, 2))
        label[label != 0] = 1
        yield torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
              torch.from_numpy(label.astype(np.uint8)).long()


def generator_2D_labeled_tumor(labeled_img, labeled_mask,
                               batch_size=1, resize=None, is_neg_sampling=False):
    labeled_number = labeled_img.shape[0]
    arr = np.arange(labeled_number)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        X = labeled_img[labeled_indices, ...]
        X = X[:, np.newaxis, ...]
        mask = labeled_mask[labeled_indices, ...]
        mask[mask != 0] = 1
        label = np.sum(mask, axis=(1, 2))
        label[label != 0] = 1
        yield torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
              torch.from_numpy(label.astype(np.uint8)).long()


def generator_2D_labeled_tumor_roi(img_csv,
                                   batch_size=1, resize=None, is_neg_sampling=False):
    img_pathes = file_io.get_file_list(img_csv)
    arr = np.arange(len(img_pathes))
    while True:
        # 0. 随机获取数据索引
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))

        # 1. 读取数据


def CLS_base(root_dir, batch_size):
    with open(root_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        img_list = [line.strip().split("|")[0] for line in strs]
        labels = [int(line.strip().split('|')[1]) for line in strs]
    arr = np.arange(len(img_list)-1)
    # temp = np.eye(max(labels)+1)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        imgs = []
        label_out = []
        for i in labeled_indices:
            _, img_arr = file_io.read_array_nii(img_list[i])  # We have transposed the data from WHD format to DHW
            img_arr = img_arr[np.newaxis, ...]
            imgs.append(img_arr)
            # label_out.append(temp[labels[i]])
            label_out.append(labels[i])
        img_arrs = np.stack(imgs, axis=0)
        label_out = np.stack(label_out, axis=0)
        yield torch.from_numpy(img_arrs.astype(np.float32)), \
              torch.from_numpy(label_out.astype(np.uint8)).long()

class Cls_base(Dataset):
    def __init__(self, root_dir):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            self.img_list = [line.strip().split("|")[0] for line in strs]
            self.labels = [int(line.strip().split('|')[1]) for line in strs]

        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels

        img_name = self.img_list[idx]
        label_n = np.array(self.labels[idx])
        assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name)  # We have transposed the data from WHD format to DHW
        # print(img_name)
        assert img_arr is not None
        img_arr = img_arr[np.newaxis,...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(label_n.astype(np.uint8)).long()



class generator_3D_roi(Dataset):
    def __init__(self, img_csv,  with_tumor=False):
        self.img_pathes = file_io.get_file_list(img_csv)

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, item):
        img_dir = self.img_pathes[item].replace(".nii.gz",  "_roi.nii.gz")
        mask_dir = self.img_pathes[item].replace(".nii.gz",  "_roi.nii.gz")
        img, img_arr = file_io.read_array_nii(img_dir)

        X = self.img[item, ...]
        X = X[np.newaxis, ...]
        mask = self.mask[item, ...]
        if not self.with_tumor:
            mask[mask != 0] = 1
        label = np.array([1]) if np.sum(mask) != 0 else np.array([0])
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


class generator_2D_val(Dataset):
    def __init__(self, labeled_img, labeled_mask, with_tumor=False):
        self.img = labeled_img
        self.mask = labeled_mask
        self.with_tumor = with_tumor

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, item):
        X = self.img[item, ...]
        X = X[np.newaxis, ...]
        mask = self.mask[item, ...]
        # if not self.with_tumor:
        #     mask[mask != 0] = 1
        label = np.array([1]) if np.sum(mask) != 0 else np.array([0])
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class RandomRotate(object):
    def __call__(self, x):
        c, h, w = x.shape


if __name__ == '__main__':
    data_dir_un = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_unlabeled.h5"
    unlabeled_f = h5py.File(data_dir_un, 'r')
    unlabeled_dataset = unlabeled_f['img']

    data_dir_labeled_train = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_labeled_train.h5"
    labeled_f = h5py.File(data_dir_labeled_train, 'r')
    train_labeled_img_dataset = labeled_f['img']
    train_labeled_mask_dataset = labeled_f['mask']

    transform_train = transforms.Compose([
        RandomPadandCrop(32),
        RandomFlip(),
        ToTensor(),
    ])
    a = labeled_f['img'][[1, 2, 3, ], ...]
    dataset = generator_2D_labeled(labeled_f['img'], labeled_f['mask'], batch_size=10)

    x, mask, label = next(dataset)
    print(x.size())
    print(mask.size())
    print(label.size())

    val_dataset = generator_2D_val(labeled_f['img'], labeled_f['mask'])
    val_dataset = DataLoader(val_dataset, batch_size=2)
    for i, data in enumerate(val_dataset):
        img, mask, _ = data
        print(img.size())
        print(mask.size())
        print(_.size())
        break
