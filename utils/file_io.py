#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   file_io.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/27 13:48   Bot Zhao      1.0         None
"""

# import lib
import SimpleITK as sitk
import os
import numpy as np
import nibabel
import sys
import json
sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import misc
from nets import Losses
from utils import data_generator
from nets import unet
from utils import img_utils
from utils import logger
from utils import model_io


def save_nii_array(array, save_dir, temp):
    """
    save a 3d array ti nii file.
    :param array:
    :param save_dir:
    :param temp:
    :return:
    """
    # array = array.transpose((2, 1, 0))
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(temp.GetSpacing())
    image.SetOrigin(temp.GetOrigin())
    image.SetDirection(temp.GetDirection())
    sitk.WriteImage(image, save_dir)

# sitk.sitkU
def read_array_nii(input_dir):
    image = sitk.ReadImage(input_dir)
    array = sitk.GetArrayFromImage(image)
    # array = array.transpose((2, 1, 0))
    return image, array


def dicom2nii(rootdir, cur_name, new_name):
    if not os.listdir(rootdir):
        return
    if not os.path.isdir(os.path.join(rootdir, os.listdir(rootdir)[0])):
        outdir = rootdir.replace(cur_name, new_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        os.system("D: && cd D:\jupyter\cest_pipeline\external tools && dcm2niix -f " +
                  "%f_%p_%t_%s" + ' -i n -l y -p y -x n -v 0 -z y -o ' + outdir
                  + " " + rootdir)
    else:
        for i in os.listdir(rootdir):
            print(i)
            dicom2nii(os.path.join(rootdir, i), cur_name, new_name)


def load_module_from_disk(file):
    from importlib.machinery import SourceFileLoader
    func = SourceFileLoader('module.name', file).load_module()
    return func


def save_nii_fromH5(img_arr, meta_info):
    import pdb
    ref_path = meta_info["img_path"]
    ref_path = ref_path.replace("\\\\", "/")
    true_ref_path = ref_path.replace("D:/dataset/tumor_classification","/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
    mask_path = true_ref_path.replace(".nii.gz", "_pred_mask_with_tumor.nii.gz")
    # print(ref_path)
    # print(true_ref_path)
    img = sitk.ReadImage(true_ref_path)
    # pad and resize
    x, y, z = img.GetSize()
    if x <= 512 and y <= 512:
        minx, miny = (512-x)//2, (512-y)//2
        maxx, maxy = ((512-x)//2)+x, ((512-y)//2)+y
        # pdb.set_trace()
        img_arr = img_arr[:, miny:maxy, minx:maxx]
        save_nii_array(img_arr, mask_path, img)
    else:
        img_arr = sitk.GetImageFromArray(img_arr)
        temp_mask = img_utils.resize_image_size(img_arr, (x, y, z))
        img_arr = sitk.GetArrayFromImage(temp_mask)
        save_nii_array(img_arr, mask_path, img)
    print(mask_path)
    root_dir, name = os.path.split(true_ref_path)
    # return mask_path, os.path.join(root_dir, "mask_"+name)


def get_roi_dataset(img_dir, mask_dir):
    try:
        image, img_arr = read_array_nii(img_dir)
        mask, mask_arr = read_array_nii(mask_dir)
        mask_arr = img_utils.rm_small_cc(mask_arr.astype("int32"), rate=0.5)
        st, ed = img_utils.get_bbox(mask_arr)
        img_roi = img_utils.crop_img(img_arr, st, ed)
        save_nii_array(img_roi, img_dir.replace('.nii.gz', "_roi.nii.gz"), image)
        return img_roi.shape, image.GetSize()
    except:
        return [0,0,0], [0,0,0]


def get_file_list(data_dir):
    import pandas as pd
    data = pd.read_csv(data_dir)
    pathes = []
    infos = []
    for i in data.iterrows():
        if i[1]["T1CE_num"] >= 1:
            ref_path = i[1]["T1CE"][2:-2]
            ref_path = ref_path.replace("\\\\", "/")
            true_ref_path = ref_path.replace("D:/dataset/tumor_classification",
                                             "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
            if not os.path.isfile(true_ref_path):
                print("There is a file bug!!!")
            pathes.append(true_ref_path)
            # import pdb
            # pdb.set_trace()
            info = json.loads(i[1]["T1CE_info"])
            infos.append(info)
        # if i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] != "0":
        #     # ref_path = i[1]["T1CE"][2:-2]
        #     ref_path = i[1]["labeled_T1CE"]
        #     ref_path = ref_path.replace("\\\\", "/")
        #     true_ref_path = ref_path.replace("D:/dataset/tumor_classification",
        #                                      "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
        #     if not os.path.isfile(true_ref_path):
        #         print("There is a file bug!!!")
        #     pathes.append(true_ref_path)
    # import pdb
    # pdb.set_trace()
    return pathes, infos


if __name__ == '__main__':
    # root_die = r"D:\dataset\tumors_renamed_data\classified_data\general_hospital\pilocytic_astrocytoma"
    # dicom2nii(root_die, "tumors_renamed_data", "tumor_classification")
    import tqdm

    data_dir = r"/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_summary_v1.csv"
    pathes, infos = get_file_list(data_dir)
    title = ["file_name", "size_x", "size_y", "size_z", "spacing_x", "spacing_y", "spacing_z"]
    # title = ["file_name", "spacing_x", "spacing_y", "spacing_z"]
    log = logger.Logger_csv(title, data_dir.replace('dataset1_summary_v1.csv', ''), "T1CE_spacing_info_temp.csv")
    print("running")
    # print(pathes)
    # img_dir = r"D:\dataset\tumor_classification\temp\T1+_T1_MPRAGE_TRA_iso1.0_20130923131342_2.nii.gz"
    # mask_dir = img_dir.replace(".nii.gz", "_pred_mask_baseling.nii.gz")
    for idx, img_dir in tqdm.tqdm(enumerate(pathes[145:146]), ncols=10):
        # spacing = infos[idx]['img_spacing']
        # log.update({"file_name": img_dir,"spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2]})
        #
        import pdb
        pdb.set_trace()

        print(img_dir)
        mask_dir = img_dir.replace(".nii.gz", "_pred_mask_baseling.nii.gz")
        try:
            [z, x, y], [sx, sy, sz] = get_roi_dataset(img_dir, mask_dir)
            log.update({"file_name": img_dir, "size_x": x, "size_y": y, "size_z": z,
                        "spacing_x": sx, "spacing_y": sy, "spacing_z": sz})
        except:
            log.update({"file_name": img_dir, "size_x": 0, "size_y": 0, "size_z": 0})