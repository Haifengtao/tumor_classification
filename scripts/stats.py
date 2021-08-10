#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   stats.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/28 13:45   Bot Zhao      1.0         None
"""

# import lib
import pandas as pd
import os
from utils import logger
import glob
from utils import file_io
import numpy as np
from tqdm import tqdm
import json
import shutil


def get_img_info(path):
    if not path or not os.path.isfile(path[0]):
        return None
    img, array = file_io.read_array_nii(path[0])
    return {
        "img_path": path,
        "img_size": array.shape,
        "img_interval": [np.min(array).tolist(), np.max(array).tolist()],
        "img_spacing": img.GetSpacing(),
        "img_origin": img.GetOrigin()
    }


def check_same(pathes):
    out_path = []
    temp_array = []
    for i in pathes:
        img, array = file_io.read_array_nii(i)
        if not temp_array:
            temp_array.append(array)
            out_path.append(i)
        else:
            same = False
            for arr in temp_array:
                if arr.shape == array.shape and (array == arr).all():
                    same = True
                    break
            if same:
                continue
            else:
                out_path.append(i)
    return out_path


def get_file_list(dirs, save_name='data_summary_unclassified.csv'):
    maps = {"uid": [], "tumor type": [], "name": [],
            "T1_num": [], "T2F_num": [], "T1CE_num": [], "DW_num": [], "ADC_num": [], "T2_num": [],
            "T1": [], "T2F": [], "T1CE": [], "DW": [], "ADC": [], "T2": [],
            "T1_info": [], "T2F_info": [], "T1CE_info": [], "DW_info": [], "ADC_info": [], "T2_info": []}
    saver = logger.Logger_csv(maps.keys(), '../data/', save_name)

    uid = 1
    for i in os.listdir(dirs):
        tumor_type = i
        temp_dir = os.path.join(dirs, i)
        for name in tqdm(os.listdir(temp_dir)):
            T1 = glob.glob(os.path.join(temp_dir, name) + "/T1/" + "T1*.nii.gz")
            T2F = glob.glob(os.path.join(temp_dir, name) + "/T2F/" + "T2F*.nii.gz")
            T1CE = glob.glob(os.path.join(temp_dir, name) + "/T1+/" + "T1+*.nii.gz")
            DW = glob.glob(os.path.join(temp_dir, name) + "/DW*/" + "*DW*.nii.gz")
            ADC = glob.glob(os.path.join(temp_dir, name) + "/AD*/" + "*AD*.nii.gz")
            T2 = glob.glob(os.path.join(temp_dir, name) + "/T2/" + "*T2*.nii.gz")

            T1 += glob.glob(os.path.join(temp_dir, name) + "/*T1.nii.gz")
            T2F += glob.glob(os.path.join(temp_dir, name) + "/*T2F.nii.gz")
            T1CE += glob.glob(os.path.join(temp_dir, name) + "/*T1+.nii.gz")
            DW += glob.glob(os.path.join(temp_dir, name) + "/*DWI.nii.gz")
            ADC += glob.glob(os.path.join(temp_dir, name) + "/*ADC.nii.gz")
            T2 += glob.glob(os.path.join(temp_dir, name) + "/*T2.nii.gz")

            T1 = check_same(T1)
            T2 = check_same(T2)
            T2F = check_same(T2F)
            T1CE = check_same(T1CE)
            DW = check_same(DW)
            ADC = check_same(ADC)
            temp_map = {"uid": uid, "tumor type": tumor_type, "name": name,
                        "T1_num": len(T1), "T2F_num": len(T2F), "T1CE_num": len(T1CE), "DW_num": len(DW),
                        "ADC_num": len(ADC), "T2_num": len(T2),
                        "T1": T1[0], "T2F": T2F[0], "T1CE":  T1CE[0], "DW": DW[0], "ADC":  ADC[0], "T2": T2[0],
                        "T1_info": json.dumps(get_img_info(T1)), "T2F_info": json.dumps(get_img_info(T2F)),
                        "T1CE_info": json.dumps(get_img_info(T1CE)),
                        "DW_info": json.dumps(get_img_info(DW)), "ADC_info": json.dumps(get_img_info(ADC)),
                        "T2_info": json.dumps(get_img_info(T2))}
            # return
            uid += 1
            saver.update(temp_map)


def find_seq(root_dir):
    """
    :param root_dir:
    :return:
    """
    saver = logger.Logger_csv(["sequences", "number"], '../data/', 'data_seqs.csv')
    files = glob.glob(root_dir+"/*/*/*.json")
    print(len(files))
    files += glob.glob(root_dir + "/*/*/*/*.json")
    print(len(files))
    modu_seq_maps = {}
    unkown = []
    for file in files:
        try:
            with open(file, "r") as f:
                seq = json.load(f)["SeriesDescription"]
            if seq not in modu_seq_maps:
                modu_seq_maps[seq] = 1
            else:
                modu_seq_maps[seq] += 1
        except Exception:
            unkown.append(file)
    for i in modu_seq_maps:
        saver.update({"sequences": i, "number": modu_seq_maps[i]})
    print(modu_seq_maps)
    print(unkown)


def stats(path):
    # {'DW', 'adc', 'dw', 'DWI', 'T2F', '+', 'T2SPACE', 't2f', 'T1', 'T1+', 't1+', '2F', 't+', 'ADC', 'T2TSE',
    # 't2space', 'T2S', 'T2BLADE', 'T+', 'T1+SAG', '1', 'AD', 'T2'}
    """
    {'ADC': ['Apparent_Diffusion_Coefficient_(mm2_s)', 'ep2d_diff_orth_p2_ADC', 'ep2d_diff_dwi_TE91_ADC',
             'ep2d_diff_dwi_ADC', 'AX_DWI_ADC'],
     'DW': ['Ax_DWI_Asset', 'ep2d_diff_dwi_TE91', 'ep2d_diff_orth_p2', 'ep2d_diff_dwi', 'AX_DWI'],
     'T1': ['Ax_T1_FLAIR', 't1_flair_tra', 'AX_T1W_FLAIR', 'OAx_T1', '_Ax_T1_FLAIR', 'OAx_T1_FLAIR'],
     'T1+': ['Ax_T1+C', 'Ax_3D_T1BRAVO+C', 't1_flair_tra_+c', 'T1_MPRAGE_TRA_iso1.0', 'AX_T1W_FLAIR_+C', 'AX_3D_T1W+C',
             'AX_T1W_FLAIR+C', 'T1_MPRAGE+C', 'AX_3D_T1W', '_Ax_T1_C+', '_Ax_T1_FLAIR+C', 'OAx_T1_FLAIR+C',
             'Ax_LAVA_T1+C', 'Ax_3D_T1BRAVO_+C'],
     'T2F': ['Ax_T2_FLAIR', 't2_flair_tra', 't2_tirm_tra_dark-fluid', 't2_flair_tra-2DIMRI-2MM', 'AX_T2W_FLAIR',
             '_Ax_FlAIR_8', 't2_flair_tra-2DIMRIS-2MM'],
     'AD': ['Apparent_Diffusion_Coefficient_(mm2_s)', 'ep2d_diff_dwi_TE91_ADC', 'ep2d_diff_orth_p2_ADC'],
     'T2SPACE': ['t2_spc_tra_p2_3D_2mm', 'AX_T2W_BLADE', 't2_spc_tra_p2_3D_2mm_(T2)'], 'T2S': ['t2_spc_tra_p2_3D_2mm'],
     'DWI': ['Ax_DWI_Asset', 'ep2d_diff_orth_p2', '_Ax_DWI', 'ep2d_diff_dwi_TE91', 'AX_DWI', 'OAx_DWI_Asset',
             'ep2d_diff_dwi'], 'T2': ['AX_T2W_82'],
     't1+': ['AX_3D_T1W_+C', 'OAx_T1+C', 'T1_MPRAGE_TRA_iso1.0', 't1_flair_tra_+c'], 'T1+SAG': ['t1_mprage_3D'],
     't+': ['T1_MPRAGE_TRA_iso1.0'], 'T+': ['Ax_T1+C', 'AX_T1W_FLAIR+C', 'T1_MPRAGE_TRA_iso1.0', 't1_flair_tra_+c'],
     't2space': ['t2_spc_tra_p2_3D_2mm'], 'T2BLADE': ['AX_T2W_BLADE'], '2F': ['Ax_T2_FLAIR'], '1': ['t1_flair_tra'],
     '+': ['T1_MPRAGE_TRA_iso1.0'], 'adc': ['ep2d_diff_dwi_TE91_ADC'], 'dw': ['ep2d_diff_dwi_TE91'],
     't2f': ['t2_flair_tra'], 'T2TSE': ['t2_tse_tra']}
    """
    hashmap = set()
    modu_seq_maps = {}
    saver = logger.Logger_csv("", '../data/', 'data_summary_classified_name.csv')
    for m in os.listdir(path):
        temp = os.path.join(path, m)
        for i in os.listdir(temp):
            # saver.update({"path", })
            temp2 = os.path.join(temp, i)
            for k in os.listdir(temp2):
                hashmap.add(k)
                if k == 't2space' or k == 'T2S':
                    os.rename(os.path.join(temp2, k), os.path.join(temp2, "T2SPACE"))
                if k == 't2f' or k == '2f' or k == '2F':
                    os.rename(os.path.join(temp2, k), os.path.join(temp2, "T2F"))
                if k == 'DW' or k == "dw":
                    os.rename(os.path.join(temp2, k), os.path.join(temp2, "DWI"))
                if k == 't+' or k == "+" or k == "T+" or k == "t1+":
                    os.rename(os.path.join(temp2, k), os.path.join(temp2, "T1+"))
                if k == 'AD' or k == "ad" or k == "adc":
                    os.rename(os.path.join(temp2, k), os.path.join(temp2, "ADC"))
                if k == '1':
                    os.rename(os.path.join(temp2, k), os.path.join(temp2, "T1"))
            for k in os.listdir(temp2):
                files = glob.glob(os.path.join(temp2, k) + "/*.json")
                for file in files:
                    print(file)
                    with open(file, "rb") as f:
                        seq = json.load(f)["SeriesDescription"]
                    if k not in modu_seq_maps:
                        modu_seq_maps[k] = [seq]
                    else:
                        if seq not in modu_seq_maps[k]:
                            modu_seq_maps[k].append(seq)
                        else:
                            continue

            print(os.path.join(temp, i), )
    print(hashmap)
    print(modu_seq_maps)


    pass


def get_type(target):
    """
    {'T1+': array(['WIP_MIP_-_s3DI_MC_HR', 'sT1W_3D+C', 'T1W_3D_Nav', 'T1W_3D_Nav+C',
        'OAx_T1Flair+C', 'OSag_T1Flair+C', 'AX_T1+C', 'SG_T1+C',
        'OCor_T1WI_+C', 'AX_T1W_IR+C', 'COR_T1W_IR+C', 'SAG_T1W_IR+C',
        'SAG_T1W_IR', 'AX_T1_+C', 'COR_3D_T1_+C', 'SAG_3D_T1_+C',
        'SAG_3D_T1_FS_+C', 'SG_T1_+C', 'Ax_T1_BRAVO_daohang', 'AX_FLAIR+C',
        'T1W_TSE+C', 'Sag_T1_FLAIR+c', 'c-OAx_T1_+C', 'c-OSag_T1_Shim+C',
        'L-OAx_T1_fse+C', 'L-OSag_T1flair+c', 'FireFLY+C', 'AX_T1WI_+C',
        'SG_T1WI_FS_+C', 'AX_FLAIR_+C', 'Ax_3D_PC_MRV+C',
        'COL:Ax_3D_PC_MRV+C', 'PJN:Ax_3D_PC_MRV+C', 'Ax_BRAVO+C',
        'AX_T2FLAIR+C', 'AX_T2FLAIR_LongTR+C', 'OCor_CUBE_T1_fs_+C',
        'OSag_CUBE_T1_fs_+C', 'CO_T1_+C', 'T1W_3D_Nav_FAST',
        'OCor_CUBE_T1_+C', 'OSag_CUBE_T1_+C', 'COR_T1W_2.0mm+C',
        'SAG_T1W_2.0mm+C', 'OAx_T2Flair+C', 'T-OAx_T1_fse+C',
        'T-OSag_T1_FSE+C_Shim'], dtype=object),
 'T1': array(['sT1W_3D_TFE', 'OAx_T1Flair', 'OSag_T1Flair', 'AX_T1', 'AX_T1W_IR',
        'c-OAx_T1', 'c-OSag_T1', 'L-OAx_T1_fse', 'L-OSag_T1_flair',
        'AX_3D_T1WI', 'AX_T1WI', 'SG_T1WI', 'sT1W_3D_IR_TFE_tra',
        'OCor_CUBE_T1', 'OSag_CUBE_T1', 'COR_T1W_2.0mm', 'SAG_T1W_2.0mm',
        'OCor_T1WI', 'T-OAx_T1_fse', 'T-OSag_T1_FSE'], dtype=object),
 'T2': array(['T2W_TSE_2MM', 'OAx_T2', 'AX_T2W_TSE', 'AX_T2', 'COR_3D_T2',
        'T2W_MVXD', 'c-OSag_T2FSE', 'L-OSag_T2_FRFSE', 'SG_T2WI',
        '3D_Brain_VIEW_T2', 'OCor_CUBE_T2', 'c-OAx_T2FSE',
        'T-OAx_T2_FRFSE', 'T-OSag_T2FSE'], dtype=object),
 'T2F': array(['T2_FLAIR_MVXD', 'T2_FLAIR_LongTR_2MM', 'OAx_T2Flair', 'AX_FLAIR',
        'AX_T2FLAIR_LongTR', 'AX_FLAIR_NEW'], dtype=object),
 'ADC': array(['ADC_(10_-6_mm_s):Apr_21_2019_12-59-32_CST',
        'ADC_(10_-6_mm_s):Apr_11_2019_08-13-19_CST', 'IsoADC',
        'ADC_(10_-6_mm_s):Aug_05_2020_13-32-19_CST',
        'ADC_(10_-6_mm_s):Nov_07_2020_19-22-50_CST',
        'ADC_(10_-6_mm_s):Jan_11_2020_14-57-51_CST',
        'ADC_(10_-6_mm_s):Oct_31_2019_19-23-49_CST',
        'ADC_(10_-6_mm_s):Jun_17_2020_16-54-02_CST',
        'ADC_(10_-6_mm_s):Jul_09_2020_16-49-50_CST',
        'ADC_(10_-6_mm_s):Dec_11_2019_18-27-00_CST',
        'ADC_(10_-6_mm_s):Nov_30_2019_19-06-28_CST',
        'ADC_(10_-6_mm_s):Aug_10_2020_18-18-42_CST',
        'ADC_(10_-6_mm_s):Jun_20_2019_17-24-56_CST',
        'ADC_(10_-6_mm_s):Sep_11_2020_11-22-19_CST',
        'ADC_(10_-6_mm_s):Jun_15_2019_13-25-57_CST',
        'ADC_(10_-6_mm_s):Apr_28_2019_18-07-42_CST',
        'ADC_(10_-6_mm_s):Jan_16_2019_11-25-49_CST',
        'ADC_(10_-6_mm_s):Dec_15_2018_09-31-02_CST',
        'ADC_(10_-6_mm_s):Sep_08_2020_20-24-24_CST',
        'ADC_(10_-6_mm_s):Oct_12_2020_18-07-21_CST',
        'ADC_(10_-6_mm_s):Aug_27_2019_18-29-07_CST',
        'ADC_(10_-6_mm_s):Jul_26_2020_12-37-51_CST'], dtype=object),
 'DWI': array(['OAx_DWI_1000', 'AX_DWI', 'IsoDWI', 'dDWI_SENSE', 'eDWI_SENSE'],
       dtype=object)}
    :param target:
    :return:
    """
    file_name = "seq_classify.xlsx"
    root_dir = r"D:\pycharm_project\tumor_classification\data"
    t1_plus = pd.read_excel(os.path.join(root_dir, file_name), sheet_name=["T1+", "T1", "T2", "T2F", "ADC", "DWI"],
                            header=0, names=None, index_col=None,)
    maps = {}
    for i in t1_plus:
        maps[i] = t1_plus[i].values[:, 0]
    # target = "WIP_MIP_-_s3DI_MC_HR"
    for i in maps:
        if target in maps[i]:
            return i
    return None


def resave_raw_data(root_dir, raw_dir, new_dir):
    """
    :param root_dir:
    :return:
    """
    # saver = logger.Logger_csv(["sequences", "number"], '../data/', 'data_seqs.csv')
    files = glob.glob(root_dir+"/*/*/*.json")
    print(len(files))
    files += glob.glob(root_dir + "/*/*/*/*.json")
    print(len(files))
    modu_seq_maps = {}
    unkown = []
    for file in tqdm(files):
        try:
            with open(file, "r") as f:
                seq = json.load(f)["SeriesDescription"]
            type = get_type(seq)
            if not type:
                continue
            target_dir = file.replace(".json", ".nii.gz")

            # print(target_dir)
            outdir = target_dir.replace(raw_dir, new_dir)
            outdir = outdir.replace(".nii.gz", "_"+type+".nii.gz")
            (filepath, tempfilename) = os.path.split(outdir)
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
            shutil.copyfile(file, file.replace(raw_dir, new_dir).replace(".json", "_"+type+".json"))
            shutil.copyfile(target_dir, outdir)


        except Exception:
            unkown.append(file)
    # for i in modu_seq_maps:
    #     saver.update({"sequences": i, "number": modu_seq_maps[i]})
    # print(modu_seq_maps)
    # print(unkown)


if __name__ == '__main__':
    # root_dir =
    # root_dir = r"D:\dataset\tumor_classification\raw_data"
    # get_file_list(r"D:\dataset\tumor_classification\tumor_dataset1",
    #               save_name="dataset1_summary.csv")
    get_file_list(r"D:\dataset\tumor_classification\tumor_dataset1",
                  save_name="dataset1_summary_v1.csv")
    # get_file_list(, save_name="dataset1_summary.csv")

    # stats(root_dir)
    # find_seq(root_dir)
    # resave_raw_data(root_dir, "raw_data", "resaved_data")
    pass
