# -*- coding : UTF-8 -*-
# @file   : eval.py
# @Time   : 2023-06-21 23:25
# @Author : wmz

import logging
import SimpleITK as sitk
import os
import time
import numpy as np
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm
import pandas as pd

from utils import *

from config import Config
CUDA = 1
if(CUDA):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')


def seg_pred(net, eval_dataset, model_str, bz=1, saved_flag=False, eval_img_scale=1.0):
    eval_loader = DataLoader(eval_dataset, batch_size=bz)
    t = time.strftime("%m%d%H%M%S", time.localtime())
    eval_check_dir = os.path.join('check_val', model_str+'_'+t)
    net.eval()
    with tqdm(total=len(eval_dataset), desc='validation', unit='img') as pbar:
        for f, img in eval_loader:
            with torch.no_grad():
                img = img.to(device)
                _mark_pre = net(img)
            itk_img_file = f[0]

            if saved_flag:
                mark_pred = _mark_pre[0].cpu().numpy()
                result_dir = eval_check_dir
                save_result(itk_img_file, mark_pred, result_dir, eval_img_scale, 'Final')


def GetLargestConnectedCompont(binarysitk_image_array):
    binarysitk_image = sitk.GetImageFromArray(binarysitk_image_array)
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0
    outmask = outmask.astype(np.uint8)
    return outmask


def get_single_heatmap_dice_rm_small_ccs(m, gt):
    # remove small connected components before calculate dice.
    m = GetLargestConnectedCompont(m)
    TP = (m * gt).sum()
    e = 10e-20
    dice = (2 * TP + e) / (m.sum() + gt.sum() + e)
    return dice


def get_single_heatmap_dice(m, gt):
    TP = (m * gt).sum()
    e = 10e-20
    dice = (2 * TP + e) / (m.sum() + gt.sum() + e)
    return dice


def calculate_single_sample_dice(p, gt, p_thresh=[0.3, 0.5, 0.7]):
    assert p.shape == gt.shape
    p_shp = p.shape
    print('p_shp = ', p_shp)
    assert p_shp[0] >= Config.object_class_num
    # dice_list format: [ [femur_right_dice, femur_left, hip_right, hip_left] * 3]
    dice_list = []
    for t in p_thresh:
        m = (p > t) * 1
        heatmap_dice_list = []
        for c in range(Config.object_class_num):
            heatmap_dice = get_single_heatmap_dice(m[c], gt[c])
            # heatmap_dice = get_single_heatmap_dice_rm_small_ccs(m[c], gt[c])
            heatmap_dice_list.append(heatmap_dice)
        print('\t t = ', t, '\t dice = ', heatmap_dice_list)
        dice_list.append(heatmap_dice_list)
    return dice_list


def get_dice_info(f, p, gt, p_thresh=[0.3, 0.5, 0.7]):
    dice_list = calculate_single_sample_dice(p, gt, p_thresh)
    dice_collection = []
    m_dice = []
    for idx in range(len(dice_list)):
        dice = dice_list[idx]
        dice_collection.extend(dice)
        m_dice.append(np.array(dice).mean())
    dice_collection.extend(m_dice)
    return dice_collection


def save_summary_dict(d, saved_csv_dir, sub_dir='Summary', file_name='summary.csv'):
    if sub_dir is not None:
        saved_csv_dir = os.path.join(saved_csv_dir, sub_dir)
    if not os.path.exists(saved_csv_dir):
        os.makedirs(saved_csv_dir)
    summary_csv_file = os.path.join(saved_csv_dir, file_name)
    pd.DataFrame.from_dict(d).to_csv(summary_csv_file, index=False)
    return saved_csv_dir, summary_csv_file


def save_dice_info_to_csv(fid_list, dice_info_list, saved_dir):
    d = {}
    fid_list.append('Summary')
    d.setdefault('file', fid_list)
    avg_list = np.mean(dice_info_list, 0).tolist()
    dice_info_list.append(avg_list)
    dice_info_array = np.array(dice_info_list)

    # FR5: means femur_right dice at probability threshold = 0.5
    # HL75: means hip_left dice at probability threshold = 0.75
    # keys = ["FR3", "FL3", "HR3", "HL3",
    #         "FR5", "FL5", "HR5", "HL5",
    #         "FR7", "FL7", "HR7", "HL7",
    #         "Dice_3", "Dice_5", "Dice_7"]
    keys = ["F3",
            "F5",
            "F7",
            "Dice_3", "Dice_5", "Dice_7"]
    for idx in range(len(keys)):
        d.setdefault(keys[idx], dice_info_array[:, idx])

    csv_dir, csv_file = save_summary_dict(d, saved_dir, sub_dir=None, file_name='summary.csv')
    avg = d['Dice_5'][-1]
    print('avg = ', avg)
    new_dir = csv_dir + '_' + str(avg)
    os.rename(csv_dir, new_dir)
    old_csv = os.path.join(new_dir, os.path.basename(csv_file))
    new_csv = os.path.join(new_dir, os.path.basename(csv_file).split('.', 1)[0]+'_'+str(avg)+'.csv')
    os.rename(old_csv, new_csv)
    return avg


def check_direction_and_flip(direction, img_arr):
    img_flip = img_arr.copy()
    if (direction == np.array((1, 0, 0, 0, 1, 0, 0, 0, 1))).all():
        for idx in range(img_arr.shape[0]):
            img_flip[idx] = np.flip(img_arr[idx], 2).copy()
    return img_flip.copy()


def check_direction_and_adjust_heatmap(direction, heatmaps):
    # Can use permute function to do this thing...
    if ((direction == np.array((1, 0, 0, 0, 1, 0, 0, 0, 1))).all()):
        assert len(heatmaps) % 2 == 0
        for i in range(len(heatmaps)//2):
            L = heatmaps[i*2].copy()
            R = heatmaps[i*2+1].copy()
            heatmaps[i*2] = R
            heatmaps[i*2+1] = L
    return heatmaps


def seg_result_postprocess(p, itk_img_file):
    # p.shape = [C,X,Y,Z]
    itk_img = sitk.ReadImage(itk_img_file)
    direction = np.array(itk_img.GetDirection())
    if 'left' in itk_img_file:
        p = check_direction_and_flip(direction, p[0:Config.object_class_num])

    return p.copy()


def seg_eval(net, eval_dataset, model_str, bz=1, saved_dir='check_val', saved_flag=False, eval_img_scale=1.0):
    eval_loader = DataLoader(eval_dataset, batch_size=bz)
    t = time.strftime("%m%d%H%M%S", time.localtime())
    net.eval()
    dice_info_list = []
    fid_list = []
    # Batch Size  always be 1 .  if it;s large than 1, the code need to be re-write.
    with tqdm(total=len(eval_dataset), desc='validation', unit='img') as pbar:
        for f, img, gt in eval_loader:
            with torch.no_grad():
                img = img.to(device)
                _mark_pre = net(img)
                p = _mark_pre[0].cpu().numpy()
                gt = gt[0].cpu().numpy()
                itk_img_file = f[0]
                # p = seg_result_postprocess(p, itk_img_file)
            dice_info = get_dice_info(f[0], p, gt)

            itk_img = sitk.ReadImage(itk_img_file)
            dice_info_list.append(dice_info)
            fid_list.append(os.path.basename(itk_img_file).split('.', 1)[0])
            if saved_flag:
                save_result(itk_img_file, p, saved_dir, eval_img_scale, 'Final')
    avg = save_dice_info_to_csv(fid_list, dice_info_list, saved_dir)
    return avg


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
