# -*- coding : UTF-8 -*-
# @file   : SemanticDataset.py
# @Time   : 2023-06-18 16:00
# @Author : wmz
import os
from glob import glob
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import resize_image
from data_preprocess import transform


class SemanticDataset(Dataset):
    def __init__(self, class_nums, data_source=None, label_dir=None, img_scale=1.0, label_scale=None, trans_flag=True):
        '''
        :param data_source: either a dir name or a file lists. If it is a dir name, we can search all files under this dir.
            If it is a file list, we will use it directly.
        :param label_dir: either a dir name or a list of dir name; we can automatically check whether the heatmap exists
            in which directory.
        :param img_scale: setting this parameter, we can resize the volume when loading it
        :param label_scale: same with data_scale
        :param trans_flag: when training, the data often with trans, when validation or testing the data without trans
        '''
        self.label_dir = label_dir
        self.input_channels = 1
        self.output_channels = class_nums
        self.img_scale = img_scale
        self.label_scale = label_scale if label_scale is not None else img_scale

        self.trans_flag = trans_flag
        self.all_fs = self.get_all_file(data_source)

    def __len__(self):
        return len(self.all_fs)

    def get_all_file(self, input_str):
        if isinstance(input_str, list):
            fs = [f for f in input_str if os.path.exists(f)]
        elif isinstance(input_str, str):
            fs = [file for file in glob(os.path.join(input_str, "*")) if not file.startswith('.')]
        else:
            raise ValueError('Dataset Input Error!')
        return sorted(fs)

    def get_seg_file_name(self, itk_img_file, label_id):
        file_id = os.path.basename(itk_img_file).split('.', 1)[0]
        postfix = '.'+ os.path.basename(itk_img_file).split('.',1)[1]
        # concatenate paths according to your need
        fn = 'Seg_' + file_id + postfix
        if isinstance(self.label_dir, list):
            for d in self.label_dir:
                label_path = os.path.join(d, fn)
                if os.path.exists(label_path):
                    break
        else:
            label_path = os.path.join(self.label_dir, fn)
        return label_path

    def check_direction_and_flip(self, direction, img_arr):
        if (direction == np.array((1,0,0,0,1,0,0,0,1))).all():
            img_arr = np.flip(img_arr, [1, 2]).copy()
        return img_arr

    def get_array_from_volume(self, itk_img_file, img_scale):
        itk_img = sitk.ReadImage(itk_img_file)
        itk_img = resize_image(itk_img, img_scale)
        img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float)
        return itk_img, img_arr

    def __getitem__(self, idx):
        itk_img_file = self.all_fs[idx]
        itk_img, img_arr = self.get_array_from_volume(itk_img_file, self.img_scale)
        direction = np.array(itk_img.GetDirection())
        # print(itk_img_file)
        # 数据正则化
        img_arr = img_arr[np.newaxis, :, :, :]
        if self.label_dir is not None:
            labels = []
            for label_id in range(self.output_channels):
                label_path = self.get_seg_file_name(itk_img_file, label_id+1)
                print('\t' + label_path)
                sitk_label, label_arr = self.get_array_from_volume(label_path, self.label_scale)
                labels.append(label_arr)
            labels = np.array(labels).astype(np.float)
            if self.trans_flag:
                img_arr, labels = transform(img_arr, labels)
            return itk_img_file, torch.from_numpy(img_arr).float(), torch.from_numpy(labels).float()
        else:
            return itk_img_file, torch.from_numpy(img_arr).float











