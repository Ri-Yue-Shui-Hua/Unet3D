# -*- coding : UTF-8 -*-
# @file   : CommonDataset.py
# @Time   : 2024-02-29 13:41
# @Author : wmz

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import SimpleITK as sitk
import numpy as np
from utils import resize_image, img_arr_threshold
from config import Config
from data_preprocess import transform
from glob import glob
import pandas as pd


class CommonDataset(Dataset):
	def __init__(self, root_dir, mode, image_list, label_list, scale=1):
		"""
		Args:
			root_dir: dataset root directory
			mode: train or test
			image_list: image file list in relative path
			label_list: label file list in relative path
		"""
		self.root_dir = root_dir
		self.mode = mode
		self.image_list = image_list
		self.label_list = label_list
		self.scale = scale

	def __len__(self):
		return len(self.image_list)

	def get_image_array_from_file(self, file):
		image = sitk.ReadImage(file)
		image_arr = sitk.GetArrayFromImage(image)
		return image, image_arr

	def __getitem__(self, idx):
		image_file = os.path.join(self.root_dir, self.image_list[idx])
		label_file = os.path.join(self.root_dir, self.label_list[idx])
		image, image_arr = self.get_image_array_from_file(image_file)
		label_image, label_arr = self.get_image_array_from_file(label_file)
		image_arr = image_arr[np.newaxis, :, :, :]
		label_arr = label_arr[np.newaxis, :, :, :]
		image_arr = image_arr.astype(np.float32)
		label_arr = label_arr.astype(np.float32)

		if self.mode == "train":
			image_arr, label_arr = transform(image_arr, label_arr)
			return image_file, torch.from_numpy(image_arr).float(), torch.from_numpy(label_arr).float()
		else:
			return image_file, torch.from_numpy(image_arr).float()


def get_file_info_csv(csv_file, mode):
	data = pd.read_csv(csv_file, delimiter='\t')
	title_list = data.columns[0].split(",")
	i_index = title_list.index("image")
	l_index = title_list.index('label')
	m_index = title_list.index("mode")
	file_list = []
	label_list = []
	for i, v in enumerate(data.values):
		list_name = v[0].split(",")
		if mode == list_name[m_index]:
			file_list.append(list_name[i_index])
			label_list.append(list_name[l_index])
	return file_list, label_list


if __name__ == "__main__":
	up_femur_csv_file = "up_femur_seg_info.csv"
	root_dir = r"E:\Dataset\CropForUpDnFemur"
	data_list = []
	mask_list = []
	mode = "test"
	data_list, mask_list = get_file_info_csv(up_femur_csv_file, mode)
	common_dataset = CommonDataset(root_dir, mode, data_list, mask_list)
	dataloader = DataLoader(common_dataset)
	num_epoches = 1
	index = 0
	for epoch in range(num_epoches):
		for image_file, img, label in dataloader:
			print(f"{index}", image_file)
			index += 1
