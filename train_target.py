# -*- coding : UTF-8 -*-
# @file   : train_target.py
# @Time   : 2024-02-29 16:34
# @Author : wmz

from train import *
from eval import *
import torch
from glob import glob
import time
from nets.UNet_3Ddc2 import UNet_3D as UNet
import argparse
from CommonDataset import CommonDataset

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model_str = 'Deconv_ReLu_MSE'


def get_args():
	parser = argparse.ArgumentParser(description='Train the 3D Unet on images and target masks')
	parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
	parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate',
	                    dest='lr')
	parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
	parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
	parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
	                    help='Percent of the data that is used as validation (0-100)')

	return parser.parse_args()


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


def train_and_eval():
	data_root_dir = ""
	target_cvs_file = "up_femur_seg_info.cvs"
	mode = "train"
	data_list, mask_list = get_file_info_csv(target_cvs_file, mode)
	common_dataset = CommonDataset(data_root_dir, mode, data_list, mask_list)
	model_path_list = sorted(glob(os.path.join("trained_model", model_str + "*/*.pth")))
	saved_dir = data_root_dir + '/test_result'
	# 1. train
	train_for_seg(common_dataset, model_str, model_path_list)

	# 2. test
	mode = "test"
	test_data_list, test_mask_list = get_file_info_csv(target_cvs_file, mode)
	common_test_dataset = CommonDataset(data_root_dir, mode, test_data_list, test_mask_list)
	eval_all_models_for_seg(common_test_dataset, model_str, model_path_list, saved_dir, postfix='')


def train_for_seg(dataset, model_str, model_path_list):
	start_time = time.strftime("%m%d%H%M%S", time.localtime())
	net = UNet(1, Config.class_nums)
	net.to(device)
	Resume = True
	if Resume:
		if len(model_path_list) != 0:
			prev_epoch_list = []
			for epoch, model_file in enumerate(model_path_list):
				file_ID = os.path.basename(model_file)
				prev_epoch_list.append(int(file_ID.split("_")[10]))
			start_epoch = np.array(prev_epoch_list).max()
			path_checkpoint = model_path_list[np.where(prev_epoch_list == start_epoch)[0][0]]
			print("path_checkpoint: ", path_checkpoint)
			checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
			net.load_state_dict(checkpoint)
		else:
			start_epoch = 0
	else:
		start_epoch = 0
	val_rate = 0
	Config.lr = 0.0001
	epochs = 200
	bz = 1
	train(net, dataset, val_rate, start_epoch, epochs, bz, Config.lr, model_str)
	end_time = time.strftime("%m%d%H%M%S", time.localtime())
	print(model_str, '\tStart Time', start_time, '\tEnd Time: ', end_time)


def eval_for_seg(dataset, model_path, saved_dir, model_str, saved_flag=True):
	net = UNet(1, Config.class_nums)
	net.to(device)
	net.load_state_dict(torch.load(model_path, map_location=device))
	bz = 1
	avg = seg_eval(net, dataset, model_str, bz=bz, saved_dir=saved_dir, saved_flag=saved_flag, eval_img_scale=1.0)
	return avg


def eval_all_models_for_seg(dataset, model_str, model_path_list, saved_dir):
	saved_dir = os.path.join(saved_dir, model_str)

	def get_wanted_path(m, path):
		epoch = str(m.split('_')[-4])
		return os.path.join(path, epoch)

	max_avg = -1
	best_info = {"model": None, "dice": None}
	for model_path in model_path_list:
		result_saved_dir = get_wanted_path(model_path, saved_dir)
		avg = eval_for_seg(dataset, model_path, saved_dir=result_saved_dir, model_str=model_str, saved_flag=True)
		if (max_avg < avg):
			max_avg = avg
			best_info['model'] = model_path
			best_info['dice'] = avg
	print(best_info)

	def run_best_save_result():
		result_saved_dir = os.path.join(saved_dir, '0_Best_model_result')
		if not os.path.exists(result_saved_dir):
			os.makedirs(result_saved_dir)
		model_path = best_info['model']
		import shutil
		shutil.copy(model_path, os.path.join(result_saved_dir, os.path.basename(model_path)))
		eval_for_seg(dataset, model_path, saved_dir=result_saved_dir, model_str=model_str, saved_flag=True)

	run_best_save_result()
