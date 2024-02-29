# -*- coding : UTF-8 -*-
# @file   : trainSemanticSegmentation.py
# @Time   : 2023-06-21 23:24
# @Author : wmz

from train import *
from eval import *
import argparse
from glob import glob
import numpy as np
import csv
import torch
from nets.UNet_3Ddc2 import UNet_3D as UNet


def train_for_seg(data_dir, label_dir, model_str, args):
    start_time = time.strftime("%m%d%H%M%S", time.localtime())
    dataset = SemanticDataset(data_dir, label_dir, img_scale=1.0, label_scale=1.0)
    # Here , class_nums --> objet_nums in segmentation....
    net = UNet(1, args.num_class)
    net.to(device)
    print(device)
    lr = args.lr
    epochs = args.epochs
    bz = args.batch_size
    train_parameters = TrainParameters(val_rate=0, lr=lr, batch_size=bz, epochs=epochs, save_epoch_step=1)
    train(net, dataset, train_parameters, model_str)
    end_time = time.strftime("%m%d%H%M%S", time.localtime())
    print(model_str, '\tStart Time', start_time, '\tEnd Time: ', end_time)


def export_onnx(model, input, input_names, output_names, modelname):
    model.eval()
    dummy_input = input
    torch.onnx.export(model, dummy_input, modelname,
                      export_params=True,
                      verbose=False,
                      opset_version=12,
                      input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': [2, 3, 4], 'output': [2, 3, 4]})
    print("export onnx model success!")


def pred_for_seg(eval_data, model_path, model_str, class_nums):
    start_time = time.strftime("%m%d%H%M%S", time.localtime())
    dataset = SemanticDataset(eval_data, None, 1.0, 1.0, trans_flag=False)
    net = UNet(1, class_nums)
    net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(device)
    with torch.no_grad():
        input = torch.randn(1, 1, 256, 96, 96, device=device)
        input_names = ['input']
        output_names = ['output']
        export_onnx(net, input, input_names, output_names, "FemurSeg.onnx")
        exit(0)
    bz = 1
    seg_pred(net, dataset, model_str, bz, saved_flag=True, eval_img_scale=1.0)
    end_time = time.strftime("%m%d%H%M%S", time.localtime())
    print(model_str, '\tStart Time', start_time, '\tEnd Time: ', end_time)


def eval_for_seg(eval_data, eval_heatmap_dir, model_path, saved_dir, model_str, class_nums, saved_flag=True):
    dataset = SemanticDataset(eval_data, eval_heatmap_dir, 1.0, 1.0, trans_flag=False)
    net = UNet(1, class_nums)
    net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(device)
    bz = 1
    avg = seg_eval(net, dataset, model_str, bz=bz, saved_dir=saved_dir, saved_flag=saved_flag, eval_img_scale=1.0)
    return avg


def eval_all_models_for_seg(eval_data, eval_heatmap_dir, model_str, model_path_list, saved_dir, postfix=''):
    saved_dir = os.path.join(saved_dir, model_str+postfix)

    def get_wanted_path(m, path):
        epoch = str(m.split('_')[-4])
        return os.path.join(path, epoch)
    max_avg = -1
    best_info = {"model": None, "dice": None}
    for model_path in model_path_list:
        result_saved_dir = get_wanted_path(model_path, saved_dir)
        avg = eval_for_seg(eval_data, eval_heatmap_dir, model_path, saved_dir=result_saved_dir, model_str=model_str, saved_flag=True)
        if(max_avg < avg):
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
        eval_for_seg(eval_data, eval_heatmap_dir, model_path, saved_dir=result_saved_dir, model_str=model_str, saved_flag=True)

    run_best_save_result()


def get_args():
    parser = argparse.ArgumentParser(description='Train the Segmentation')
    parser.add_argument('--num_class', '-c', metavar='C', type=int, default=1, help='Number of class')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default="./checkpoints/checkpoint_epoch54.pth",
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == "__main__":
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model_str = 'Deconv_ReLu_MSE'
    args = get_args()
    data_folder_name = "HipSegmentation"
    folder_prev = 'D:/Dataset/'
    train_image_folder = "imagesTr"
    train_label_folder = "labelsTr"
    train_data_dir = os.path.join(folder_prev, data_folder_name, train_image_folder)
    train_label_dir = os.path.join(folder_prev, data_folder_name, train_label_folder)
    # 训练
    train_for_seg(train_data_dir, train_label_dir, model_str, args)
    # 评估
    # eval_data = folder_prev + data_folder_name + '/test'
    # eval_heatmap_dir = folder_prev + data_folder_name + '/label'
    # saved_dir = folder_prev + data_folder_name + '/test_result'
    # model_path_list = sorted(glob(os.path.join('trained_model', model_str + '*/*.pth')))
    # eval_all_models_for_seg(eval_data, eval_heatmap_dir, model_str, model_path_list, saved_dir, postfix='')


