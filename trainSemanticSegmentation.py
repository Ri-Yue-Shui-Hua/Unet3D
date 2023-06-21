# -*- coding : UTF-8 -*-
# @file   : trainSemanticSegmentation.py
# @Time   : 2023-06-21 23:24
# @Author : wmz

from train import *
from eval import *
from glob import glob
import numpy as np
import csv
import torch
from nets.UNet_3Ddc2 import UNet_3D as UNet

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model_str = 'Deconv_ReLu_MSE'


def train_and_eval():
    dataFolderName = "AnkleSegmentation"
    train_data_dir = r'D:\Dataset\TibiaSegmentation\train'
    train_heatmap_dir = r'D:\Dataset\TibiaSegmentation\label'
    # train_data_dir = '/Data/wmz/Dataset/' + dataFolderName + '/train'
    # train_heatmap_dir = '/Data/wmz/Dataset/' + dataFolderName + '/label'
    eval_data = '/Data/wmz/Dataset/' + dataFolderName + '/test'
    eval_heatmap_dir = '/Data/wmz/Dataset/' + dataFolderName + '/label'
    saved_dir = '/Data/wmz/Dataset/' + dataFolderName + '/test_result'


    train_for_Seg(train_data_dir, train_heatmap_dir, model_str)
    # model_path_list = sorted(glob(os.path.join('trained_model', model_str + '*/*.pth')))
    # print(model_path_list)
    # eval_all_models_for_seg(eval_data, eval_heatmap_dir, model_str, model_path_list, saved_dir, postfix='')
    # eval_all_models_for_seg(patient_eval_data, patient_eval_mask_dir, model_str, model_path_list, saved_dir, postfix='_patient')


def train_for_Seg(train_data_dir, train_heatmap_dir, model_str):
    start_time = time.strftime("%m%d%H%M%S", time.localtime())
    dataset = SemanticDataset(train_data_dir, train_heatmap_dir, 1.0, 1.0)
    # Here , class_nums --> objet_nums in segmentation....
    net = UNet(1, Config.class_nums)
    net.to(device)
    print(device)
    val_rate = 0
    Config.lr = 0.0001
    epoch = 200
    bz = 2
    train(net, dataset, val_rate, epoch, bz, Config.lr, model_str)
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
                      output_names=output_names, dynamic_axes={'input': [2, 3, 4], 'output': [2, 3, 4]})  # , dynamic_axes={'input': [2, 3, 4], 'output': [2, 3, 4]}
    print("export onnx model success!")


def pred_for_Seg(eval_data, eval_heatmap_dir, model_path, model_str):
    start_time = time.strftime("%m%d%H%M%S", time.localtime())
    dataset = SemanticDataset(eval_data, None, 1.0, 1.0, trans_flag=False)
    net = UNet(1, Config.class_nums)
    net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(device)
    with torch.no_grad():
        input = torch.randn(1, 1, 256, 96, 96, device=device)
        input_names = ['input']
        output_names = ['output']
        export_onnx(net, input, input_names, output_names, "FemurSeg.onnx")
        # save_trace(net, input, "Hip_Femur_Locate.pt")
        # save_script(net, "Hip_Femur_Locate.pt")

        exit(0)

    bz = 1
    # seg_eval(net, dataset, model_str, bz, eval_heatmap_dir, saved_flag=True, eval_img_scale=1.0)
    seg_pred(net, dataset, model_str, bz, saved_flag=True, eval_img_scale=1.0)
    end_time = time.strftime("%m%d%H%M%S", time.localtime())
    print(model_str, '\tStart Time', start_time, '\tEnd Time: ', end_time)


def eval_for_Seg(eval_data, eval_heatmap_dir, model_path, saved_dir, model_str, saved_flag=True):
    dataset =SemanticDataset(eval_data, eval_heatmap_dir, 1.0, 1.0, trans_flag=False)
    net = UNet(1, Config.class_nums)
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
        avg = eval_for_Seg(eval_data, eval_heatmap_dir, model_path, saved_dir=result_saved_dir, model_str=model_str, saved_flag=True)
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
        eval_for_Seg(eval_data, eval_heatmap_dir, model_path, saved_dir=result_saved_dir, model_str=model_str, saved_flag=True)

    run_best_save_result()


if __name__ == "__main__":
    train_and_eval()

