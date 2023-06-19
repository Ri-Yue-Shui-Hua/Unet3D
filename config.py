# -*- coding : UTF-8 -*-
# @file   : config.py
# @Time   : 2023-06-18 16:26
# @Author : wmz


class Config(object):
    class_nums = 1
    object_class_num = 1
    train_img_scale = 1.0
    train_label_scale = 1.0
    eval_img_scale = 1.0
    sigma = 2
    thresh = 0.9
    img_suffix = '.nii.gz'

    # preprocessing
    training_data_threshold_flag = True
    train_data_threshold_low = -1024
    training_data_threshold_high = 2048

    training_data_norm_flag = False
    Norm_method = 'method_1'

    # training parameters
    val_rate = 0
    lr = 0.0001
    batch_size = 1
    epochs = 120

    # Saved
    saved_epoch_step = 1
    saved_global_step = 50
