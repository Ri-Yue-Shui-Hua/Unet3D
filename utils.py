# -*- coding : UTF-8 -*-
# @file   : utils.py
# @Time   : 2023-06-18 20:31
# @Author : wmz

import numpy as np
import SimpleITK as sitk
import os
from config import Config
import torch


def resize_image(itkimage, factor, resamplemethod=sitk.sitkNearestNeighbor):
    '''
    :param itkimage: SimpleITK.Image
    :param factor:
    :param resamplemethod:
    :return:
    '''
    # python can compare two float numbers
    if factor == 1.0:
        return itkimage
    return resample_volume(itkimage, scaleFactor=factor, newSpacing=None)


def resample_volume(itkImage, scaleFactor=None, newSpacing=None):
    resampler = sitk.ResampleImageFilter()
    resample_method = sitk.sitkLinear
    originSize = itkImage.GetSize()
    originSpacing = itkImage.GetSpacing()
    originDirection = itkImage.GetDirection()
    if newSpacing is not None:
        newSize = [int(originSize[0] * (originSpacing[0] / newSpacing[0])),
                   int(originSize[1] * (originSpacing[1] / newSpacing[1])),
                   int(originSize[2] * (originSpacing[2] / newSpacing[2]))]
    elif scaleFactor is not None:
        newSize = (np.array(originSize) * scaleFactor).astype(int).tolist()
        newSpacing = (np.array(originSpacing) / scaleFactor).tolist()
    else:
        return itkImage

    def isListEqual(a, b):
        return (np.array(originSize) == np.array(newSize)).all()

    if isListEqual(originSize, newSize) and isListEqual(originSpacing, newSpacing):
        return itkImage

    resampler.SetReferenceImage(itkImage)
    resampler.SetSize(newSize)
    resampler.SetOutputSpacing(newSpacing)
    resampler.SetOutputDirection(originDirection)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    itkImageResampled = resampler.Execute(itkImage)
    return itkImageResampled


def convert_numpy_to_sitk(sitk_img, np_arr, scale=None):
    origin = sitk_img.GetOrigin()
    spacing = sitk_img.GetSpacing()
    direction = sitk_img.GetDirection()
    spacing = np.array(spacing) / float(scale)
    new_sitk_img = sitk.GetImageFromArray(np_arr)
    new_sitk_img.SetOrigin(origin)
    new_sitk_img.SetSpacing(spacing)
    new_sitk_img.SetDirection(direction)
    return new_sitk_img


def norm_method_1(img_arr):
    img_arr = (img_arr -512) / 1536.0
    return img_arr.astype(np.float32)


def norm_method_2(img_arr):
    img_arr = (img_arr - img_arr.min()) / img_arr.max()
    return img_arr.astype(np.float32)


def norm_method_3(img_arr):
    img_arr = img_arr.astype(np.float32) / img_arr.astype(np.float32).max()
    return img_arr.astype(np.float32)


def img_arr_threshold(img_arr, low=-1025, high=2048, norm_flag=False, norm_method='method_2'):
    '''
    :param img_arr: numpy ndarray
    :param low:
    :param high:
    :param norm_flag:
    :param norm_method:
    :return:
    '''
    def get_norm_func(norm_method):
        norm_func = None
        if norm_method == 'method_1':
            norm_func = norm_method_1
        elif norm_method == 'method_2':
            norm_func = norm_method_2
        elif norm_method == 'method_3':
            norm_func = norm_method_3
        return norm_func

    img_arr[img_arr < low] = low
    img_arr[img_arr > high] = high
    if norm_flag:
        norm_func = get_norm_func(norm_method)
        img_arr = norm_func(img_arr)
    return img_arr





