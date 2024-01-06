# -*- coding : UTF-8 -*-
# @file   : utils.py
# @Time   : 2023-06-18 20:31
# @Author : wmz

import numpy as np
import SimpleITK as sitk
import os
import torch


class Parameter:
    def __init__(self, val_rate, lr, batch_size, epochs):
        self.val_rate = val_rate
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs



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


def save_result(itk_img_file, mark_pred, saved_dir, scale=None, sub_dir=None, post_process=None):
    pass
