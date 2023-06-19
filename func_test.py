# -*- coding : UTF-8 -*-
# @file   : func_test.py
# @Time   : 2023-06-17 21:32
# @Author : wmz

import torch

if __name__ == '__main__':
    count = torch.cuda.device_count()
    print("cuda count: ", count)
    cuda_available = torch.cuda.is_available()
    print("cuda available :", cuda_available)

