# -*- coding : UTF-8 -*-
# @file   : init_util.py
# @Time   : 2023/6/13 0013 22:30
# @Author : wmz

# https://gitcode.net/mirrors/panxiaobai/lits_pytorch/-/blob/master/init_util.py

from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight)











































