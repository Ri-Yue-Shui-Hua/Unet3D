# -*- coding : UTF-8 -*-
# @file   : train.py
# @Time   : 2023/6/19 0019 22:24
# @Author : wmz

import logging
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from SemanticDataset import SemanticDataset
from utils import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def observe_for_training(net, epoch, epochs, critic, global_step, batch_size, lr, loss, saved_dir):
    return None









