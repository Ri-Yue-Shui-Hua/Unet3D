# -*- coding : UTF-8 -*-
# @file   : train.py
# @Time   : 2023/6/19 0019 22:24
# @Author : wmz

import logging
import os.path
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from SemanticDataset import SemanticDataset
from utils import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def observe_for_training(net, epoch, epochs, critic, global_step, batch_size, lr, loss, sample_file_name, saved_dir):
    """
    Observe the training process
    :param net:
    :param epoch:
    :param epochs:
    :param critic:
    :param global_step:
    :param batch_size:
    :param lr:
    :param loss:
    :param sample_file_name:
    :param saved_dir:
    :return:
    """
    return None
    sample_epoch_saved_dir_name = saved_dir
    if not os.path.exists(sample_epoch_saved_dir_name):
        os.makedirs(sample_epoch_saved_dir_name)
    sample_global_step_saved_dir = os.path.join(sample_epoch_saved_dir_name, str(epoch) + '_' + str(global_step) + '_' + str(loss))
    if not os.path.exists(sample_global_step_saved_dir):
        os.makedirs(sample_global_step_saved_dir)
    net_saved_name = 'unet' + f'_LR_{lr}_BS_{batch_size}_{critic}_ep_{epoch}_{epochs}' + '.pth'
    net_saved_path = os.path.join(sample_global_step_saved_dir, net_saved_name)
    torch.save(net.state_dict(), net_saved_path)
    observer_dataset = SemanticDataset([sample_file_name])
    observer_loader = DataLoader(observer_dataset, batch_size=1)
    with torch.no_grad():
        for file_id, s_img in observer_loader:
            s_img = s_img.to(device)
            _mark_pred = net(s_img).to(device)
            mark_pred = _mark_pred[0].cpu().numpy()
            print(file_id[0], "\t Gen and Save Heatmap......")
            itk_img_file = os.path.join(observer_dataset.img_dir, file_id[0] + '.nii.gz')
            save_result(itk_img_file, mark_pred, sample_global_step_saved_dir)


def norm_for_each_heatmap(heatmap_batch):
    norm_heatmap = heatmap_batch / torch.exp(heatmap_batch).sum(dim=(2, 3, 4), keepdim=True)
    return norm_heatmap


def KL_loss_for_each_heatmap(p, q):
    KL_Loss = torch.nn.KLDivLoss()
    N, C, X, Y, Z = p.size()
    loss = 0.0
    for bz in range(N):
        for ch in range(C):
            ph = p[bz, ch, :]
            qh = p[bz, ch, :]
            loss += KL_Loss(ph.log(), qh)
    return loss


def JS_loss(pred, gt):
    KL_Loss = torch.nn.KLDivLoss()
    pred_norm = norm_for_each_heatmap(pred)
    gt_norm = norm_for_each_heatmap(gt)
    m = (pred_norm + gt_norm) * 0.5
    pm_loss = KL_loss_for_each_heatmap(pred_norm, m)
    qm_loss = KL_loss_for_each_heatmap(gt_norm, m)
    loss = (pm_loss + qm_loss) * 0.5
    return loss


def train(net, dataset, val_rate, epochs, batch_size, lr, saved_epoch_step, model_str=''):
    num_val = int(len(dataset) * val_rate)
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion, crit = torch.nn.MSELoss(), "MSE"
    log_saved_dir = os.path.join('runs', model_str) if model_str != '' else None
    writer = SummaryWriter(log_dir=log_saved_dir, comment=f'LR_{lr}_{crit}_EP_{epochs}')
    logging.info(f'''Start training:
    Device:     {device}
    Epochs:     {epochs}
    Batch Size: {batch_size}
    LR:         {lr}
    Criterion:  {crit}
    Train size: {num_train}
    Val size:   {num_val}
    ''')
    t = time.strftime("%m%d%H%M%S", time.localtime())
    saved_model_dir = os.path.join('trained_model', model_str + '_' + t)
    global_step = 0
    for epoch in range(epochs):
        net.train()
        with tqdm(total=num_train, dec=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for f, img, mark_gt in train_loader:
                img, mark_gt = img.to(device), mark_gt.to(device)
                mark_pred = net(img)
                loss = criterion(mark_pred, mark_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(img.shape[0])
                writer.add_scalar('Loss/train_MSE', loss.item(), global_step)
                global_step += 1
        net.eval()
        with tqdm(total=num_val, desc='validation', unit='img') as pbar:
            for _, img, mark_gt in val_loader:
                with torch.no_grad():
                    img, mark_gt = img.to(device), mark_gt.to(device)
                    mark_pred = net(img)
                    loss_mse = criterion(mark_pred, mark_gt)
                    pbar.set_postfix(**{'loss': loss.item()})
                    pbar.update(img.shape[0])
                    writer.add_scalar('Loss/Val_MSE', loss_mse.item(), global_step)

        if epoch % saved_epoch_step == 0:
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            saved_model_name = model_str + '_' + t + f'LR_{lr}_BS_{batch_size}_{crit}_ep_{epoch}_{epochs}' + str(loss.item()) + '.pth'
            saved_model_name = os.path.join(saved_model_dir, saved_model_name)
            torch.save(net.state_dict(), saved_model_name)
    writer.close()
    logging.info(f'saving model')


def train_ep_200(des=''):
    # Train data: half size
    start_time = time.strftime("%m%d%H%M%S", time.localtime())
    from nets.UNetModel import UNet
    train_heatmap_dir = '/Data/wmz/Dataset/Train/'
    train_data_dir = ''
    dataset = SemanticDataset(train_data_dir, train_heatmap_dir, 1.0, 1.0)
    net = UNet(dataset.input_channels, dataset.output_channels)
    net.to(device)
    val_rate = 0
    lr = 0.0001
    epoch = 200
    bz = 1
    train(net, dataset, val_rate, epoch, bz, lr, model_str='')
    end_time = time.strftime("%m%d%H%M%S", time.localtime())
    print(des, '\tStart Time', start_time, '\tEnd Time: ', end_time)


if __name__ == "__main__":
    train_ep_200()

















































