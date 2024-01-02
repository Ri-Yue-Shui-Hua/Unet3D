import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2):
        super().__init__()
        self.doubleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channel1, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channel1, out_channel2, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)

class EdgeOut(nn.Module):
    def __init__(self, in_channels, edge_num):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
        self.avgpool = nn.AvgPool3d(3)
        self.linear = nn.Linear(42 * 42 * 66, edge_num * 3)
        self.edge_num = edge_num

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.max(x, dim=1)[0]
        x = torch.reshape(x, (1, 42 * 42 * 66))
        x = self.linear(x)
        x = torch.reshape(x, (1, self.edge_num, 3))
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, padding=0),
            DoubleConv(in_channels, out_channel1, out_channel2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2):
        super().__init__()
        self.up =nn.Upsample(scale_factor=2.0, mode="trilinear")
        self.conv = DoubleConv(in_channels, out_channel2, out_channel2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.conv.weight, 0, 0.0001)
        nn.init.constant_(self.conv.bias, 0)
        # self.conv = nn.Conv3d(in_channels, out_channels,kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Spacial_Info(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Upsample(scale_factor=0.25, mode="trilinear")
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, padding=3)

        self.conv4 = nn.Conv3d(64, out_channels, kernel_size=7, padding=3)
        self.conv5 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        self.conv6 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        nn.init.normal_(self.conv6.weight, 0, 0.0001)
        nn.init.constant_(self.conv6.bias, 0)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        a, b, c = x.shape[2:]
        x = self.down(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.softmax(x)
        self.up = nn.Upsample((a, b, c), mode="trilinear")
        x = self.up(x)
        return x

class Spacial_Info_Tanh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Upsample(scale_factor=0.25, mode="trilinear")
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, padding=3)

        self.conv4 = nn.Conv3d(64, out_channels, kernel_size=7, padding=3)
        self.conv5 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        self.conv6 = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3)
        nn.init.normal_(self.conv6.weight, 0, 0.0001)
        nn.init.constant_(self.conv6.bias, 0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        a, b, c = x.shape[2:]
        x = self.down(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.tanh(x)
        self.up = nn.Upsample((a, b, c), mode="trilinear")
        x = self.up(x)
        return x

















