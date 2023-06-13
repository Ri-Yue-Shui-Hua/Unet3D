# -*- coding : UTF-8 -*-
# @file   : Unet3d-02.py
# @Time   : 2023/6/7 0007 21:59
# @Author : wmz

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64],
                 kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        if isinstance(out_channels, int):
            out_channels = [out_channels] * 2

        self.doubleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[0],
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[0], out_channels[1],
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64]):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # self.trans_conv = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.trans_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, dilation=2, padding=1, output_padding=1)

        self.double_conv = DoubleConv(out_channels + out_channels, out_channels)

    def forward(self, x_trans, x_cat):
        x = self.trans_conv(x_trans)
        #print("x_trans", x_trans.shape)
        #print("x", x.shape)
        #print("x_cat", x_cat.shape)
        x = torch.cat([x_cat, x], dim=1)
        return self.double_conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        #self.bn = nn.BatchNorm3d(out_channels)
        self.norm = nn.ReLU()
        #self.norm = nn.Tanh()

    def forward(self, x):
        return self.norm(self.conv(x))


ch = [32,32,64,128,256,512]
class UNet_3D(nn.Module):
    def __init__(self, in_channels, out_classes, last_layer=Out):
        super(UNet_3D, self).__init__()
        self.encoder_1 = DoubleConv(in_channels, [ch[0], ch[1]]) # 32,32
        self.encoder_2 = Down(ch[1], [ch[2], ch[2]]) #32, 64 64
        self.encoder_3 = Down(ch[2], [ch[3], ch[3]]) # 64,128,128
        self.encoder_4 = Down(ch[3], [ch[4], ch[4]])  # 128,256,256
        self.buttom = Down(ch[4], [ch[5], ch[5]]) # 256,512,512
        self.decoder_4 = Up(ch[5], ch[4])  # 512,256
        self.decoder_3 = Up(ch[4], ch[3]) #256,128
        self.decoder_2 = Up(ch[3], ch[2]) # 128, 64
        self.decoder_1 = Up(ch[2], ch[1]) # 64, 32
        self.output = last_layer(ch[1], out_classes) if last_layer is not None else None

    # def forward(self, x):
    #     e1 = self.encoder_1(x)
    #     e2 = self.encoder_2(e1)
    #     e3 = self.encoder_3(e2)
    #     e4 = self.encoder_4(e3)
    #     e = self.buttom(e4)
    #     #print(e.shape, e3.shape)
    #     d4 = self.decoder_4(e, e4)
    #     d3 = self.decoder_3(d4, e3)
    #     d2 = self.decoder_2(d3, e2)
    #     d1 = self.decoder_1(d2, e1)
    #     if self.output is not None:
    #         d1 = self.output(d1)
    #     return d1

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)
        # e = self.buttom(e4)
        #print(e.shape, e3.shape)
        # d4 = self.decoder_4(e, e4)
        d3 = self.decoder_3(e4, e3)
        d2 = self.decoder_2(d3, e2)
        d1 = self.decoder_1(d2, e1)
        if self.output is not None:
            d1 = self.output(d1)
        return d1


def export_onnx(model, input, input_names, output_names, modelname):
    dummy_input = input
    torch.onnx.export(model, dummy_input, modelname,
                      export_params=True,
                      verbose=False,
                      opset_version=13,
                      input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': {2: "input_dynamic_axes_1", 3: "input_dynamic_axes_1",
                    4: "input_dynamic_axes_1"}, 'output': {2: "input_dynamic_axes_1", 3: "input_dynamic_axes_1", 4: "input_dynamic_axes_1"}})
    # torch.onnx.export(model, dummy_input, modelname,
    #                   export_params=True,
    #                   verbose=False,
    #                   opset_version=12,
    #                   input_names=input_names,
    #                   output_names=output_names)
    print("export onnx model success!")


if __name__ == "__main__":
    import numpy as np
    device = torch.device('cpu')

    x = np.random.rand(1, 1, 224, 144, 144)
    pt = torch.from_numpy(x).float()
    net = UNet_3D(in_channels=1, out_classes=1)
    # net.train()
    y = net(pt)
    with torch.no_grad():
        # net.eval()
        input = torch.randn(1, 1, 224, 144, 144, device=device)
        # itk_img = sitk.ReadImage('D:/Dataset/Hip_Femur_Detection/Detect/BJ_02006.nii.gz')
        # img_arr = sitk.GetArrayFromImage(itk_img)
        # img_arr = img_arr[np.newaxis, np.newaxis, :, :, :]
        # input = torch.from_numpy(img_arr).float()
        input_names = ['input']
        output_names = ['output']
        export_onnx(net, input, input_names, output_names, "test.onnx")

        exit(0)
    print(y.shape)




