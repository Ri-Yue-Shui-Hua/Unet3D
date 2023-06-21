from nets.UNetParts import *


class UNet(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.conv = DoubleConv(channels, 16, 32)
        self.down1 = Down(32, 32, 64)
        self.down2 = Down(64, 64, 128)
        self.down3 = Down(128, 128, 256)

        self.up1 = Up(384, 128, 128)
        self.up2 = Up(192, 64, 64)
        self.up3 = Up(96, 32, 32)
        self.out = OutConv(32, classes)
        self.spacial = Spacial_Info_Tanh(32, classes)
        #self.spacial = Spacial_Info(32, classes).cuda(0)
        #self.norm = nn.ReLU(inplace=True)

        #self.edge = EdgeOut(32, 26).cuda(1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = self.up1(x3,x2)
        x = self.up2(x,x1)
        x = self.up3(x,x0)
        #edge = self.edge(x)

        local = self.out(x)

        spacial = self.spacial(x)
        x = local * spacial

        # for check
        # self.localt = local
        # self.spacialt = spacial
        #x = self.norm(x)
        return x
        # return [x, edge]

class SCN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(channels=1,classes=15)
        self.sccov0 = nn.Conv3d(in_channels=15,out_channels=64,kernel_size=[7,7,7],padding=3)
        self.sccov1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=[7,7,7],padding=3)
        self.sccov2 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=[7,7,7],padding=3)
        self.sccov3 = nn.Conv3d(in_channels=64,out_channels=15,kernel_size=[7,7,7],padding=3)
        self.downsample = nn.MaxPool3d(kernel_size=4)
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)

    def forward(self,x):
        local_pred = self.unet(x).cuda(1)
        spatial_pred = self.downsample(local_pred)
        spatial_pred = self.sccov0(spatial_pred)
        spatial_pred = self.sccov1(spatial_pred)
        spatial_pred = self.sccov2(spatial_pred)
        spatial_pred = self.sccov3(spatial_pred)
        spatial_pred = self.upsample(spatial_pred)
        prediction = spatial_pred*local_pred
        return prediction
