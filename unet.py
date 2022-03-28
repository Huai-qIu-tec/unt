import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

# 定义连续两次的Conv层
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels == None:
            mid_channels = out_channels

        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )


class Up(nn.Module):
    """
    这里的in_channels是指concat之后的in_channels
    这里之所以会出现这些情况：采用双线性插值bilinear之后的mid_channels=in_channels // 2
        而采用转置卷积的mid_channels=out_channels
        是因为本项目采用的双线性插值和原论文中的具体实现思路不同
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    # forward函数实现对上采样的channels和对应层数的Decoder中的通道进行concat,
    # 需要注意的是在原图像进行MaxPooling之后可能会在边缘像素大小进行向下取整, 如 7 // 2 = 3, 因此需要进行padding
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x.shape = [Batch, Channels, Height, Width]

        diff_x = x2.size()[3] - x1.size()[3]
        diff_y = x2.size()[2] - x1.size()[2]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, pad=[diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, bilinear=True, base_channels=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels=in_channels, out_channels=base_channels)
        self.down1 = Down(in_channels=base_channels, out_channels=base_channels * 2)
        self.down2 = Down(in_channels=base_channels * 2, out_channels=base_channels * 4)
        self.down3 = Down(in_channels=base_channels * 4, out_channels=base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(in_channels=base_channels * 8, out_channels=base_channels * 16 // factor)
        self.up1 = Up(in_channels=base_channels * 16, out_channels=base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = Up(in_channels=base_channels * 8, out_channels=base_channels * 4 // factor, bilinear=bilinear)
        self.up3 = Up(in_channels=base_channels * 4, out_channels=base_channels * 2 // factor, bilinear=bilinear)
        self.up4 = Up(in_channels=base_channels * 2, out_channels=base_channels, bilinear=bilinear)
        self.out_conv = OutConv(in_channels=base_channels, num_classes=num_classes)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}

