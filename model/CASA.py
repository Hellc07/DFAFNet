import torch
from torch import nn
import torch.nn.functional as F


class MMFI(nn.Module):
    """Multi Modal Feature Interaction"""
    def __init__(self, in_ch):
        super(MMFI, self).__init__()

        self.ca = CA(in_ch)
        self.sa_1 = SA(kernel_size=3)
        self.sa_2 = SA(kernel_size=3)

    def forward(self, x, y):
        channel_map = self.ca(x, y)
        x_ca = x*channel_map
        y_ca = y*channel_map

        x_sa = self.sa_1(x_ca)
        y_sa = self.sa_2(y_ca)

        xy_sa = torch.cat((x_sa, y_sa), dim=1)
        xy_sa = F.softmax(xy_sa, dim=1)
        x_sa_map = xy_sa[:, 0, :, :]
        y_sa_map = xy_sa[:, 1, :, :]
        x_sa_map = x_sa_map.unsqueeze(1)
        y_sa_map = y_sa_map.unsqueeze(1)

        out = x_sa_map*x + y_sa_map*y
        # out = torch.matmul(x_sa_map, x) + torch.matmul(y_sa_map, y)
        # print(out.shape)
        return out


class CA(nn.Module):
    """cs: channel Attention"""
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        input_ch = in_ch * 2
        self.avg_fc1 = nn.Conv2d(input_ch, input_ch//16, 1, bias=False)
        self.avg_relu = nn.ReLU(inplace=True)
        self.avg_fc2 = nn.Conv2d(input_ch//16, in_ch, 1, bias=False)

        self.max_fc1 = nn.Conv2d(input_ch, input_ch//16, 1, bias=False)
        self.max_relu = nn.ReLU(inplace=True)
        self.max_fc2 = nn.Conv2d(input_ch//16, in_ch, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        avg_xy = self.avgpool(xy)
        max_xy = self.maxpool(xy)

        avg_xy = self.avg_relu(self.avg_fc1(avg_xy))
        max_xy = self.max_relu(self.max_fc1(max_xy))

        avg_xy_back = avg_xy
        max_xy_back = max_xy

        avg_xy = avg_xy + max_xy_back
        max_xy = max_xy + avg_xy_back

        avg_xy = self.avg_fc2(avg_xy)
        max_xy = self.max_fc2(max_xy)

        channel_map = self.sigmoid((avg_xy+max_xy))

        return channel_map


class SA(nn.Module):
    """Spatial Attention"""
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x