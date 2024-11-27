from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class Attention_Fuse(nn.Module):
    def __init__(self, in_ch, size):
        super(Attention_Fuse, self).__init__()

        self.weight_conv_fuse = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.weight_conv_depth = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.sigmoid_fuse = nn.Sigmoid()
        self.sigmoid_depth = nn.Sigmoid()

    def forward(self, feature_RGB, feature_D):
        # Compute the Activate_Map
        Activate_Map = feature_D * feature_RGB
        Max_Map, _ = torch.max(Activate_Map, dim=1, keepdim=True)
        Avg_Map = torch.mean(Activate_Map, dim=1, keepdim=True)
        Response_Map = torch.cat([Max_Map, Avg_Map], 1)

        # Compute the Activate_Map of depth
        Max_Map_D, _ = torch.max(feature_D, dim=1, keepdim=True)
        Avg_Map_D = torch.mean(feature_D, dim=1, keepdim=True)
        Response_Map_D = torch.cat([Max_Map_D, Avg_Map_D], 1)


        # Generate the Weighted_Map and Ignore_Map
        Weighted_Map = self.sigmoid_fuse(self.weight_conv_fuse(Response_Map))
        Depth_Map = self.sigmoid_depth(self.weight_conv_depth(Response_Map_D))
        Ignore_Map = 1 - Weighted_Map

        # Generate the select_map
        Select_map_depth = Depth_Map * Ignore_Map

        # Fuse the feature
        Map = torch.cat([Weighted_Map, Select_map_depth], 1)
        Map = F.softmax(Map, dim=1)
        Weighted_Map = Map[:, 0, :, :]
        Select_map_depth = Map[:, 1, :, :]
        Weighted_Map = Weighted_Map.unsqueeze(1)
        Select_map_depth = Select_map_depth.unsqueeze(1)

        new_feature_rgb = Weighted_Map * feature_RGB
        new_feature_depth = Select_map_depth * feature_D
        Fuse_feature = new_feature_depth + new_feature_rgb

        return Fuse_feature


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch_rgb=3, in_ch_d=1, out_ch=5):
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1_rgb = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_rgb = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_rgb = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_rgb = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_rgb = conv_block(in_ch_rgb, filters[0])
        self.Conv2_rgb = conv_block(filters[0], filters[1])
        self.Conv3_rgb = conv_block(filters[1], filters[2])
        self.Conv4_rgb = conv_block(filters[2], filters[3])
        self.Conv5_rgb = conv_block(filters[3], filters[4])

        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block(in_ch_d, filters[0])
        self.Conv2_d = conv_block(filters[0], filters[1])
        self.Conv3_d = conv_block(filters[1], filters[2])
        self.Conv4_d = conv_block(filters[2], filters[3])
        self.Conv5_d = conv_block(filters[3], filters[4])

        self.fuse_1 = Attention_Fuse(in_ch=32, size=512)
        self.fuse_2 = Attention_Fuse(in_ch=64, size=256)
        self.fuse_3 = Attention_Fuse(in_ch=128, size=128)
        self.fuse_4 = Attention_Fuse(in_ch=256, size=64)
        self.fuse_5 = Attention_Fuse(in_ch=512, size=32)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb, d):
        e1_rgb = self.Conv1_rgb(rgb)
        e1_d = self.Conv1_d(d)
        fuse1 = self.fuse_1(e1_rgb, e1_d)

        e2_rgb = self.Maxpool1_rgb(e1_rgb + fuse1)
        e2_rgb = self.Conv2_rgb(e2_rgb)
        e2_d = self.Maxpool1_d(e1_d + fuse1)
        e2_d = self.Conv2_d(e2_d)
        fuse2 = self.fuse_2(e2_rgb, e2_d)

        e3_rgb = self.Maxpool2_rgb(e2_rgb + fuse2)
        e3_rgb = self.Conv3_rgb(e3_rgb)
        e3_d = self.Maxpool2_d(e2_d + fuse2)
        e3_d = self.Conv3_d(e3_d)
        fuse3 = self.fuse_3(e3_rgb, e3_d)

        e4_rgb = self.Maxpool3_rgb(e3_rgb + fuse3)
        e4_rgb = self.Conv4_rgb(e4_rgb)
        e4_d = self.Maxpool3_d(e3_d + fuse3)
        e4_d = self.Conv4_d(e4_d)
        fuse4 = self.fuse_4(e4_rgb, e4_d)

        e5_rgb = self.Maxpool4_rgb(e4_rgb + fuse4)
        e5_rgb = self.Conv5_rgb(e5_rgb)
        e5_d = self.Maxpool4_d(e4_d + fuse4)
        e5_d = self.Conv5_d(e5_d)
        fuse5 = self.fuse_5(e5_rgb, e5_d)

        d5 = self.Up5(e5_rgb + fuse5)
        d5 = torch.cat((e4_rgb + fuse4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3_rgb + fuse3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2_rgb + fuse2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1_rgb + fuse1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


def AttBaseline(in_ch_rgb, in_ch_d, out_ch):
    model = U_Net(in_ch_rgb=in_ch_rgb, in_ch_d=in_ch_d, out_ch=out_ch)
    return model