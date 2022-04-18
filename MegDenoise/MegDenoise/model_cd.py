#!/usr/bin/env python3
from collections import OrderedDict
import numpy as np
import megengine as mge
import megengine.module as M
import megengine.functional as F


def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = M.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = M.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = M.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = M.LeakyReLU(0.15)

    return M.Sequential(modules)


class EncoderBlock(M.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride,
                            padding=2, is_seperable=True, has_relu=True)
#         self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=3, stride=1,
#                             padding=1, is_seperable=False, has_relu=False)
#         self.conv3 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1,
#                             padding=1, is_seperable=True, has_relu=False)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=True, has_relu=False)

        self.proj = (
            M.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=True, has_relu=False)
        )
        self.relu = M.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)
#         x = self.conv3(x)
        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//2,
            out_channels=out_channels,
            stride=2,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//2,
                out_channels=out_channels,
                stride=1,
            )
        )

    return M.Sequential(*blocks)


class DecoderBlock(M.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=True,
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=False,
        )

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(M.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3)
        self.upsample = M.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True, has_relu=True)

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y


class Network(M.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = Conv2D(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=16, out_channels=64, num_blocks=2)
        self.enc2 = EncoderStage(in_channels=64, out_channels=126, num_blocks=3)

        self.encdec = Conv2D(in_channels=126, out_channels=32, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)

        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.dec4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def forward(self, inp):
        n, c, h, w = inp.shape
#         inp = inp.reshape((n, c, h // 2, 2, w // 2, 2)).transpose((0, 1, 3, 5, 2, 4)).reshape((n, c * 4, h // 2, w // 2))
        img_r = inp[:, :, 0:h:2, 0:w:2]
        img_g1 = inp[:, :, 0:h:2, 1:w:2]
        img_g2 = inp[:, :, 1:h:2, 0:w:2]
        img_b = inp[:, :, 1:h:2, 1:w:2]
        inp = mge.tensor(np.concatenate((img_r, img_g1, img_g2, img_b), axis=1))
        
        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        
        conv3 = self.encdec(conv2)

        up1 = self.dec3((conv3, conv1))
        x = self.dec4((up1, conv0))
        x = self.out0(x)
        x = self.out1(x)
        
        pred = inp + x
        out = mge.tensor(np.zeros((n, c, h, w)))
        out[:, 0, 0:h:2, 0:w:2] = pred[:, 0]
        out[:, 0, 0:h:2, 1:w:2] = pred[:, 1]
        out[:, 0, 1:h:2, 0:w:2] = pred[:, 2]
        out[:, 0, 1:h:2, 1:w:2] = pred[:, 3]
#         pred = pred.reshape((n, c, 2, 2, h // 2, w // 2)).transpose((0, 1, 4, 2, 5, 3)).reshape((n, c, h, w))
        return out

