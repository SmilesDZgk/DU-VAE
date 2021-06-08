import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import *
from .encoder import GaussianEncoderBase

"""
A better ResNet baseline
"""


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1, bias=False)


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 deconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1,
                              output_padding=output_padding, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = nn.ELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.downsample = downsample
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        residual = x if self.downsample is None else self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.activation(out + residual)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out


class ResNet(nn.Module):
    def __init__(self, inplanes, planes, strides):
        super(ResNet, self).__init__()
        assert len(planes) == len(strides)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = ResNetBlock(inplanes, plane, stride=stride)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main(x)


class ResNetEncoderV2(GaussianEncoderBase):
    def __init__(self, args, ngpu=1):
        super(ResNetEncoderV2, self).__init__()
        self.ngpu = ngpu
        self.nz = args.nz
        self.nc = 1
        hidden_units = 512
        self.main = nn.Sequential(
            ResNet(self.nc, [64, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, hidden_units, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ELU(),
        )
        self.linear = nn.Linear(hidden_units, 2 * self.nz)
        self.reset_parameters()
        self.delta_rate = args.delta_rate

        self.args = args

    def reset_parameters(self):
        for m in self.main.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = self.linear(output.view(output.size()[:2]))
        mu, logvar = output.chunk(2, 1)

        return mu, logvar


class BNResNetEncoderV2(GaussianEncoderBase):
    def __init__(self, args, ngpu=1):
        super(BNResNetEncoderV2, self).__init__()
        self.ngpu = ngpu
        self.nz = args.nz
        self.nc = 1
        hidden_units = 512
        self.main = nn.Sequential(
            ResNet(self.nc, [64, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, hidden_units, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ELU(),
        )
        self.linear = nn.Linear(hidden_units, 2 * self.nz)
        self.mu_bn = nn.BatchNorm1d(args.nz)
        self.gamma = args.gamma
        self.args = args

        self.reset_parameters()
        self.delta_rate = args.delta_rate

    def reset_parameters(self, reset=False):
        for m in self.main.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        if not reset:
            nn.init.constant_(self.mu_bn.weight, self.gamma)
        else:
            print('reset bn!')

            nn.init.constant_(self.mu_bn.weight, self.gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = self.linear(output.view(output.size()[:2]))
        mu, logvar = output.chunk(2, 1)
        if self.args.gamma > 0:
            self.mu_bn.weight.requires_grad = True
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            #if ss < self.gamma:
            self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss

            mu = self.mu_bn(mu.squeeze(0))
        else:
            mu = mu.squeeze(0)

        if self.args.kl_weight == 1:
            logvar = torch.log(torch.exp(logvar) + self.delta_rate * 1.0 / (2 * math.e * math.pi))

        if torch.rand(1).sum() <= 0.001:
            scale = torch.exp(logvar / 2)
            # print('gamma', self.mu_bn.weight)
            # print('train loc mean', torch.mean(mu, dim=0))
            # print('train scale std', torch.std(scale, dim=0))
            # print('train scale mean', torch.mean(scale, dim=0))

        return mu, logvar

