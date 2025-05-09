import sys
import math
import random

import numpy as np
import torch
import torch.nn as nn
import numpy.linalg as LA
import torch.nn.functional as F

from conic_anchor import ConicConv_test as ConicConv
from conic_anchor import ConicConv37_test as ConicConv37
from conic_anchor import ConicConv73_test as ConicConv73

multires=[0.0051941870036646, 0.02004838034795, 0.0774278195486317, 0.299564810864565]

class VanishingNet(nn.Module):
    def __init__(self, backbone, output_stride=4, upsample_scale=1):
        super().__init__()
        self.backbone = backbone
        self.anet = ApolloniusNet(output_stride, upsample_scale)
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        #multires=[0.0051941870036646, 0.02004838034795, 0.0774278195486317, 0.299564810864565]

    def forward(self, x,vps):
        x = self.backbone(x)[0]
        N, _, H, W = x.shape
        c = len(vps)
        x = x[:, None].repeat(1, c, 1, 1, 1).reshape(N * c, _, H, W)
        vpts = [to_pixel(v) for v in vps]
        vpts = torch.tensor(vpts, device=x.device)
        return self.anet(x, vpts).sigmoid()

class ApolloniusNet(nn.Module):
    def __init__(self, output_stride, upsample_scale):
        super().__init__()
        self.fc0 = nn.Conv2d(64, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = ConicConv(32, 64,(3,3))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = ConicConv(64, 128,(3,3))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = ConicConv(128, 256,(3,3))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = ConicConv(256, 256,(3,3))
        
        self.conv12 = ConicConv37(32, 64,(3,7))
        self.bn22 = nn.BatchNorm2d(64)
        self.conv22 = ConicConv37(64, 128,(3,7))
        self.bn32 = nn.BatchNorm2d(128)
        self.conv32 = ConicConv37(128, 256,(3,7))
        self.bn42 = nn.BatchNorm2d(256)
        self.conv42 = ConicConv37(256, 256,(3,7))
        
        self.conv13 = ConicConv73(32, 64,(7,3))
        self.bn23 = nn.BatchNorm2d(64)
        self.conv23 = ConicConv73(64, 128,(7,3))
        self.bn33 = nn.BatchNorm2d(128)
        self.conv33 = ConicConv73(128, 256,(7,3))
        self.bn43 = nn.BatchNorm2d(256)
        self.conv43 = ConicConv73(256, 256,(7,3))
        
        self.bn1x1 = nn.BatchNorm2d(768)
        self.conv1x1 = nn.Conv2d(768,256,1)

        self.fc1 = nn.Linear(16384, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, len(multires))

        self.upsample_scale = upsample_scale
        self.stride = output_stride / upsample_scale

    def forward(self, input, vpts):
        # for now we did not do interpolation
        if self.upsample_scale != 1:
            input = F.interpolate(input, scale_factor=self.upsample_scale)
        x = self.fc0(input)

        # 128
        x = self.bn1(x)
        xin = self.relu(x)
        
        ## 3x3
        x = self.conv1(xin, vpts / self.stride - 0.5)
        x = self.pool(x)
        # 64
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x, vpts / self.stride / 2 - 0.5)
        x = self.pool(x)
        # 32
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x, vpts / self.stride / 4 - 0.5)
        x = self.pool(x)
        # 16
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4(x, vpts / self.stride / 8 - 0.5)
        x33out = self.pool(x)
        
        ## 3x7
        x = self.conv12(xin, vpts / self.stride - 0.5)
        x = self.pool(x)
        # 64
        x = self.bn22(x)
        x = self.relu(x)
        x = self.conv22(x, vpts / self.stride / 2 - 0.5)
        x = self.pool(x)
        # 32
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv32(x, vpts / self.stride / 4 - 0.5)
        x = self.pool(x)
        # 16
        x = self.bn42(x)
        x = self.relu(x)
        x = self.conv42(x, vpts / self.stride / 8 - 0.5)
        x37out = self.pool(x)
        
        ## 7x3
        x = self.conv13(xin, vpts / self.stride - 0.5)
        x = self.pool(x)
        # 64
        x = self.bn23(x)
        x = self.relu(x)
        x = self.conv23(x, vpts / self.stride / 2 - 0.5)
        x = self.pool(x)
        # 32
        x = self.bn33(x)
        x = self.relu(x)
        x = self.conv33(x, vpts / self.stride / 4 - 0.5)
        x = self.pool(x)
        # 16
        x = self.bn43(x)
        x = self.relu(x)
        x = self.conv43(x, vpts / self.stride / 8 - 0.5)
        x73out = self.pool(x)
        
        x=torch.cat((x33out,x37out,x73out),1)
        x = self.bn1x1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        # 8
        x = x.view(x.shape[0], -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


def sample_sphere(v, theta0, theta1):
    costheta = random.uniform(math.cos(theta1), math.cos(theta0))
    phi = random.random() * math.pi * 2
    v1 = orth(v)
    v2 = np.cross(v, v1)
    r = math.sqrt(1 - costheta ** 2)
    w = v * costheta + r * (v1 * math.cos(phi) + v2 * math.sin(phi))
    return w / LA.norm(w)


def to_label(w, vpts):
    degree = np.min(np.arccos(np.abs(vpts @ w).clip(max=1)))
    return [int(degree < res + 1e-6) for res in multires]


def to_pixel(w):
    x = w[0] / w[2] * 1.0 * 256 + 256
    y = -w[1] / w[2] * 1.0 * 256 + 256
    return y, x