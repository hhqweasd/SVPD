import sys
import math
import random

import numpy as np
import torch
import torch.nn as nn
import numpy.linalg as LA
import torch.nn.functional as F

from conic import ConicConv

multires=[0.0051941870036646, 0.02004838034795, 0.0774278195486317, 0.299564810864565]

class CA(nn.Module):
    def __init__(self, channels, reduction=4):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class VanishingNet(nn.Module):
    def __init__(self, backbone, output_stride=4, upsample_scale=1):
        super().__init__()
        self.backbone = backbone
        self.anet = ApolloniusNet(output_stride, upsample_scale)
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        #multires=[0.0051941870036646, 0.02004838034795, 0.0774278195486317, 0.299564810864565]

    def forward(self, x,vps,mask,IS_mask):
        x = self.backbone(x,mask,IS_mask)[0]
        N, _, H, W = x.shape
        c = 3 + 1 * len(multires) * (1 + 1)
        x = x[:, None].repeat(1, c, 1, 1, 1).reshape(N * c, _, H, W)
        vpts_gt = vps.cpu().numpy()
        vpts, y = [], []
        for n in range(N):
            def add_sample(p):
                vpts.append(to_pixel(p))
                y.append(to_label(p, vpts_gt[n]))

            for vgt in vpts_gt[n]:
                for st, ed in zip([0] + multires[:-1], multires):
                    # positive samples
                    for _ in range(1):
                        add_sample(sample_sphere(vgt, st, ed))
                    # negative samples
                    for _ in range(1):
                        add_sample(sample_sphere(vgt, ed, ed * 2))
            # random samples
            for _ in range(3):
                add_sample(sample_sphere(np.array([0, 0, 1]), 0, math.pi / 2))

        y = torch.tensor(y, device=x.device, dtype=torch.float)
        vpts = torch.tensor(vpts, device=x.device)

        x,x8,x16,x32 = self.anet(x, vpts)
        L = self.loss(x, y)+self.loss(x8, y)+self.loss(x16, y)+self.loss(x32, y)
        maskn = (y == 0).float()
        maskp = (y == 1).float()
        losses = {}
        for i in range(len(multires)):
            #assert maskn[:, i].sum().item() != 0
            #assert maskp[:, i].sum().item() != 0
            losses[f"lneg{i}"] = (L[:, i] * maskn[:, i]).sum() / (maskn[:, i].sum()+1e-8)
            losses[f"lpos{i}"] = (L[:, i] * maskp[:, i]).sum() / (maskp[:, i].sum()+1e-8)

        return {
            "losses": [losses],
            "preds": {"vpts": vpts, "scores": x.sigmoid(), "ys": y},
        }


class ApolloniusNet(nn.Module):
    def __init__(self, output_stride, upsample_scale):
        super().__init__()
        self.fc0 = nn.Conv2d(64, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = ConicConv(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = ConicConv(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = ConicConv(128, 256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = ConicConv(256, 256)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, len(multires))
        
        
        self.bn_32 = nn.BatchNorm2d(4)
        self.conv_32 = ConicConv(64, 4, 1)
        self.fc1_32 = nn.Linear(4096, 1024)
        self.fc2_32 = nn.Linear(1024, 1024)
        self.fc3_32 = nn.Linear(1024, len(multires))
        
        self.bn_16 = nn.BatchNorm2d(16)
        self.conv_16 = ConicConv(128, 16, 1)
        self.fc1_16 = nn.Linear(4096, 1024)
        self.fc2_16 = nn.Linear(1024, 1024)
        self.fc3_16 = nn.Linear(1024, len(multires))
        
        self.bn_8 = nn.BatchNorm2d(64)
        self.conv_8 = ConicConv(128, 64, 1)
        self.fc1_8 = nn.Linear(4096, 1024)
        self.fc2_8 = nn.Linear(1024, 1024)
        self.fc3_8 = nn.Linear(1024, len(multires))
        
        self.upsample_scale = upsample_scale
        self.stride = output_stride / upsample_scale

    def forward(self, input, vpts):
        # for now we did not do interpolation
        if self.upsample_scale != 1:
            input = F.interpolate(input, scale_factor=self.upsample_scale)
        x = self.fc0(input)

        # 128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x, vpts / self.stride - 0.5)
        x = self.pool(x)
        
        x32 = self.bn_32(x)
        x32 = self.relu(x32)
        x32 = self.conv_32(x32, vpts / self.stride / 2 - 0.5)
        x32 = x32.view(x32.shape[0], -1)
        x32 = self.relu(x32)
        x32 = self.fc1_32(x32)
        x32 = self.relu(x32)
        x32 = self.fc2_32(x32)
        x32 = self.relu(x32)
        x32 = self.fc3_32(x32)
        
        
        
        # 64
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x, vpts / self.stride / 2 - 0.5)
        x = self.pool(x)
        
        
        x16 = self.bn_16(x)
        x16 = self.relu(x16)
        x16 = self.conv_16(x16, vpts / self.stride / 4 - 0.5)
        x16 = x16.view(x16.shape[0], -1)
        x16 = self.relu(x16)
        x16 = self.fc1_16(x16)
        x16 = self.relu(x16)
        x16 = self.fc2_16(x16)
        x16 = self.relu(x16)
        x16 = self.fc3_16(x16)
        
        # 32
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x, vpts / self.stride / 4 - 0.5)
        x = self.pool(x)
        
        
        x8 = self.bn_8(x)
        x8 = self.relu(x8)
        x8 = self.conv_8(x8, vpts / self.stride / 8 - 0.5)
        x8 = x8.view(x8.shape[0], -1)
        x8 = self.relu(x8)
        x8 = self.fc1_8(x8)
        x8 = self.relu(x8)
        x8 = self.fc2_8(x8)
        x8 = self.relu(x8)
        x8 = self.fc3_8(x8)
        
        # 16
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4(x, vpts / self.stride / 8 - 0.5)
        x = self.pool(x)
        # 8
        x = x.view(x.shape[0], -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x,x8,x16,x32


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
    x = w[0] / w[2] * 1.0 * 128 + 128
    y = -w[1] / w[2] * 1.0 * 128 + 128
    return y, x