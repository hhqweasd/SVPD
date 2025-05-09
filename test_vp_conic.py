from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
from PIL import Image
import torch
import numpy as np
from skimage import io
from skimage.transform import resize

from hourglass_pose import hg
from model_vp_conic_test import VanishingNet
import numpy.linalg as LA
import math
from datasets_vp4_test import ImageDataset
from torch.utils.data import DataLoader

import scipy.io as scipyio
import matplotlib.pyplot as plt

def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold

os.environ["CUDA_VISIBLE_DEVICES"]="2,4,3,0,7,5,6,1"
torch.manual_seed(628)

multires=[0.0051941870036646, 0.02004838034795, 0.0774278195486317, 0.299564810864565]

def sample_sphere(v, alpha, num_pts):
    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o
    
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_1', type=str, default='model_vp_conic/vpnet_100.pth', help='A2B generator checkpoint file')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
opt = parser.parse_args()

opt.dataroot = '/home/liuzhihao/dataset/vpdataset'
if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)

vpnet=hg(planes=64, depth=4, num_stacks=1, num_blocks=1)
vpnet=VanishingNet(vpnet,4,1)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vpnet=vpnet.to(device)
vpnet=torch.nn.DataParallel(vpnet,device_ids=[0,1])
vpnet.load_state_dict(torch.load(opt.generator_1))
vpnet.eval()

err = []
errdist = []
imgname = []
dataloader=DataLoader(ImageDataset(opt.dataroot),batch_size=opt.batchSize,shuffle=False,num_workers=opt.n_cpu)

for i, (real_nsr,mask,realco,name) in enumerate(dataloader):
    real_nsr=real_nsr.cuda()
    vpts_gt = realco[0]
    vpts_gt *= (vpts_gt[:, 2:3] > 0).float() * 2 - 1
    vpts = sample_sphere(np.array([0, 0, 1]), np.pi / 2, 64)
    with torch.no_grad():
        score = vpnet(real_nsr,vpts)[:, -1].cpu().numpy()
    index = np.argsort(-score)
    candidate = [index[0]]
    for i in index[1:]:
        if len(candidate) == 1:
            break
        dst = np.min(np.arccos(np.abs(vpts[candidate] @ vpts[i])))
        if dst < np.pi / 1:
            continue
        candidate.append(i)
    vpts_pd = vpts[candidate]

    for res in range(1, len(multires)):
        vpts = [sample_sphere(vpts_pd[vp], multires[-res], 64) for vp in range(1)]
        vps = np.vstack(vpts)
        with torch.no_grad():
            score = vpnet(real_nsr,vps)[:, -res - 1].cpu().numpy().reshape(1, -1)
        for i, s in enumerate(score):
            vpts_pd[i] = vpts[i][np.argmax(s)]
    
    
    for vp in vpts_gt.numpy():
        angle=min(np.arccos(np.abs(vpts_pd @ vp).clip(max=1))) / np.pi * 180
        #print(angle)
        err.append(angle)
        
    # 单点距离估计
    # set the focal length to the half of the sensor width (image width)
    # using focal length to evaluate the distance
    vpts_gt=vpts_gt/vpts_gt[:,2]
    vpts_gt[:,0]=(vpts_gt[:,0]+1.0)*256
    vpts_gt[:,1]=(vpts_gt[:,1]+1.0)*256
    focal_length=4032/2    
    vpts_pd=vpts_pd/vpts_pd[:,2]
    vpts_pd[:,0]=(vpts_pd[:,0]+1)*256
    vpts_pd[:,1]=(vpts_pd[:,1]+1)*256
    dist=torch.pow(torch.pow(vpts_gt[:,0]-torch.from_numpy(vpts_pd[:,0].copy()).float(),2)+torch.pow(vpts_gt[:,1]-torch.from_numpy(vpts_pd[:,1].copy()).float(),2),0.5)
    dist=dist/focal_length
    errdist.append(dist)
    #print(dist)
    imgname.append(name)

#err = np.sort(np.array(err))
err = np.array(err)
np.savez('./model_vp_conic/angle_100.npz', err=err)
scipyio.savemat('model_vp_conic/err.mat', {'name': err})
errdist = np.array(errdist)
np.savez('./model_vp_conic/dist_100.npz', errdist=errdist)
scipyio.savemat('model_vp_conic/dist_100.mat', {'name': errdist})
scipyio.savemat('model_vp_conic/imgname_100.mat', {'name': imgname})
