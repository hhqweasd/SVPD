from __future__ import print_function
import os
import torchvision
import torch
import numpy as np
from skimage import io
from skimage.transform import resize
import numpy.linalg as LA
import math
from torch.utils.data import DataLoader
import scipy.io as scipyio
from datasets_test import ImageDataset_IS
from hourglass_pose_with2mask import hg

from model_vp_conic_test_256 import VanishingNet

os.environ["CUDA_VISIBLE_DEVICES"]="2,3,5,4,7,6,1"

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

mode='test'
dataroot = '/home/liuzhihao/dataset/vpdataset'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vpnet=hg(planes=64, depth=4, num_stacks=1, num_blocks=1)
vpnet=VanishingNet(vpnet,4,1)
vpnet=vpnet.to(device)
vpnet=torch.nn.DataParallel(vpnet,device_ids=[0,1])

for ee in range(100,90,-1):
    generator_1='model_vpnet_with2mask/vpnet_%s.pth'%(ee)
    vpnet.load_state_dict(torch.load(generator_1))
    vpnet.eval()

    err = []
    errdist = []
    dataloader=DataLoader(ImageDataset_IS(dataroot,mode),batch_size=1,shuffle=False,num_workers=2)

    for i, (real_nsr,mask,realco, mask_IS) in enumerate(dataloader):
        real_nsr=real_nsr.cuda()
        mask=mask.cuda()
        mask_IS=mask_IS.cuda()
        vpts_gt = realco[0]
        vpts_gt *= (vpts_gt[:, 2:3] > 0).float() * 2 - 1
        vpts = sample_sphere(np.array([0, 0, 1]), np.pi / 2, 64)
        with torch.no_grad():
            score = vpnet(torch.cat((real_nsr,mask,mask_IS),1),vpts)[:, -1].cpu().numpy()
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
                score = vpnet(torch.cat((real_nsr,mask,mask_IS),1),vps)[:, -res - 1].cpu().numpy().reshape(1, -1)
            for i, s in enumerate(score):
                vpts_pd[i] = vpts[i][np.argmax(s)]
        
        for vp in vpts_gt.numpy():
            angle=min(np.arccos(np.abs(vpts_pd @ vp).clip(max=1))) / np.pi * 180
            err.append(angle)
            
        # 单点距离估计
        # set the focal length to the half of the sensor width (image width)
        # using focal length to evaluate the distance
        vpts_gt=vpts_gt/vpts_gt[:,2]
        vpts_gt[:,0]=(vpts_gt[:,0]+1.0)*128*3024/256
        vpts_gt[:,1]=(1-vpts_gt[:,1])*128*4032/256
        focal_length=4032/2    
        vpts_pd=vpts_pd/vpts_pd[:,2]
        vpts_pd[:,0]=(vpts_pd[:,0]+1)*128*3024/256
        vpts_pd[:,1]=(1-vpts_pd[:,1])*128*4032/256
        dist=torch.pow(torch.pow(vpts_gt[:,0]-torch.from_numpy(vpts_pd[:,0].copy()).float(),2)+torch.pow(vpts_gt[:,1]-torch.from_numpy(vpts_pd[:,1].copy()).float(),2),0.5)
        dist=dist/focal_length
        errdist.append(dist)
        log = 'GT: (%d,%d), Pre (%d,%d), dist %.2f, Angle %.2f,'%(vpts_gt[0,1],vpts_gt[0,0],vpts_pd[0,1],vpts_pd[0,0],dist,angle)
        print(log)
        print('\n')
    
    err = np.array(err)
    nnn='model_vpnet_with2mask/err_%s.mat'%(ee)
    scipyio.savemat(nnn, {'name': err})
    errdist = np.array(errdist)
    nnn='model_vpnet_with2mask/dist_%s.mat'%(ee)
    scipyio.savemat(nnn, {'name': errdist})
