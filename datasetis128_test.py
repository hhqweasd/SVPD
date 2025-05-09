import glob
import random
import os

from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch
import string
import numpy.linalg as LA

class ImageDataset(Dataset):
    def __init__(self, root,mode='test'):
        if mode=='test':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/test_A_256' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/test_B_256' % mode) + '/*.*'))
            self.files_C = sorted(glob.glob(os.path.join(root, '%s/test_C' % mode) + '/*.*'))
        
    def __getitem__(self, index):
        k=random.randint(0,100)
        kk=random.randint(0,100)
        item_A=io.imread(self.files_A[index % len(self.files_A)])
        ori_x=3024.0
        ori_y=4032.0
        item_B=color.rgb2gray(io.imread(self.files_B[index % len(self.files_B)]))
        item_A=np.asarray(item_A)/255.0
        item_A=torch.from_numpy(item_A.copy()).float()
        item_A=item_A.resize_(128,128,3)
        item_A=item_A.view(128,128,3)
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_B=np.asarray(item_B)
        item_B=torch.from_numpy(item_B.copy()).float()
        item_B=item_B.resize_(128,128,1)
        item_B=item_B.view(128,128,1)
        item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()
 
        x,y=[],[]
        f = open(self.files_C[index % len(self.files_C)],"r")
        for eachline in f:
            tmp = eachline.split()
            x=(np.asarray(tmp[1],dtype=float))
            x=x*128.0/ori_x
            y=(np.asarray(tmp[0],dtype=float))
            y=y*128.0/ori_y
        f.close()
        item_C=np.array([[x/64-1.0,1.0-y/64,1.0]])# 为什么y取负（宽），因为python的y轴从上到下
        item_C[0] /= LA.norm(item_C[0])
        item_C=torch.from_numpy(item_C.copy()).float()
        return item_A,item_B,item_C
        # 图像是否需要归一化？
        # 就不裁剪了，怕把影子减掉
        
        
    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C))

