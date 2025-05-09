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

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/test_A_512' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/test_B_512' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/test_C' % mode) + '/*.*'))
        
    def __getitem__(self, index):
        item_A=io.imread(self.files_A[index % len(self.files_A)])
        ori_x=3024.0
        ori_y=4032.0 
        item_B=color.rgb2gray(io.imread(self.files_B[index % len(self.files_B)]))
        #item_A=item_A*item_B
        item_A=np.asarray(item_A)/255.0
        item_A=torch.from_numpy(item_A.copy()).float()
        item_A=item_A.view(512,512,3)
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_B=np.asarray(item_B)
        item_B=torch.from_numpy(item_B.copy()).float()
        item_B=item_B.view(512,512,1)
        item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()
 
        x,y=[],[]
        f = open(self.files_C[index % len(self.files_C)],"r")
        for eachline in f:
            tmp = eachline.split()
            x=(np.asarray(tmp[1],dtype=float))
            x=x*512.0/ori_x
            y=(np.asarray(tmp[0],dtype=float))
            y=y*512.0/ori_y
        f.close()
        item_C=np.array([[x/256-1.0,1.0-y/256,1.0]])# 为什么y取负（宽），不知道
        item_C[0] /= LA.norm(item_C[0])
        item_C=torch.from_numpy(item_C.copy()).float()
        item_D=self.files_A[index % len(self.files_A)]
        return item_A,item_B,item_C,item_D
        # 图像是否需要归一化？不要，因为bce输入要是01之间
        # 就不裁剪了，怕把影子减掉
        
        
    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C))

