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

class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/train_A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/train_B' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/train_C' % mode) + '/*.*'))
        
    def __getitem__(self, index):
        k=random.randint(0,100)
        
        item_A=io.imread(self.files_A[index % len(self.files_A)])
        ori_x=item_A.shape[0]
        ori_y=item_A.shape[1]
        if k>50:
            item_A=np.fliplr(item_A)
        item_A=resize(item_A,(500,500,3))
        
        item_B=io.imread(self.files_B[index % len(self.files_B)])
        if k>50:
            item_B=np.fliplr(item_B)
        item_B=resize(item_B,(500,500,1))
        item_A=item_A*item_B
        item_A=2.0*(np.asarray(item_A))-1.0
        item_A=torch.from_numpy(item_A.copy()).float()
        item_A=item_A.view(500,500,3)
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_B=2.0*(np.asarray(item_B))-1.0
        item_B=torch.from_numpy(item_B.copy()).float()
        item_B=item_B.view(500,500,1)
        item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()
 
        x,y=[],[]
        f = open(self.files_C[index % len(self.files_C)],"r")
        for eachline in f:
            tmp = eachline.split()
            x=(np.asarray(tmp[1],dtype=float))
            x=x*500.0/ori_x
            y=(np.asarray(tmp[0],dtype=float))
            y=y*500.0/ori_y
            if k>50:
                y=500.0-y
        f.close()
        item_C=[x,y]
        item_C=np.asarray(item_C)
        item_C=torch.from_numpy(item_C.copy()).float()
        item_C=item_C.view(2)
        return {'A': item_A,'B': item_B,'C': item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C))

