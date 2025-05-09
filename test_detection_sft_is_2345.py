from __future__ import print_function
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import scipy.io as scipyio

from dataset_vpnet_test_name import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"]="6,2,7,5,4,3,0,1"

mode='test'
dataroot = '/home/liuzhihao/dataset/vpdatasetdata3'
for ee in range(100,90,-1):
    ee=92
    imname = []
    dataloader=DataLoader(ImageDataset(dataroot,mode),batch_size=1,shuffle=False,num_workers=2)

    # if not os.path.exists('model_detection_sft_is_run3_18_data3/maskis'):
        # os.makedirs('model_detection_sft_is_run3_18_data3/maskis')
        
    for i, (real_nsr,name) in enumerate(dataloader):
  
        imname.append(name)
        print(name)
   
    imname = np.array(imname)
    print(imname)
    nnn='model_detection_sft_is_run3_18_data3/data3_%s.mat'%(ee)
    scipyio.savemat(nnn, {'name': imname})
    exit(0)
