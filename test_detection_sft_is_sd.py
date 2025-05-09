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
from os.path import exists, join as join_paths

from bdrar_full import BDRAR
# from datasets_test import ImageDataset
from datasetis_test import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"]="7,5,4,6,1,3,0,2"

mode='test'
dataroot = '/home/liuzhihao/dataset/vpdataset'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sdnet=BDRAR()
sdnet=sdnet.to(device)
sdnet=torch.nn.DataParallel(sdnet,device_ids=[0,1])


for ee in range(100,99,-1):
    generator_2='model_detection_sft_is_data3_sd/sdnet_%s.pth'%(ee)
    sdnet.load_state_dict(torch.load(generator_2))
    sdnet.eval()

    dataloader=DataLoader(ImageDataset(dataroot,mode),batch_size=1,shuffle=False,num_workers=2)

    if not os.path.exists('model_detection_sft_is_data3_sd/maskis'):
        os.makedirs('model_detection_sft_is_data3_sd/maskis')
        
    # for i, (real_nsr,mask,realco) in enumerate(dataloader):
    for i, (real_nsr,realco) in enumerate(dataloader):
        real_nsr=real_nsr.cuda()
        sdmask,s1,s2,s3,s4,s5,s6,s7,s8=sdnet(real_nsr)       
        # sdmask=mask.cuda()
        
        fake_B448 = torch.sigmoid(sdmask).data
        fake_B448=255.0*(fake_B448)
        fake_B448=fake_B448.data.squeeze(0).cpu()
        fake_B448=fake_B448.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        fake_B448=resize(fake_B448,(256,256,1))
        outputimage=fake_B448
        save_result=join_paths('./model_detection_sft_is_data3_sd/maskis/%s'% (str(i) + '.png'))
        io.imsave(save_result,outputimage)
