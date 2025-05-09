from __future__ import print_function
import os
import datetime
import argparse
import torchvision
from torch.utils.data import DataLoader
import torch
from utils import weights_init_normal
import numpy as np



from bdrar_full import BDRAR
from hourglass_pose_mask_wj2 import hg
from model_detection_sft_dia3 import VanishingNet
from  datasets import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,0,1,2,6,7"
if not os.path.exists('model_detection_sft_dia3'):
    os.mkdir('model_detection_sft_dia3')
log_path = os.path.join('model_detection_sft_dia3', str(datetime.datetime.now()) + '.txt')
mode='train'



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--iter_loss', type=int, default=10, help='average loss for n iterations')
opt = parser.parse_args()
print(opt)

dataroot = '/home/liuzhihao/dataset/vpdataset'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sdnet=BDRAR()
sdnet.load_state_dict(torch.load('vps_3001.pth'))
sdnet=sdnet.to(device)
sdnet=torch.nn.DataParallel(sdnet,device_ids=[0,1])
#sdnet.load_state_dict(torch.load('3000.pth'), map_location={'cuda:1':"cuda:1"})

vpnet=hg(planes=64, depth=4, num_stacks=1, num_blocks=1)
vpnet=VanishingNet(vpnet,4,1)
vpnet=vpnet.to(device)
vpnet=torch.nn.DataParallel(vpnet,device_ids=[0,1])
#vpnet.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()
# Optimizers & LR schedulers
optimizer_SD = torch.optim.Adam(sdnet.parameters(),lr=opt.lr*0.2,weight_decay=0.0006,amsgrad=True)
optimizer_G = torch.optim.Adam(vpnet.parameters(),lr=opt.lr*2,weight_decay=0.0006,amsgrad=True)
# Dataset loader
dataloader = DataLoader(ImageDataset(dataroot,mode),batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

open(log_path, 'w').write(str(opt) + '\n\n')

def _loss(result):
    losses = result["losses"]
    # Don't move loss label to other place.
    # If I want to change the loss, I just need to change this function.
    loss_labels = ["sum"] + list(losses[0].keys())
    metrics = np.zeros([1, len(loss_labels)])
    total_loss = 0
    for i in range(1):
        for j, name in enumerate(loss_labels):
            if name == "sum":
                continue
            if name not in losses[i]:
                assert i != 0
                continue
            loss = losses[i][name].mean()
            metrics[i, 0] += loss.item()
            metrics[i, j] += loss.item()
            total_loss += loss
    return total_loss

for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_nsr,mask,realco) in enumerate(dataloader):
        vpnet.train()
        sdnet.train()
        real_nsr=real_nsr.cuda()
        mask=mask.cuda()
        realco=realco.cuda()
        
        optimizer_SD.zero_grad()
        optimizer_G.zero_grad()
        sdmask,s1,s2,s3,s4,s5,s6,s7,s8=sdnet(real_nsr)             
        loss_fuse = criterion_GAN(sdmask, mask)
        loss1_h2l = criterion_GAN(s1, mask)
        loss2_h2l = criterion_GAN(s2, mask)
        loss3_h2l = criterion_GAN(s3, mask)
        loss4_h2l = criterion_GAN(s4, mask)
        loss1_l2h = criterion_GAN(s5, mask)
        loss2_l2h = criterion_GAN(s6, mask)
        loss3_l2h = criterion_GAN(s7, mask)
        loss4_l2h = criterion_GAN(s8, mask)
        loss_sd=loss_fuse+loss1_h2l+loss2_h2l+loss3_h2l+loss4_h2l+loss1_l2h+loss2_l2h+loss3_l2h+loss4_l2h
        result=vpnet(real_nsr,realco,sdmask.sigmoid()) 
        loss_vp=_loss(result)
        loss_G=loss_vp+loss_sd
        
        loss_G.backward()
        optimizer_G.step() 
        optimizer_SD.step()

        if (i+1) % opt.iter_loss == 0:
            log = 'Epoch: %d,[loss_G %.5f]'%(epoch,loss_G,)
            print(log)
            open(log_path, 'a').write(log + '\n')

    if epoch>89:
        torch.save(vpnet.state_dict(), ('model_detection_sft_dia3/vpnet_%d.pth' % (epoch + 1)))
        torch.save(sdnet.state_dict(), ('model_detection_sft_dia3/sdnet_%d.pth' % (epoch + 1)))
    if epoch == opt.decay_epoch:
        optimizer_G.param_groups[0]["lr"] /= 10
        optimizer_SD.param_groups[0]["lr"] /= 10

    print('Epoch:{}'.format(epoch))