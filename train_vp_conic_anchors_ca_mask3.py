from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
import numpy as np
from skimage import io,color
from skimage.transform import resize

from hourglass_pose import hg
from model_vp_conic_anchors_ca_mask3 import VanishingNet
from datasets_vp_full import ImageDataset
from datasets_vp_full_test import ImageDataset as TestImageDataset

os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6,7,0,1,2"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=60,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--iter_loss', type=int, default=10, help='average loss for n iterations')
opt = parser.parse_args()

# ISTD
opt.dataroot = '/home/liuzhihao/dataset/vpdataset'

if not os.path.exists('model_vp_conic_anchors_ca_mask3'):
    os.mkdir('model_vp_conic_anchors_ca_mask3')
opt.log_path = os.path.join('model_vp_conic_anchors_ca_mask3', str(datetime.datetime.now()) + '.txt')

print(opt)

vpnet=hg(planes=64, depth=4, num_stacks=1, num_blocks=1)
vpnet=VanishingNet(vpnet,4,1)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vpnet=vpnet.to(device)
vpnet=torch.nn.DataParallel(vpnet,device_ids=[0,1])
#vpnet.apply(weights_init_normal)

# Lossess
#criterion_GAN = torch.nn.MSELoss()  # lsgan
criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(vpnet.parameters(),lr=opt.lr*2,weight_decay=0.0006,amsgrad=True)

# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot),batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
testdataloader = DataLoader(TestImageDataset(opt.dataroot),batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

curr_iter = 0
G_losses_temp = 0
G_losses = []
open(opt.log_path, 'w').write(str(opt) + '\n\n')


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
    
mean_loss=10000
best_mean_loss=10000
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_nsr,mask,realco) in enumerate(dataloader):
        vpnet.train()
        real_nsr=real_nsr.cuda()
        realco=realco.cuda()
        mask=mask.cuda()
        optimizer_G.zero_grad()
        result = vpnet(real_nsr,realco,mask)
        
        loss_identity_B = _loss(result)
        if np.isnan(loss_identity_B.item()):
            raise ValueError("loss is nan while training")
        loss_G = loss_identity_B
        loss_G.backward()

        G_losses_temp += loss_G.item()

        optimizer_G.step()
        ###################################

        curr_iter += 1
        if (i+1) % opt.iter_loss == 0:
            log = 'Epoch: %d, [iter %d], [loss_G %.5f]' % \
                  (epoch, curr_iter, loss_G,)
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            G_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f]' \
                      % (opt.iter_loss, G_losses[G_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

    # Save models checkpoints
    # torch.save(vpnet.state_dict(), 'model_vp_conic_anchors_ca_mask3/vpnet.pth')

    if epoch>89:
        torch.save(vpnet.state_dict(), ('model_vp_conic_anchors_ca_mask3/vpnet_%d.pth' % (epoch + 1)))

    print('Epoch:{}'.format(epoch))
    '''
    torch.set_grad_enabled(False)
    if epoch>90:
        with torch.no_grad(): 
            # evaluate
            eval_loss=0
            for i, (real_nsr,mask,realco,name) in enumerate(testdataloader):
                vpnet.eval()
                real_nsr=real_nsr.cuda()
                realco=realco.cuda()
                mask=mask.cuda()
                result=vpnet(real_nsr,realco,mask)
                loss_eval=_loss(result)
                eval_loss=eval_loss+loss_eval
            mean_loss=eval_loss/len(testdataloader)
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                torch.save(vpnet.state_dict(), ('model_vp_conic_anchors_ca_mask3/vpnet_best.pth'))
    torch.set_grad_enabled(True)
    '''
            
            
            
            
            
            
            
            
            
            
            
            
