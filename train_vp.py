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
from model_vp import Discriminator
from datasets_vp import ImageDataset
import numpy as np
import scipy.io as io
os.environ["CUDA_VISIBLE_DEVICES"]="0,7,3,4,2,5,6,1"
torch.manual_seed(628)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--iter_loss', type=int, default=1, help='average loss for n iterations')
opt = parser.parse_args()


# ISTD
opt.dataroot = '/home/liuzhihao/dataset/vpdataset'

if not os.path.exists('model_vp'):
    os.mkdir('model_vp')
opt.log_path = os.path.join('model_vp', str(datetime.datetime.now()) + '.txt')

if torch.cuda.is_available():
    opt.cuda = True

print(opt)

###### Definition of variables ######
# Networks
netG_A2B = Discriminator()  # shadow to shadow_free

if opt.cuda:
    netG_A2B.cuda()

netG_A2B.apply(weights_init_normal)


# Lossess
criterion_GAN = torch.nn.MSELoss()  # lsgan
# criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers

optimizer_G = torch.optim.Adam(netG_A2B.parameters(),lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocations
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, 3, 500, 500)
input_B = Tensor(1,2)


# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),
						batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

curr_iter = 0
G_losses_temp = 0
G_losses = []
open(opt.log_path, 'w').write(str(opt) + '\n\n')

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_nsr = Variable(input_A.copy_(batch['A']))#non shadow region:input;step1-gt
        realco = Variable(input_B.copy_(batch['C']))#non shadow region:input;step1-gt
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_nsr)
        loss_identity_B = criterion_GAN(same_B, realco)

        # Total loss
        loss_G = loss_identity_B
        loss_G.backward()

        #G_losses.append(loss_G.item())
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
    # Update learning rates
    lr_scheduler_G.step()


    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'model_vp/netG_A2B.pth')

    if (epoch + 1) % opt.snapshot_epochs == 0:
        torch.save(netG_A2B.state_dict(), ('model_vp/netG_A2B_%d.pth' % (epoch + 1)))

    print('Epoch:{}'.format(epoch))
