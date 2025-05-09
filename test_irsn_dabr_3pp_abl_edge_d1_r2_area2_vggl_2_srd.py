import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize

from model_irsn_dabr_3pp_abl_edge_d1_r2_rl_2 import dadnet
os.environ["CUDA_VISIBLE_DEVICES"]="2,7,1,0,4,5,6,3"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

## ISTD
opt.dataroot_A = '/home/liuzhihao/dataset/SRD/test/shadow'

opt.dataroot_mask = '/home/liuzhihao/dataset/SRD/test/SRD_testmask/'

opt.im_suf_A = '.jpg'
if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)

dad = dadnet()

if opt.cuda:
    # dad = torch.nn.DataParallel(dad)
    dad.cuda()

gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]
with torch.no_grad():

    for ee in range(100,84,-1):
        # generator='model_irsn_dabr_3pp_abl_edge_d1_r2_area2_vggl_2_srd_run2/dad.pth'
        generator='model_irsn_dabr_3pp_abl_edge_d1_r2_area2_vggl_2_srd_run2/dad_%s.pth'%(ee)
        dad.load_state_dict(torch.load(generator))
        dad.eval()
                
        savepath='model_irsn_dabr_3pp_abl_edge_d1_r2_area2_vggl_2_srd_run2/B_%s'%(ee)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for idx, img_name in enumerate(gt_list):
            rgbimage=io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A))
            labimage = color.rgb2lab(rgbimage)
            
            labimage480=resize(labimage,(480,640,3))
            labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
            labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
            labimage480=torch.from_numpy(labimage480).float()
            labimage480=labimage480.view(480,640,3)
            labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
            labimage480=labimage480.unsqueeze(0).to(device)
            
            mask=io.imread(os.path.join(opt.dataroot_mask, img_name + opt.im_suf_A))
            mask480=resize(mask,(480,640,1))
            mask480[mask480>0.5] = 1.0
            mask480[mask480<=0.5]= 0
            mask480=np.asarray(mask480)
            mask480=torch.from_numpy(mask480.copy()).float()
            mask480=mask480.view(480,640,1)
            mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
            mask480=mask480.unsqueeze(0).to(device)
            
            _,_,_,_,temp_B480=dad(labimage480,mask480)
                        
            
            fake_B480=temp_B480.data
            fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
            fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
            fake_B480=fake_B480.data.squeeze(0).cpu()
            fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
            fake_B480=resize(fake_B480,(480,640,3))
            fake_B480=color.lab2rgb(fake_B480)

            
            save_result = join_paths(savepath+'/%s'% (img_name + opt.im_suf_A))
            io.imsave(save_result,fake_B480)
            
            print('Generated ori_images %04d of %04d' % (idx+1, len(gt_list)))