# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

import os
import sys
import warnings
import logging
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from numpy import mean
from tqdm import tqdm

from utils import image_read,image_save,loss_fusion,loss_recon,inner_update,outer_update,Dataset_refusion
from ReFusion import ReFusion,LPN,ReN

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 50
lr = 1e-4
step_size=10               
gamma=0.1
batch_size = 4

coef_f_int=1
coef_f_gradient=1
coef_r_int=1
coef_r_grad=1

Switch_r=1              # Exchanging inputs for stable training.
max_meta_step=600
accumulation_steps=2    # Accumulate gradients over multiple batches for stable training. 

Fusion_model= ReFusion().to(device) # Fusion Module
Recon_model = ReN().to(device)      # Reconstruction Module
Loss_model= LPN().to(device)        # Loss Proposal Module

# Optimizer for Fusion Module
optimizer1 = torch.optim.Adam(Fusion_model.parameters(), lr=lr, weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
# Optimizer for Reconstruction Module
optimizer2 = torch.optim.Adam(Recon_model.parameters(), lr=lr, weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)
# Optimizer for Loss Proposal Module
optimizer3 = torch.optim.Adam(Loss_model.parameters(), lr=lr, weight_decay=0,eps=1e-10)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=step_size, gamma=gamma)

# Training dataset
trainloader = DataLoader(Dataset_refusion("data/MSRS_train"),batch_size=batch_size,shuffle=True,num_workers=0)

exp_path=os.path.join("exp",time.strftime("%y_%m_%d_%H_%M", time.localtime()))
os.makedirs(os.path.join(exp_path,"model"),exist_ok=True)

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

f_step = 0
meta_step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

Fusion_model.train()
Recon_model.train()
Loss_model.train()

pbar = tqdm(total=len(trainloader))

for epoch in range(num_epochs):
    coef_f_grad= coef_f_gradient*(epoch+1)/num_epochs

    lr_epoch=optimizer1.param_groups[0]['lr']
    F_loss=[]
    F_loss_int=[]
    F_loss_grad=[]
    R_loss=[]
    R_loss_int=[]
    R_loss_grad=[]

    # Learning LPM

    pbar.total=max_meta_step
    Loss_model.train()
    for i, (data_IR,data_VIS,index) in enumerate(trainloader):
        if i==max_meta_step*2:
            break

        # inner update
        if i %2 ==0:  
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            if Switch_r:
                data_IR,data_VIS=torch.cat((data_IR,data_VIS),0),torch.cat((data_VIS,data_IR),0)
                index=torch.cat((index,index),0)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            data_Fuse=Fusion_model(torch.cat((data_IR,data_VIS),1))
            w1,w2=Loss_model(torch.cat((data_IR,data_VIS),1))

            loss_f_int=loss_fusion(data_Fuse,data_IR,data_VIS,w1[:,0:1,:,:],w1[:,1:2,:,:],"int")
            loss_f_grad=loss_fusion(data_Fuse,data_IR,data_VIS,w2[:,0:1,:,:],w2[:,1:2,:,:],"grad")
            Fusion_loss=coef_f_int*loss_f_int + coef_f_grad*loss_f_grad
            Fusion_loss.backward(create_graph=True,retain_graph=True)
            inner_update(Fusion_model,lr_epoch)  # update F as F'  

            if Switch_r:
                data_IR,data_VIS=torch.cat((data_IR[:batch_size,:,:,:],data_VIS[batch_size:,:,:,:]),0),torch.cat((data_VIS[:batch_size,:,:,:],data_IR[batch_size:,:,:,:]),0)
            
            data_re=Recon_model(data_Fuse.detach())
            data_IR_re=data_re[:,0:1,:,:]
            data_VIS_re=data_re[:,1:2,:,:]

            loss_r_int=loss_recon(data_IR,data_IR_re,data_VIS,data_VIS_re,"int")
            loss_r_grad=loss_recon(data_IR,data_IR_re,data_VIS,data_VIS_re,"grad")
            Recon_loss=coef_r_int*loss_r_int +  coef_r_grad*loss_r_grad
            Recon_loss.backward()
            inner_update(Recon_model,lr_epoch)  # update R as R'

        # outer update   
        else: 
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            if Switch_r:
                data_IR,data_VIS=torch.cat((data_IR,data_VIS),0),torch.cat((data_VIS,data_IR),0)
                index=torch.cat((index,index),0)

            if (i-1) %(2*accumulation_steps) ==0 and accumulation_steps!=0:
                optimizer3.zero_grad()

            data_Fuse=Fusion_model(torch.cat((data_IR,data_VIS),1),meta=True)
            data_re=Recon_model(data_Fuse,meta=True)
            data_IR_re=data_re[:,0:1,:,:]
            data_VIS_re=data_re[:,1:2,:,:]

            if Switch_r:
                data_IR,data_VIS=torch.cat((data_IR[:batch_size,:,:,:],data_VIS[batch_size:,:,:,:]),0),torch.cat((data_VIS[:batch_size,:,:,:],data_IR[batch_size:,:,:,:]),0)
            
            loss_r_int=loss_recon(data_IR,data_IR_re,data_VIS,data_VIS_re,"int")
            loss_r_grad=loss_recon(data_IR,data_IR_re,data_VIS,data_VIS_re,"grad")
            Recon_loss=coef_r_int*loss_r_int + coef_r_grad*loss_r_grad

            if accumulation_steps!=0:
                Recon_loss/=accumulation_steps

            Recon_loss.backward()

            if (i+1) %(2*accumulation_steps)==0:
                outer_update(Loss_model,lr_epoch)   #update P
                optimizer3.zero_grad()

            meta_step+=1 
            F_loss.append(Fusion_loss.item())
            R_loss.append(Recon_loss.item())
            F_loss_int.append(loss_f_int.item())
            F_loss_grad.append(loss_f_grad.item())
            R_loss_int.append(loss_r_int.item())
            R_loss_grad.append(loss_r_grad.item())
            pbar.update(1)

        pbar.set_description("Inner/Outer update Epoch: %d/%d, FusionLoss: %f, F_int: %f, F_grad: %f, ReconLoss: %f, R_int: %f, R_grad: %f"% (epoch + 1, num_epochs , mean(F_loss),mean(F_loss_int),mean(F_loss_grad),mean(R_loss),mean(R_loss_int),mean(R_loss_grad)))

    pbar.reset()  

    # Learning F and R
    Loss_model.eval()

    pbar.total=len(trainloader)
    F_loss=[]
    F_loss_int=[]
    F_loss_grad=[]
    R_loss=[]
    R_loss_int=[]
    R_loss_grad=[]

    for i, (data_IR,data_VIS,index) in enumerate(trainloader):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        if Switch_r:
            data_IR,data_VIS=torch.cat((data_IR,data_VIS),0),torch.cat((data_VIS,data_IR),0)
            index=torch.cat((index,index),0)
        # update F
        optimizer1.zero_grad()
        data_Fuse=Fusion_model(torch.cat((data_IR,data_VIS),1))
        
        w1,w2=Loss_model(torch.cat((data_IR,data_VIS),1))
        w1=w1.detach()
        w2=w2.detach()

        loss_f_int=loss_fusion(data_Fuse,data_IR,data_VIS,w1[:,0:1,:,:],w1[:,1:2,:,:],"int")
        loss_f_grad=loss_fusion(data_Fuse,data_IR,data_VIS,w2[:,0:1,:,:],w2[:,1:2,:,:],"grad")
        Fusion_loss=coef_f_int*loss_f_int + coef_f_grad*loss_f_grad
        Fusion_loss.backward()
        optimizer1.step() 

        # update R
        optimizer2.zero_grad()
        data_re=Recon_model(data_Fuse.detach())
        data_IR_re=data_re[:,0:1,:,:]
        data_VIS_re=data_re[:,1:2,:,:]

        if Switch_r:
            data_IR,data_VIS=torch.cat((data_IR[:batch_size,:,:,:],data_VIS[batch_size:,:,:,:]),0),torch.cat((data_VIS[:batch_size,:,:,:],data_IR[batch_size:,:,:,:]),0)
        
        loss_r_int=loss_recon(data_IR,data_IR_re,data_VIS,data_VIS_re,"int")
        loss_r_grad=loss_recon(data_IR,data_IR_re,data_VIS,data_VIS_re,"grad")
        Recon_loss=coef_r_int*loss_r_int +  coef_r_grad*loss_r_grad
        
        Recon_loss.backward()
        optimizer2.step() 

        f_step+=1
        F_loss.append(Fusion_loss.item())
        R_loss.append(Recon_loss.item())
        F_loss_int.append(loss_f_int.item())
        F_loss_grad.append(loss_f_grad.item())
        R_loss_int.append(loss_r_int.item())
        R_loss_grad.append(loss_r_grad.item())

        pbar.update(1)
        pbar.set_description("Fusion update Epoch: %d/%d, FusionLoss: %f, ReconLoss: %f"% (epoch + 1, num_epochs , mean(F_loss),mean(R_loss)))
    pbar.reset()

    # adjust the learning rate and save model
    scheduler1.step()  
    scheduler2.step()
    scheduler3.step()

    if epoch % 1 == 0:
        torch.save(Fusion_model.state_dict(), os.path.join(exp_path,'model', 'ckpt_%s.pth' % (str(epoch+1))))