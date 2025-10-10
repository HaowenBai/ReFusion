import numpy as np
import cv2
import os
from skimage.io import imsave
import torch
import torch.nn.functional as F
import torch.utils.data as Data

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ['RGB','GRAY','YCrCb'], 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def image_save(image, imagename, savepath, CrCb=None):
    temp = np.squeeze(image)
    path1 = os.path.join(savepath, 'RGB')
    path2 = os.path.join(savepath, 'Gray')
    if not os.path.exists(path2):
        os.makedirs(path2)
    imsave(os.path.join(path2, "{}.png".format(imagename)), temp) # If error, make sure imageio==2.26.0 is installed.

    if CrCb is not None:
        assert len(CrCb.shape) == 3 and CrCb.shape[2] == 2, "CrCb error"
        temp_RGB = cv2.cvtColor(np.concatenate((temp[..., np.newaxis], CrCb), axis=2), cv2.COLOR_YCrCb2RGB)
        if not os.path.exists(path1):
            os.makedirs(path1)
        temp_RGB[temp_RGB<0]=0
        temp_RGB[temp_RGB>255]=255
        imsave(os.path.join(path1, "{}.png".format(imagename)), temp_RGB)

def fuse_CrCb(CrCb1,CrCb2):
    assert len(CrCb1.shape) == 3 and CrCb1.shape[2] == 2, "CrCb error"
    assert len(CrCb2.shape) == 3 and CrCb2.shape[2] == 2, "CrCb error"
    Cf=(CrCb1*np.abs(CrCb1-0.5)+CrCb2*np.abs(CrCb2-0.5))/(np.abs(CrCb1-0.5)+np.abs(CrCb2-0.5)+1e-4)
    return Cf

def is_grayscale(image):
    return np.all(image[:,:,0] == image[:,:,1]) and np.all(image[:,:,1] == image[:,:,2])


class Dataset_refusion(Data.Dataset):
    def __init__(self, file_path=r"data/MSRS_train"):
        self.file_path = file_path
        
    def __len__(self):
        return len(os.listdir(self.file_path))
    
    def __getitem__(self, index):
        patch_path=os.path.join(self.file_path,str(index))
        img1=image_read(os.path.join(patch_path,"img_1.png"),"GRAY")[None,...]/255.
        img2=image_read(os.path.join(patch_path,"img_2.png"),"GRAY")[None,...]/255.
        
        return  torch.Tensor(img1),torch.Tensor(img2),index
    
def grad(tensor):
    def sobel_x(tensor):
        kernel = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )        
        kernel = kernel.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)
        grad_x = F.conv2d(tensor, kernel, padding=1, groups=tensor.shape[1])
        return grad_x 

    def sobel_y(tensor):
        kernel = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel = kernel.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)
        grad_y = F.conv2d(tensor, kernel, padding=1, groups=tensor.shape[1])
        return grad_y

    return torch.abs(sobel_x(tensor))+torch.abs(sobel_y(tensor))

def loss_fusion(img_F,img_A,img_B,w_A,w_B,stat="int"): 
    if stat=="int":
        return F.mse_loss(w_A*(img_A-img_F),torch.zeros_like(img_F))+F.mse_loss(w_B*(img_B-img_F),torch.zeros_like(img_F))
    elif stat=="grad":
        return F.mse_loss(w_A*(grad(img_A)-grad(img_F)),torch.zeros_like(img_F))+F.mse_loss(w_B*(grad(img_B)-grad(img_F)),torch.zeros_like(img_F))
    
def loss_recon(img_X,img_X_re,img_Y,img_Y_re,stat="int"):
    if stat=="int":
        return F.mse_loss(torch.max(torch.abs(img_X-img_X_re),torch.abs(img_Y-img_Y_re)),torch.zeros_like(img_X))
        
    elif stat=="grad":
        return F.mse_loss(torch.max(torch.abs(grad(img_X)-grad(img_X_re)),torch.abs(grad(img_Y)-grad(img_Y_re))),torch.zeros_like(img_X))

def named_params(curr_module, prefix=''): 
    memo=set()
    if hasattr(curr_module, 'named_leaves'):
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in named_params(module, submodule_prefix):
                yield name, p
                
def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        setattr(curr_mod, name, param)

def inner_update(curr_mod,lr):
    for name, p  in  named_params(curr_mod):
        set_param(curr_mod, name+"_meta", p-lr*p.grad)

def outer_update(curr_mod,lr):
    with torch.no_grad():
        max_grad=0
        for _, param in curr_mod.named_parameters():
            max_grad=max(torch.max(torch.abs(param.grad)).item(),max_grad)
        if max_grad >0:  
            for _, param in curr_mod.named_parameters():
                param.add_(-lr*(param.grad/max_grad))