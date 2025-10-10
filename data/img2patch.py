import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from utils import image_read

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

data_name="MSRS_train"
patchsize=128   
stride=200

folderpath=os.path.join("data",data_name)
os.makedirs(folderpath,exist_ok=True)

imgpath1=... #ir path
imgpath2=... #vi path

patch_num=0

p_H_num=(480-patchsize)//stride + 1
p_W_num=(640-patchsize)//stride + 1

file_name_list=os.listdir(imgpath1) 

for k in tqdm(range(len(file_name_list))):
    img1=image_read(os.path.join(imgpath1,file_name_list[k]),mode='GRAY')
    img2=image_read(os.path.join(imgpath2,file_name_list[k]),mode='GRAY')

    H,W=img1.shape

    p_H_num=(H-patchsize)//stride + 1
    p_W_num=(W-patchsize)//stride + 1

    for kk in range(p_H_num*p_W_num):
        os.makedirs(os.path.join(folderpath,str(patch_num)),exist_ok=True)

        a0=kk//p_W_num 
        a1=kk-a0*p_W_num 

        img1_patch=img1[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize]
        img2_patch=img2[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize]
        
        if not (is_low_contrast(img1_patch) or is_low_contrast(img2_patch)):
            cv2.imwrite(os.path.join(folderpath,str(patch_num),"img_1.png"), img1_patch)
            cv2.imwrite(os.path.join(folderpath,str(patch_num),"img_2.png"), img2_patch)
            patch_num+=1

print('training set, # samples %d\n' % (patch_num))   