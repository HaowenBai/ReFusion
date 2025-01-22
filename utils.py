import numpy as np
import cv2
import os
from skimage.io import imsave

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

def img_save(image, imagename, savepath, CrCb=None):
    temp = np.squeeze(image)
    path1 = os.path.join(savepath, 'RGB')
    path2 = os.path.join(savepath, 'Gray')
    if not os.path.exists(path2):
        os.makedirs(path2)
    imsave(os.path.join(path2, "{}.png".format(imagename)), temp)

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