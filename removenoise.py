import os
import cv2
import numba
import skimage

dirpath = r'D:\code\CVandDL\images'
savepath = r'D:\code\CVandDL\images'
imagelist = os.listdir(dirpath)

for imagename in imagelist:
    fullpath = os.path.join(dirpath, imagename)
    Img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
    if Img.shape[2] == 3:
        I = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    else:
        I = Img
Ig = skimage.util.random_noise(I, mode='poisson')
s=