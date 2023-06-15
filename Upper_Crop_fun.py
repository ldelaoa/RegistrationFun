import numpy as np
from CropBinary_fun import *
import matplotlib.pyplot as plt
import torch


def Upper_Crop(image,mask,key):
    areaLung = np.copy(mask[0].numpy())
    indices = np.where(areaLung == 1)
    depth = np.max(indices[2])-np.min(indices[2])
    top_crop = depth // 2
    
    if key=="Upper":
        areaLung[:, :, :top_crop] = 0
    if key=="Lower":
        areaLung[:,:,top_crop:] = 0
    if key=="Bronchial" or key=="empty":
        areaLung = areaLung
    if key=="Mediastinum" or key =="Middle/hilar":
        quarter_crop = top_crop//2
        areaLung[:,:,quarter_crop] = 0
        areaLung[:,:,quarter_crop+top_crop:] = 0

    image_upper, mask_upper = CropBinary_monai(image[0], torch.from_numpy(areaLung))

    return image_upper, mask_upper
