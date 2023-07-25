import numpy as np
from CropBinary_fun import *
import matplotlib.pyplot as plt
import torch


def Side_Crop(image,mask,key):
    if key =="Left":
        areaLung = np.copy(mask[0].numpy())
        print(np.unique(areaLung.flatten()))
        areaLung[areaLung==1] = 0
        areaLung[areaLung==2] = 1
    elif key =="Right":
        areaLung = np.copy(mask)
        print(np.unique(areaLung.flatten()))
        areaLung[areaLung>1] = 0
    elif key =="Both" or key =="Unknown" or key == "Bilateral":
        areaLung = np.copy(mask)
        areaLung[areaLung==2] = 1
    elif key =="Mediastinum" or key =="Mediastinal":
        areaLung = np.copy(mask)
        areaLung[areaLung==2] = 1

        #CenterSpatialCropd

    else:
        print("Error")
        exit(1)

    image_side, mask_side = CropBinary_monai(image[0],torch.from_numpy(areaLung.squeeze()))

    return image_side,mask_side
