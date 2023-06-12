import numpy as np
from CropBinary_fun import *
import matplotlib.pyplot as plt


def Upper_Crop(plan_ct_LM_nii,key):
    indices = np.where(plan_ct_LM_nii == 1)
    first_occurrence = (indices[0][0], indices[1][0], indices[2][0])
    last_occurrence = (indices[0][-1], indices[1][-1], indices[2][-1])

    depth = indices[0][-1] - indices[0][0]
    height = indices[1][-1] - indices[1][0]
    width  = indices[2][-1] - indices[2][0]
    top_crop = depth // 2
    
    image = np.copy(plan_ct_LM_nii)
    
    if key=="Upper":
        image[:top_crop, :, :] = 0
        cropped,coords = CropBinary(image,extra=5)
    if key=="Lower":
        image[top_crop:, :, :] = 0
        cropped,coords = CropBinary(image,extra=5)
    if key=="Bronchial" or key=="empty":
        cropped,coords = CropBinary(image,extra=5)
    if key=="Mediastinum" or key =="Middle/hilar":
        quarter_crop = top_crop//2
        image[:quarter_crop, :, :] = 0
        image[quarter_crop+top_crop:, :, :] = 0
        cropped,coords = CropBinary(image,extra=5)        
    if False:
        plt.subplot(1,2,1),plt.imshow(plan_ct_LM_nii[10,:,:]),plt.title("Upper"),plt.axis("off")
        plt.subplot(1,2,2),plt.imshow(image[10,:,:]),plt.title("Upper"),plt.axis("off")
        plt.show()
        
    return cropped,coords
