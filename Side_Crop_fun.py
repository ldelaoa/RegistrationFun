import numpy as np
from CropBinary_fun import *
import matplotlib.pyplot as plt



def Side_Crop(plan_ct_LM_nii,key):
    if key =="Left":
        leftLung = np.copy(plan_ct_LM_nii)
        print(np.unique(leftLung.flatten()))
        leftLung[leftLung==1] = 0
        leftLung[leftLung==2] = 1
        cropped,coords = CropBinary(leftLung)
    elif key =="Right":
        rightLung = np.copy(plan_ct_LM_nii)
        print(np.unique(rightLung.flatten()))
        rightLung[rightLung>1.5] = 0
        plt.imshow(rightLung[150,:,:])
        plt.show()
        cropped,coords = CropBinary(rightLung)
    elif key =="Both" or key =="Unknown":
        bothLung = np.copy(plan_ct_LM_nii)
        bothLung[bothLung==2] = 1
        cropped,coords = CropBinary(bothLung)
    elif key =="Mediastinum" or key =="Mediastinal":
        bothLung = np.copy(plan_ct_LM_nii)
        bothLung[bothLung==2] = 1
        
        cropped,coords = CropBinary(bothLung)
        
        depth = coords[5] - coords[2]
        height = coords[4] - coords[1]
        width  = coords[3] - coords[0]
        
        first_quarter_crop = width // 4 
        last_quarter_crop = ((width // 4)*3)
        
        #print(first_quarter_crop,last_quarter_crop,width)
        
        image = np.copy(bothLung)
        
        image[coords[2]:coords[5],coords[1]:coords[4],:coords[0]+first_quarter_crop] = 0
        image[coords[2]:coords[5],coords[1]:coords[4],coords[3]-first_quarter_crop:] = 0
        #print(coords[0],":",coords[3],"width")
        #print(coords[1],coords[4],"height")
        #print(coords[2],coords[5],"Depth")
        cropped,coords = CropBinary(image)
        #print(image.shape)
        #plt.subplot(1,2,1),plt.imshow(image[70,:,:],'gray'),plt.title("Side"),plt.axis("off")
        #plt.show()

    else:
        print("Error")
    return cropped,coords
