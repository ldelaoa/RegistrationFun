import numpy as np


def cropCTfromROI_lung(ct_np_ToCrop,ct_LungMask_np,pet_np,extra=20,pet_bool=True):
    ct_LM_np_copy = np.copy(ct_LungMask_np)
    ct_LM_np_copy[ct_LM_np_copy ==2] =1 
    indices = np.where(ct_LM_np_copy == 1)
    first_slice = (indices[0][0], indices[1][0], indices[2][0])
    last_slice = (indices[0][-1], indices[1][-1], indices[2][-1])
    
    beginCrop_z = np.min(indices[0][:]) - extra
    endCrop_z = np.max(indices[0][:]) + extra
    
    beginCrop_y = np.min(indices[1][:]) - extra
    endCrop_y = np.max(indices[1][:]) + extra
    
    beginCrop_x = np.min(indices[2][:]) - extra
    endCrop_x = np.max(indices[2][:]) + extra
               
    cropped_CT = ct_np_ToCrop[beginCrop_z:endCrop_z,beginCrop_y:endCrop_y, beginCrop_x:endCrop_x]
    #cropped_CT_LM = ct_LM_np[beginCrop_z:endCrop_z,beginCrop_y:endCrop_y, beginCrop_x:endCrop_x]
    if pet_bool:
        cropped_PET = pet_np[beginCrop_z:endCrop_z,beginCrop_y:endCrop_y, beginCrop_x:endCrop_x]
    else:
        cropped_PET = 0

    return cropped_CT,cropped_PET

