from skimage.filters import threshold_multiotsu
import numpy as np



def cropCTfromROI_BinaryPET(plan_ct_cropped_1,pet_cropped_1):
    
    local_otsu = threshold_multiotsu(pet_cropped_1,classes=3)
    otsu_lvl0 = local_otsu[0]
    otsu_lvl1 = local_otsu[1]
    binary_pet= pet_cropped_1 > otsu_lvl1

    # Get the indices where the binary image is non-zero (foreground)
    indices = np.where(binary_pet != 0)

    # Extract the minimum and maximum values for each dimension
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_z, max_z = np.min(indices[2]), np.max(indices[2])
    
    plan_ct_nii_cropped = plan_ct_cropped_1[min_x:max_x,min_y:max_y,min_z:max_z]
    pet_nii_cropped = pet_cropped_1[min_x:max_x,min_y:max_y,min_z:max_z]
    
    return plan_ct_nii_cropped,pet_nii_cropped
