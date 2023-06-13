from clinicInfo_fun import *
from Side_Crop_fun import *
from Upper_Crop_fun import *
from check_indices_within_bounding_box_fun import *
from display_ClinicCrops_fun import *


def cropCTfromROI_ClinicalInfo(plan_ct,plan_ct_LM,itv,clinicInfo_path,patientID,pet,bool_planCT=False):
    #bool_planCT only True when using planning CT otherwise, ITV will never be inside the info 
    if bool_planCT:
        error_slice=0
        error_row=0
        error_col=0
        count_correct=[]
        count_NOcorrect=[]
    
    side_value,upper_value = clinicInfo_values(clinicInfo_path,patientID)

    print(side_value,upper_value)
    #Crop 
    side_crop,side_coords = Side_Crop(plan_ct_LM,side_value)
    upper_crop,upper_coords = Upper_Crop(side_crop,upper_value)
    print("UpperCiirds",upper_coords)
    #Fix Coords
    sum_coords = upper_coords
    sum_coords[:3] = [x + y for x, y in zip(upper_coords[:3], side_coords[:3])]
    sum_coords[3:] = [x + y for x, y in zip(upper_coords[3:], side_coords[:3])]
    sum_coords = [sum_coords[2],sum_coords[1],sum_coords[0],sum_coords[5],sum_coords[4],sum_coords[3]]
    
    if False:
        plan_ct_nii_cropped = plan_ct[sum_coords[2]:sum_coords[5],sum_coords[1]:sum_coords[4],sum_coords[0]:sum_coords[3]]
        plan_ct_LM_nii_cropped = plan_ct_LM[sum_coords[2]:sum_coords[5],sum_coords[1]:sum_coords[4],sum_coords[0]:sum_coords[3]]
        pet_nii_cropped = pet[sum_coords[2]:sum_coords[5],sum_coords[1]:sum_coords[4],sum_coords[0]:sum_coords[3]]
    else:
        plan_ct_nii_cropped = plan_ct[sum_coords[0]:sum_coords[3],sum_coords[1]:sum_coords[4],sum_coords[2]:sum_coords[5]]
        plan_ct_LM_nii_cropped = plan_ct_LM[sum_coords[0]:sum_coords[3],sum_coords[1]:sum_coords[4],sum_coords[2]:sum_coords[5]]
        pet_nii_cropped = pet[sum_coords[0]:sum_coords[3],sum_coords[1]:sum_coords[4],sum_coords[2]:sum_coords[5]]

        
    #Check if tumor inside bb
    if bool_planCT: 
        is_within_bounding_box,corrupt_index = check_indices_within_bounding_box(itv_nii, sum_coords)

        print("Contained:",is_within_bounding_box)
        if not is_within_bounding_box:
            count_NOcorrect.append(patientID)
            if corrupt_index[0]<sum_coords[0] or corrupt_index[0]>sum_coords[3]:
                error_slice+=1
            if corrupt_index[1]<sum_coords[1] or corrupt_index[0]>sum_coords[4]:
                error_row+=1
            if corrupt_index[2]<sum_coords[2] or corrupt_index[0]>sum_coords[5]:
                error_col+=1
        else:
            count_correct.append(patientID)
        
        #Display
        display_ClinicCrops(plan_ct,sum_coords,itv_nii,plan_ct_LM,corrupt_index)
    else:
        #Display
        itv_nii=None
        corrupt_index = None
        #display_ClinicCrops(plan_ct,sum_coords,itv_nii,plan_ct_LM,corrupt_index)

    
    return plan_ct_nii_cropped,pet_nii_cropped

