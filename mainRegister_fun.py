import os.path
import csv
from clinicInfo_fun import clinicInfo_idcolumn
from FilesPerPatient_fun import *
from CreateLungMasks_fun import *
from display_LoadImgs_fun import *
from ReadAndResample_fun import *
from cropCTfromROI_lung_fun import *
from cropCTfromROI_ClinicalInfo_fun import *
from cropCTfromROI_BinaryPET_fun import *
from Register_fun import *
from Register_fun_v2 import *
from Register_fun_v4 import *
from Register_fun_v5 import *
from Register_fun_v6 import *
from Register_fun_v7 import *
from Register_fun_v8 import *
from Register_fun_v9 import *
from Register_fun_v10 import *
from Register_fun_v11 import *
from displayRegist_fun import *
from Resample_fun import *
import matplotlib.pyplot as plt
from BinaryEvaluation_fun import *
from similarityMetrics_fun import *


import torch


def mainRegister(save_register,intermediate_dict,pxID,save_CSVs):
    print("Inside Registration module")
    clinicBool = False
    PlanCT_LungCrop_tensor,ITV_LungCrop_tensor,PlanCT_LungMask_LungCrop_tensor,LDCT_LungCrop_tensor,PET_LungCrop_tensor,LDCT_LungMask_LungCrop_tensor = OnlyRead_Intermediate(intermediate_dict, True, False)

    registCT1_LM, registPET1_LM,  _, _, _, _, _, _, _ = Register_fun_v9(PlanCT_LungCrop_tensor[0], LDCT_LungCrop_tensor[0], PET_LungCrop_tensor[0], pxID)
    save_nifti_without_header(registCT1_LM, filename=save_register + "LDCT_LungCrop_Register_v9.nii.gz")
    save_nifti_without_header(registPET1_LM, filename=save_register + "PET_LungCrop_Register_v9.nii.gz")
    print("Regist 9 Done")
    registCT2_LM, registPET2_LM,  _, _, _, _, _, _, _ = Register_fun_v10(PlanCT_LungCrop_tensor[0], LDCT_LungCrop_tensor[0], PET_LungCrop_tensor[0], pxID)
    save_nifti_without_header(registCT2_LM, filename=save_register + "LDCT_LungCrop_Register_v10.nii.gz")
    save_nifti_without_header(registPET2_LM, filename=save_register + "PET_LungCrop_Register_v10.nii.gz")
    print("Regist 10 Done")
    registCT3_LM, registPET3_LM, _, _, _, _, _, _, _ = Register_fun_v11(PlanCT_LungCrop_tensor[0], LDCT_LungCrop_tensor[0], PET_LungCrop_tensor[0], pxID)
    save_nifti_without_header(registCT3_LM, filename=save_register + "LDCT_LungCrop_Register_v11.nii.gz")
    save_nifti_without_header(registPET3_LM, filename=save_register + "PET_LungCrop_Register_v11.nii.gz")
    print("Regist 11 Done")
    print("Everything is saved")

    if clinicBool:
        PlanCT_Clinic_tensor, ITV_Clinic_tensor, PlanCT_LungMask_Clinic_tensor, LDCT_Clinic_tensor, PET_Clinic_tensor, LDCT_LungMask_Clinic_tensor = OnlyRead_Intermediate(intermediate_dict, False, True)
        registCT1_Clinic, registPET1_Clinic,eval1_Clin,sX_Clin1,sY_Clin1,sZ_Clin1,rX_Clin1,rY_Clin1,rZ_Clin1 = Register_fun(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0], PET_Clinic_tensor[0], pxID)
        registCT2_Clinic, registPET2_Clinic,eval2_Clin,sX_Clin2,sY_Clin2,sZ_Clin2,rX_Clin2,rY_Clin2,rZ_Clin2 = Register_fun_v3(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0],PET_Clinic_tensor[0], pxID)
        save_nifti_without_header(registCT1_Clinic, filename=save_register+"LDCT_Clinic_Register_v1.nii.gz")
        save_nifti_without_header(registCT2_Clinic, filename=save_register+"LDCT_Clinic_Register_v2.nii.gz")
        save_nifti_without_header(registPET1_Clinic, filename=save_register + "PET_Clinic_Register_v1.nii.gz")
        save_nifti_without_header(registPET2_Clinic, filename=save_register + "PET_Clinic_Register_v2.nii.gz")

        cropOk = torch.sum(ITV_LungCrop_tensor) - torch.sum(ITV_Clinic_tensor)
    if False:
        tmp_path = save_CSVs+"Cropping_metrics_v3.csv"
        with open(tmp_path, "a", newline="") as file_tmp:
            writer = csv.writer(file_tmp)
            if clinicBool:
                writer.writerow([pxID,cropOk,eval1_LM,sX_LM1,sY_LM1,sZ_LM1,rX_LM1,rY_LM1,rZ_LM1,eval2_LM,sX_LM2,sY_LM2,sZ_LM2,rX_LM2,rY_LM2,rZ_LM2,eval1_Clin,sX_Clin1,sY_Clin1,sZ_Clin1,rX_Clin1,rY_Clin1,rZ_Clin1,eval2_Clin,sX_Clin2,sY_Clin2,sZ_Clin2,rX_Clin2,rY_Clin2,rZ_Clin2])