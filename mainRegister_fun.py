#import os
#import numpy as np
#from os.path import join
#from datetime import time, datetime
#import glob
#from skimage.draw import polygon
#import SimpleITK as sitk
#import pydicom as dicom
import os.path
import csv

#import nibabel as nib
#from skimage.filters import threshold_multiotsu
#from nibabel.processing import resample_to_output
#from skimage.util import crop

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
from displayRegist_fun import *
from Resample_fun import *
import matplotlib.pyplot as plt
from BinaryEvaluation_fun import *
from similarityMetrics_fun import *


import torch


def mainRegister(save_register,intermediate_dict,pxID,save_CSVs):
    print("Inside Registration module")
    PlanCT_LungCrop_tensor,ITV_LungCrop_tensor,PlanCT_LungMask_LungCrop_tensor,LDCT_LungCrop_tensor,PET_LungCrop_tensor,LDCT_LungMask_LungCrop_tensor = OnlyRead_Intermediate(intermediate_dict, True, False)

    registCT1_LM,registPET1_LM,eval1_LM,sX_LM1,sY_LM1,sZ_LM1,rX_LM1,rY_LM1,rZ_LM1 = Register_fun(PlanCT_LungCrop_tensor[0],LDCT_LungCrop_tensor[0],PET_LungCrop_tensor[0],pxID)
    registCT2_LM, registPET2_LM,eval2_LM,sX_LM2,sY_LM2,sZ_LM2,rX_LM2,rY_LM2,rZ_LM2 = Register_fun_v2(PlanCT_LungCrop_tensor[0], LDCT_LungCrop_tensor[0], PET_LungCrop_tensor[0], pxID)

    PlanCT_Clinic_tensor, ITV_Clinic_tensor, PlanCT_LungMask_Clinic_tensor, LDCT_Clinic_tensor, PET_Clinic_tensor, LDCT_LungMask_Clinic_tensor = OnlyRead_Intermediate(intermediate_dict, False, True)
    registCT1_Clinic, registPET1_Clinic,eval1_Clin,sX_Clin1,sY_Clin1,sZ_Clin1,rX_Clin1,rY_Clin1,rZ_Clin1 = Register_fun(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0], PET_Clinic_tensor[0], pxID)
    registCT2_Clinic, registPET2_Clinic,eval2_Clin,sX_Clin2,sY_Clin2,sZ_Clin2,rX_Clin2,rY_Clin2,rZ_Clin2 = Register_fun_v3(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0],PET_Clinic_tensor[0], pxID)

    if True:
        save_nifti_without_header(registCT1_LM, filename=save_register + "LDCT_LungCrop_Register_v1.nii.gz")
        save_nifti_without_header(registCT2_LM, filename=save_register + "LDCT_LungCrop_Register_v2.nii.gz")
        save_nifti_without_header(registPET1_LM, filename=save_register + "PET_LungCrop_Register_v1.nii.gz")
        save_nifti_without_header(registPET2_LM, filename=save_register + "PET_LungCrop_Register_v2.nii.gz")

        save_nifti_without_header(registCT1_Clinic, filename=save_register+"LDCT_Clinic_Register_v1.nii.gz")
        save_nifti_without_header(registCT2_Clinic, filename=save_register+"LDCT_Clinic_Register_v2.nii.gz")
        save_nifti_without_header(registPET1_Clinic, filename=save_register + "PET_Clinic_Register_v1.nii.gz")
        save_nifti_without_header(registPET2_Clinic, filename=save_register + "PET_Clinic_Register_v2.nii.gz")

    cropOk = torch.sum(ITV_LungCrop_tensor) - torch.sum(ITV_Clinic_tensor)

    tmp_path = save_CSVs+"Cropping_metrics.csv"
    with open(tmp_path, "a", newline="") as file_tmp:
        writer = csv.writer(file_tmp)
        writer.writerow(([pxID,cropOk]))
        writer.writerow(([eval1_LM,sX_LM1,sY_LM1,sZ_LM1,rX_LM1,rY_LM1,rZ_LM1]))
        writer.writerow(([eval2_LM,sX_LM2,sY_LM2,sZ_LM2,rX_LM2,rY_LM2,rZ_LM2]))
        writer.writerow(([eval1_Clin,sX_Clin1,sY_Clin1,sZ_Clin1,rX_Clin1,rY_Clin1,rZ_Clin1]))
        writer.writerow(([eval2_Clin,sX_Clin2,sY_Clin2,sZ_Clin2,rX_Clin2,rY_Clin2,rZ_Clin2]))