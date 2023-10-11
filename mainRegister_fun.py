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
from displayRegist_fun import *
from Resample_fun import *
import matplotlib.pyplot as plt
from BinaryEvaluation_fun import *
from similarityMetrics_fun import *
from rigid_registration_lite import tailor_registration


import torch


def mainRegister(save_register,intermediate_dict,pxID,save_CSVs):
    print("Inside Registration module")

    transf_spec = "Scale3D"
    center_spec = "Geometry"
    metric_spec = "Correlation"
    optimizer = "RegularStepGradientDescent"
    shift_sepc = "PhysicalShift"
    iterations_spec = 300
    lr = 1
    minStep = 0.0001
    gradientT = 1e-8
    offset = "Diff"

    PlanCT_LungCrop_tensor,ITV_LungCrop_tensor,PlanCT_LungMask_LungCrop_tensor,LDCT_LungCrop_tensor,PET_LungCrop_tensor,LDCT_LungMask_LungCrop_tensor = OnlyRead_Intermediate(intermediate_dict, True, False)

    registCT2_LM, registPET2_LM,_,_ = tailor_registration(PlanCT_LungCrop_tensor[0], LDCT_LungCrop_tensor[0],PET_LungCrop_tensor[0],
                                                          transf_spec, center_spec, metric_spec, optimizer, shift_sepc, offset,
                                                          iterations_spec=iterations_spec,lr=lr,minStep=minStep, gradientT=gradientT)

    save_nifti_without_header(registCT2_LM, filename=save_register + "LDCT_LungCrop_Register_v12.nii.gz")
    save_nifti_without_header(registPET2_LM, filename=save_register + "PET_LungCrop_Register_v12.nii.gz")
    print("Regist 12 Done")
    print("Everything is saved")

