import os.path
import csv

from CreateLungMasks_fun import *
from display_LoadImgs_fun import *
from ReadAndResample_fun import *
from cropCTfromROI_ClinicalInfo_fun import *
from Resample_fun import *
from BinaryEvaluation_fun import *


def mainCrop(save_root,data_dicts,device,pxID,clinicInfo_path):
    print("Creating images for registration")
    PlanCT_tensor, ITV_tensor, LDCT_CUDAtensor, PET_CUDAtensor = ReadAndOrient_monai(data_dicts, device)
    pet_CUDAResampled = OnlyResamplingPET(LDCT_CUDAtensor, PET_CUDAtensor, device)
    LDCT_tensor = LDCT_CUDAtensor.cpu()
    pet_Resampled = pet_CUDAResampled.to('cpu')
    LDCT_LM = CreateLungMasks(LDCT_tensor, save_root + "LDCT", True)
    LDCT_cropped, LDCT_LM_cropped = CropBinary_monai(LDCT_tensor, torch.from_numpy(LDCT_LM))
    PET_cropped, _ = CropBinary_monai(pet_Resampled, torch.from_numpy(LDCT_LM))
    LDCT_spaced, LDCT_LM_spaced = OnlySpacing_fun(LDCT_cropped, LDCT_LM_cropped, data_dicts[0]["LDCT"])
    PET_spaced, _ = OnlySpacing_fun(PET_cropped, LDCT_LM_cropped, data_dicts[0]["LDCT"])
    # ldct_intensity = OnlyIntensity_fun(LDCT_spaced,0)
    ldct_clinic, ldctLM_clinic = cropCTfromROI_ClinicalInfo_v2(LDCT_spaced, LDCT_LM_spaced, clinicInfo_path, pxID)
    pet_clinic, _ = cropCTfromROI_ClinicalInfo_v2(PET_spaced, LDCT_LM_spaced, clinicInfo_path, pxID)

    if True:
        save_nifti_without_header(LDCT_spaced[0].numpy(), filename=save_root + "LDCT_LungCropped.nii.gz")
        save_nifti_without_header(LDCT_LM_spaced[0].numpy(), filename=save_root + "LDCT_LungCropped_LungMask.nii.gz")

        save_nifti_without_header(ldct_clinic[0].numpy(), filename=save_root + "LDCT_ClinicC.nii.gz")
        save_nifti_without_header(ldctLM_clinic[0].numpy(), filename=save_root + "LDCT_ClinicC_LungMask.nii.gz")

        save_nifti_without_header(PET_spaced[0].numpy(), filename=save_root + "PET_LungCropped.nii.gz")
        save_nifti_without_header(pet_clinic[0].numpy(), filename=save_root + "PET_ClinicC.nii.gz")

    # PLAN CT
    # Intensity PlanCT - Missing
    PlanCT_LM = CreateLungMasks(PlanCT_tensor, save_root + "PlanCT", True)
    PlanCT_cropped, PlanCT_LM_cropped = CropBinary_monai(PlanCT_tensor, torch.from_numpy(PlanCT_LM))
    ITV_cropped, _ = CropBinary_monai(ITV_tensor, torch.from_numpy(PlanCT_LM))
    planCT_spaced, planCTLM_spaced = OnlySpacing_fun(PlanCT_cropped, PlanCT_LM_cropped, data_dicts[0]["PlanCT"])
    itv_spaced, _ = OnlySpacing_fun(ITV_cropped, PlanCT_LM_cropped, data_dicts[0]["PlanCT"])
    # planCT_intensity = OnlyIntensity_fun(planCT_spaced, selectVal_opt=0)
    planct_clinic, planctLM_clinic = cropCTfromROI_ClinicalInfo_v2(planCT_spaced, planCTLM_spaced, clinicInfo_path,
                                                                   pxID)
    itv_clinic, _ = cropCTfromROI_ClinicalInfo_v2(itv_spaced, planCTLM_spaced, clinicInfo_path, pxID)

    if True:
        save_nifti_without_header(planCT_spaced[0].numpy(), filename=save_root + "PlanCT_LungCropped.nii.gz")
        save_nifti_without_header(planCTLM_spaced[0].numpy(), filename=save_root + "PlanCT_LungCropped_LungMask.nii.gz")
        save_nifti_without_header(itv_spaced[0].numpy(), filename=save_root + "ITV_LungCropped.nii.gz")

        save_nifti_without_header(planct_clinic[0].numpy(), filename=save_root + "PlanCT_Clinic.nii.gz")
        save_nifti_without_header(planctLM_clinic[0].numpy(), filename=save_root + "PlanCT_Clinic_LungMask.nii.gz")
        save_nifti_without_header(itv_clinic[0].numpy(), filename=save_root + "ITV_Clinic_.nii.gz")

    print("Target Lung Cropped shapes:", planCT_spaced.shape, planCTLM_spaced.shape, itv_spaced.shape)
    print("Moving Lung Cropped shapes:", LDCT_spaced.shape, LDCT_LM_spaced.shape, PET_spaced.shape)

    print("Target Clinic shapes:", planct_clinic.shape, planctLM_clinic.shape, itv_clinic.shape)
    print("Moving Clinic shapes:", ldct_clinic.shape, ldctLM_clinic.shape, pet_clinic.shape)

    return 0