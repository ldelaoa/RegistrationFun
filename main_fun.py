#import os
#import numpy as np
#from os.path import join
#from datetime import time, datetime
#import glob
#from skimage.draw import polygon
#import SimpleITK as sitk
#import pydicom as dicom


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

import torch


def main(nifti_root,clinicInfo_path,pxID):
	file_path = os.path.join(nifti_root,str(pxID))
	save_root = nifti_root+str(pxID)+"/IntermediateImages/"

	#Find Paths
	planCT_path,ldct_path,data_dicts= FilesPerPatient(file_path)

	intermediate_dict = FilesperPatient_Inter_LungCroped(save_root)

	#Main - len(itv)!=0 and len(plan_ct)!=0 and len(plan_ct_LM)!=0 and len(pet_filename)!=0:

	if len(intermediate_dict)==0: #len(ldct)!=0 and len(ldct_LM)!=0 and len(pet)!=0:
		print("Creating images for registration")
		#Read and Orient
		PlanCT_tensor,ITV_tensor,LDCT_tensor,PET_tensor = ReadAndOrient_monai(data_dicts)
		# LDCT and PET
		# IntensityLDCT - Missing

		pet_Resampled = OnlyResamplingPET(LDCT_tensor, PET_tensor)
		print(pet_Resampled.shape,LDCT_tensor.shape)
		LDCT_LM = CreateLungMasks(LDCT_tensor, save_root + "LDCT", True)
		LDCT_cropped, LDCT_LM_cropped = CropBinary_monai(LDCT_tensor, torch.from_numpy(LDCT_LM))
		PET_cropped, _ = CropBinary_monai(pet_Resampled, torch.from_numpy(LDCT_LM))
		LDCT_spaced,LDCT_LM_spaced = OnlySpacing_fun(LDCT_cropped, LDCT_LM_cropped, data_dicts[0]["LDCT"])
		PET_spaced, _ = OnlySpacing_fun(PET_cropped, LDCT_LM_cropped, data_dicts[0]["LDCT"])
		#ldct_intensity = OnlyIntensity_fun(LDCT_spaced,0)
		ldct_clinic, ldctLM_clinic = cropCTfromROI_ClinicalInfo_v2(LDCT_spaced, LDCT_LM_spaced, clinicInfo_path,patientID)
		pet_clinic, _ = cropCTfromROI_ClinicalInfo_v2(PET_spaced, LDCT_LM_spaced, clinicInfo_path, patientID)

		if True:
			save_nifti_without_header(LDCT_spaced[0].numpy(), filename=save_root+"LDCT_LungCropped.nii.gz")
			save_nifti_without_header(LDCT_LM_spaced[0].numpy(), filename=save_root+"LDCT_LungCropped_LungMask.nii.gz")

			save_nifti_without_header(ldct_clinic[0].numpy(), filename=save_root+"LDCT_ClinicC.nii.gz")
			save_nifti_without_header(ldctLM_clinic[0].numpy(), filename=save_root+"LDCT_ClinicC_LungMask.nii.gz")

			save_nifti_without_header(PET_spaced[0].numpy(), filename=save_root+"PET_LungCropped.nii.gz")
			save_nifti_without_header(pet_clinic[0].numpy(), filename=save_root+"PET_ClinicC.nii.gz")

		#PLAN CT
		# Intensity PlanCT - Missing
		PlanCT_LM = CreateLungMasks(PlanCT_tensor, save_root + "PlanCT", True)
		PlanCT_cropped, PlanCT_LM_cropped = CropBinary_monai(PlanCT_tensor, torch.from_numpy(PlanCT_LM))
		ITV_cropped, _ = CropBinary_monai(ITV_tensor, torch.from_numpy(PlanCT_LM))
		planCT_spaced,planCTLM_spaced = OnlySpacing_fun(PlanCT_cropped, PlanCT_LM_cropped,data_dicts[0]["PlanCT"])
		itv_spaced, _ = OnlySpacing_fun(ITV_cropped, PlanCT_LM_cropped,data_dicts[0]["PlanCT"])
		#planCT_intensity = OnlyIntensity_fun(planCT_spaced, selectVal_opt=0)
		planct_clinic, planctLM_clinic = cropCTfromROI_ClinicalInfo_v2(planCT_spaced, planCTLM_spaced, clinicInfo_path,patientID)
		itv_clinic, _ = cropCTfromROI_ClinicalInfo_v2(itv_spaced, planCTLM_spaced, clinicInfo_path,patientID)

		if True:
			save_nifti_without_header(planCT_spaced[0].numpy(), filename=save_root+"PlanCT_LungCropped.nii.gz")
			save_nifti_without_header(planCTLM_spaced[0].numpy(), filename=save_root+"PlanCT_LungCropped_LungMask.nii.gz")
			save_nifti_without_header(itv_spaced[0].numpy(), filename=save_root + "ITV_LungCropped.nii.gz")

			save_nifti_without_header(planct_clinic[0].numpy(), filename=save_root+"PlanCT_Clinic.nii.gz")
			save_nifti_without_header(planctLM_clinic[0].numpy(), filename=save_root+"PlanCT_Clinic_LungMask.nii.gz")
			save_nifti_without_header(itv_clinic[0].numpy(), filename=save_root + "ITV_Clinic_.nii.gz")

		print("Target Lung Cropped shapes:", planCT_spaced.shape, planCTLM_spaced.shape, itv_spaced.shape)
		print("Moving Lung Cropped shapes:", LDCT_spaced.shape, LDCT_LM_spaced.shape, PET_spaced.shape)

		print("Target Clinic shapes:", planct_clinic.shape, planctLM_clinic.shape, itv_clinic.shape)
		print("Moving Clinic shapes:", ldct_clinic.shape, ldctLM_clinic.shape, pet_clinic.shape)

		intermediate_dict = FilesperPatient_Inter_LungCroped(save_root)

	if len(intermediate_dict)==1:
		print("Inside Registration module")

		if False:
			PlanCT_LungCrop_tensor,ITV_LungCrop_tensor,PlanCT_LungMask_LungCrop_tensor,LDCT_LungCrop_tensor,PET_LungCrop_tensor,LDCT_LungMask_LungCrop_tensor = OnlyRead_Intermediate(intermediate_dict, True, False)
			registCT1_LM,registPET1_LM = Register_fun(PlanCT_LungCrop_tensor[0],LDCT_LungCrop_tensor[0],PET_LungCrop_tensor[0],pxID)
			registCT2_LM, registPET2_LM = Register_fun_v2(PlanCT_LungCrop_tensor[0], LDCT_LungCrop_tensor[0], PET_LungCrop_tensor[0], pxID)

		PlanCT_Clinic_tensor, ITV_Clinic_tensor, PlanCT_LungMask_Clinic_tensor, LDCT_Clinic_tensor, PET_Clinic_tensor, LDCT_LungMask_Clinic_tensor = OnlyRead_Intermediate(intermediate_dict, False, True)
		registCT1_Clinic, registPET1_Clinic = Register_fun(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0], PET_Clinic_tensor[0], pxID)
		registCT2_Clinic, registPET2_Clinic = Register_fun_v2(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0], PET_Clinic_tensor[0], pxID)
		registCT3_Clinic, registPET3_Clinic = Register_fun_v3(PlanCT_Clinic_tensor[0], LDCT_Clinic_tensor[0],PET_Clinic_tensor[0], pxID)


		return 0

	else:
		print("Incomplete Patient")
		return 1


if __name__ == "__main__":

	nifti_root = "/home/umcg/Desktop/Ch2/Data/Registration5/"
	clinicInfo_path = os.path.join(nifti_root,"CollectedwClinicalInfo.csv")

	#nifti_root  = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti/"
	#clinicInfo = "C:/Users/delaOArevaLR/OneDrive - UMCG/Code/Code_From_Umcg/RegistrationCode/CollectedwClinicalInfo.csv"
	#save_newFolder = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_reshaped/"

	id_column = clinicInfo_idcolumn(clinicInfo_path)

	count_incomplete = []
	total_px = 0
	for patientID in id_column:
		print(patientID)
		if patientID != 32628:
			main(nifti_root,clinicInfo_path,patientID)

		total_px +=1
		if total_px==4:
			break
	
