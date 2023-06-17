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
from displayRegist_fun import *
from Resample_fun import *
import matplotlib.pyplot as plt

import torch


def main(nifti_root,clinicInfo_path,pxID):
	file_path = os.path.join(nifti_root,str(pxID))
	save_root = nifti_root+str(pxID)+"/"

	#Find Paths
	planCT_path,ldct_path,data_dicts= FilesPerPatient(file_path)

	#Main - len(itv)!=0 and len(plan_ct)!=0 and len(plan_ct_LM)!=0 and len(pet_filename)!=0:
	if True: #len(ldct)!=0 and len(ldct_LM)!=0 and len(pet)!=0:
		#Read and Orient
		PlanCT_tensor,ITV_tensor,LDCT_tensor,PET_tensor = ReadAndOrient_monai(data_dicts)
		# LDCT and PET
		# PET can not be crop with the first lung mask because x,y is different spacing than LM, so we first need to space it properly and then crop it with the lung mask.
		# To do so I think we need to separate Resampling into a different function with spacing, so we can Space it with the apropiate path, without needing to Rescale it.
		# ResamplingSpacingPET
		# CreateLungMask_LDCT
		# CropBinaryLDCT
		# CropBinaryPET
		# SpacingLDCT
		# SpacingPET
		# CropClinicalLDCT
		# CropClinicalPET
		# IntensityLDCT

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
		print("Moving shapes:", ldct_clinic.shape, ldctLM_clinic.shape, pet_clinic.shape)

		if True:
			save_nifti_without_header(LDCT_spaced[0].numpy(), filename="LDCT_LungCropped.nii.gz")
			save_nifti_without_header(LDCT_LM_spaced[0].numpy(), filename="LDCT_LungCropped_LungMask.nii.gz")

			save_nifti_without_header(ldct_clinic[0].numpy(), filename="LDCT_ClinicC.nii.gz")
			save_nifti_without_header(ldctLM_clinic[0].numpy(), filename="LDCT_ClinicC_LungMask.nii.gz")

		#PLAN CT
		#CreateLungMask PlanCT
		# CropBinary PlanCT
		# CropBinarITV
		# Spacing PLanCT
		# Spacing ITV
		# CropClinical PlanCT
		# CropClinical ITV
		# Intensity PlanCT
		PlanCT_LM = CreateLungMasks(PlanCT_tensor, save_root + "PlanCT", True)
		PlanCT_cropped, PlanCT_LM_cropped = CropBinary_monai(PlanCT_tensor, torch.from_numpy(PlanCT_LM))
		ITV_cropped, _ = CropBinary_monai(ITV_tensor, torch.from_numpy(PlanCT_LM))
		planCT_spaced,planCTLM_spaced = OnlySpacing_fun(PlanCT_cropped, PlanCT_LM_cropped,data_dicts[0]["PlanCT"])
		itv_spaced, _ = OnlySpacing_fun(ITV_cropped, PlanCT_LM_cropped,data_dicts[0]["PlanCT"])
		#planCT_intensity = OnlyIntensity_fun(planCT_spaced, selectVal_opt=0)
		planct_clinic, planctLM_clinic = cropCTfromROI_ClinicalInfo_v2(planCT_spaced, planCTLM_spaced, clinicInfo_path,patientID)
		itv_clinic, _ = cropCTfromROI_ClinicalInfo_v2(itv_spaced, planCTLM_spaced, clinicInfo_path,patientID)
		print("Target shapes:",planct_clinic.shape,planctLM_clinic.shape,itv_clinic.shape)


		if False:
			save_nifti_without_header(planct_spaced[0].numpy(), filename="PlanCT_Spaced.nii.gz")
			save_nifti_without_header(planctLM_spaced[0].numpy(), filename="PlanCT_Spaced_LungMask.nii.gz")

			save_nifti_without_header(planct_clinic[0].numpy(), filename="PlanCT_Clinic.nii.gz")
			save_nifti_without_header(planctLM_clinic[0].numpy(), filename="PlanCT_Clinic_LungMask.nii.gz")


		display_LoadImgs(ldct_clinic,)
		#Register
		registCT1,registPET1 = Register_fun(planCt_cropped_1,ldct_cropped_1,pet_cropped_1,pxID)

		registCT1_norm = registCT1/np.max(registCT1)
		registPET1_norm = (registPET1/np.max(registPET1))*5

		displayRegist(registCT1_norm+registPET1_norm,registCT2_norm+registPET2_norm,registCT3_norm+registPET3_norm,itv_cropped_1,planCt_cropped_1,itv_cropped_1)

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
	
