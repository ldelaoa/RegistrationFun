#import os
#import numpy as np
#from os.path import join
#from datetime import time, datetime
#import glob
#from skimage.draw import polygon
#import SimpleITK as sitk
#import pydicom as dicom
#from pydicom.tag import Tag
#import sys


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
		#From Path to NP - For now only LDCT, LDCT_LM and PET
		PlanCT_tensor, LDCT_tensor = ReadAndOrient_monai(data_dicts)

		PlanCT_LM = CreateLungMasks(PlanCT_tensor, save_root + "PlanCT", True)
		PlanCT_cropped, PlanCT_LM_cropped = CropBinary_monai(PlanCT_tensor, torch.from_numpy(PlanCT_LM))
		planct_spaced,planctLM_spaced = SpacingAndResampleToMatch(data_dicts[0]["PlanCT"], PlanCT_cropped, PlanCT_LM_cropped)
		planct_clinic, planctLM_clinic = cropCTfromROI_ClinicalInfo_v2(planct_spaced, planctLM_spaced, clinicInfo_path,patientID)
		if False:
			save_nifti_without_header(planct_spaced[0].numpy(), filename="PlanCT_Spaced.nii.gz")
			save_nifti_without_header(planctLM_spaced[0].numpy(), filename="PlanCT_Spaced_LungMask.nii.gz")

			save_nifti_without_header(planct_clinic[0].numpy(), filename="PlanCT_Clinic.nii.gz")
			save_nifti_without_header(planctLM_clinic[0].numpy(), filename="PlanCT_Clinic_LungMask.nii.gz")


		LDCT_LM = CreateLungMasks(LDCT_tensor, save_root + "LDCT", True)
		LDCT_cropped, LDCT_LM_cropped = CropBinary_monai(LDCT_tensor, torch.from_numpy(LDCT_LM))
		ldct_spaced, ldctLM_spaced = SpacingAndResampleToMatch(data_dicts[0]["LDCT"], LDCT_cropped, LDCT_LM_cropped)
		ldct_clinic, ldctLM_clinic = cropCTfromROI_ClinicalInfo_v2(ldct_spaced, ldctLM_spaced, clinicInfo_path,patientID)
		if False:
			save_nifti_without_header(ldct_spaced[0].numpy(), filename="LDCT_Spaced.nii.gz")
			save_nifti_without_header(ldctLM_spaced[0].numpy(), filename="LDCT_Spaced_LungMask.nii.gz")

			save_nifti_without_header(ldct_clinic[0].numpy(), filename="LDCT_ClinicC.nii.gz")
			save_nifti_without_header(ldctLM_clinic[0].numpy(), filename="LDCT_ClinicC_LungMask.nii.gz")

		exit(0)


		#Register
		registCT1,registPET1 = Register_fun(planCt_cropped_1,ldct_cropped_1,pet_cropped_1,pxID)
		registCT2,registPET2 = Register_fun(planCt_cropped_1,ldct_cropped_2,pet_cropped_2,pxID)
		registCT3,registPET3 = Register_fun(planCt_cropped_1,ldct_cropped_3,pet_cropped_3,pxID)

		registCT1_norm = registCT1/np.max(registCT1)
		registPET1_norm = (registPET1/np.max(registPET1))*5

		registCT2_norm = registCT2/np.max(registCT2)
		registPET2_norm = (registPET2/np.max(registPET2))*5

		registCT3_norm = registCT3/np.max(registCT3)
		registPET3_norm = (registPET3/np.max(registPET3))*5

		#displayRegist(registCT1,registCT2,registCT3,itv_np,planCt_np,planCt_LM_np)
		displayRegist(registCT1,registCT2,registCT3,itv_cropped_1,planCt_cropped_1,itv_cropped_1)
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
	
