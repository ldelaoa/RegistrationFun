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

		LDCT_LM = CreateLungMasks(LDCT_tensor, save_root+"LDCT",True)
		#display_LoadImgs(LDCT_tensor.numpy(), LDCT_LM)
		LDCT_cropped, LDCT_LM_cropped = CropBinary_monai(LDCT_tensor, torch.from_numpy(LDCT_LM))
		display_LoadImgs(LDCT_cropped[0].numpy(), LDCT_LM_cropped[0].numpy())

		PlanCT_LM = CreateLungMasks(PlanCT_tensor, save_root + "PlanCT", True)
		# display_LoadImgs(PlanCT_tensor.numpy(), PlanCT_LM)
		PlanCT_cropped, PlanCT_LM_cropped = CropBinary_monai(PlanCT_tensor, torch.from_numpy(PlanCT_LM))
		print(PlanCT_cropped.shape, PlanCT_LM_cropped.shape)
		display_LoadImgs(PlanCT_cropped[0].numpy(), PlanCT_LM_cropped[0].numpy())

		exit(0)

		if False:


			#Cropping by method
			ldct_cropped_1,pet_cropped_1=cropCTfromROI_lung(ldct_np,ldct_LM_np,pet_np,pet_bool=True)
			ldct_cropped_2,pet_cropped_2 = cropCTfromROI_ClinicalInfo(ldct_np,ldct_LM_np,itv,clinicInfo_path,pxID,pet_np)
			ldct_cropped_3,pet_cropped_3 = cropCTfromROI_BinaryPET(ldct_cropped_1,pet_cropped_1)

			# Crop Plan CT
			planCt_cropped_1, _ = cropCTfromROI_lung(planCt_np, planCt_LM_np, None, pet_bool=False)
			itv_cropped_1,_ = cropCTfromROI_lung(itv_np, planCt_LM_np, None, pet_bool=False)
			planCt_LM_cropped_1,_ = cropCTfromROI_lung(planCt_LM_np, planCt_LM_np, None, pet_bool=False)

			np.savez(nifti_root+'MovingArrays.npz', array1=ldct_cropped_1, array2=ldct_cropped_2,array3=ldct_cropped_3)
			np.savez(nifti_root+'TargetArrays.npz',array1=planCt_cropped_1,array2=planCt_LM_cropped_1,array3=itv_cropped_1)



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
	
