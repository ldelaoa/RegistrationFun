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
#import matplotlib.pyplot as plt

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


def main(nifti_root,clinicInfo_path,pxID):
	print(pxID)
	file_path = os.path.join(nifti_root,str(pxID))
	save_root = nifti_root+str(pxID)+"/"

	#Find Paths
	planCt,planCt_LM,ldct,ldct_LM,pet,itv,tolungmask= FilesPerPatient(file_path)

	#Make Lung Mask if necesary
	if (len(planCt_LM) == 0 and len(planCt) != 0) or (len(ldct_LM) == 0 and len(ldct) != 0):
		CreateLungMasks(tolungmask,save_root)
		planCt,planCt_LM,ldct,ldct_LM,pet,itv,tolungmask= FilesPerPatient(file_path)
	#Main - len(itv)!=0 and len(plan_ct)!=0 and len(plan_ct_LM)!=0 and len(pet_filename)!=0:
	if len(ldct)!=0 and len(ldct_LM)!=0 and len(pet)!=0:
		#From Path to NP - For now only LDCT, LDCT_LM and PET
		ldct_np,ldct_LM_np,pet_np,planCt_np,planCt_LM_np,itv_np = ReadAndResample(ldct,ldct_LM,pet,planCt,planCt_LM,itv)
		#display_LoadImgs(ldct_np,ldct_LM_np,pet_np,planCt_np,planCt_LM_np,itv_np)

		#Cropping by method
		ldct_cropped_1,pet_cropped_1=cropCTfromROI_lung(ldct_np,ldct_LM_np,pet_np,pet_bool=True)
		plan_ct_cropped_2,pet_cropped_2 = cropCTfromROI_ClinicalInfo(ldct_np,ldct_LM_np,itv,clinicInfo_path,pxID,pet_np)
		plan_ct_cropped_3,pet_cropped_3 = cropCTfromROI_BinaryPET(ldct_cropped_1,pet_cropped_1)

		# Crop Plan CT
		planCt_cropped_1, _ = cropCTfromROI_lung(planCt_np, planCt_LM_np, None, pet_bool=False)
		itv_cropped_1, _ = cropCTfromROI_lung(itv_np, planCt_LM_np, None, pet_bool=False)
		planCt_LM_cropped_1 = cropCTfromROI_lung(planCt_LM_np, planCt_LM_np, None, pet_bool=False)

		np.savez(nifti_root+'MovingArrays.npz', array1=ldct_cropped_1, array2=plan_ct_cropped_2,array3=plan_ct_cropped_3)
		np.savez(nifti_root+'TargetArrays.npz',array1=planCt_cropped_1,array2=planCt_LM_cropped_1,array3=itv_cropped_1)
		exit(0)


		#Register
		registCT1,registPET1 = Register_fun(planCt_cropped_1,ldct_cropped_1,pet_cropped_1,pxID)
		registCT2,registPET2 = Register_fun(planCt_cropped_1,plan_ct_cropped_2,pet_cropped_2,pxID)
		registCT3,registPET3 = Register_fun(planCt_cropped_1,plan_ct_cropped_3,pet_cropped_3,pxID)

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
		main(nifti_root,clinicInfo_path,patientID)
		break
	
