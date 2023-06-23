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
from mainCrop_fun import *
from mainRegister_fun import *
from mainEvaluation_fun import *

import torch


def main(nifti_root,clinicInfo_path,pxID,device,save_path,save_Registered):
	file_path = os.path.join(nifti_root,str(pxID))
	save_root = save_path+str(pxID)+"/"
	if not os.path.exists(save_root):
		os.makedirs(save_root)
	save_register = save_Registered+str(pxID)+"/"
	if not os.path.exists(save_register):
		os.makedirs(save_register)
	planCT_path,ldct_path,data_dicts= FilesPerPatient(file_path)
	intermediate_dict = FilesperPatient_Inter_LungCroped(save_root)
	registered_dict = FilesPerPatient_Registered(save_register)

	print("Init",len(data_dicts),"Inter",len(intermediate_dict),"Regist",len(registered_dict))

	#Check there is patient data, otherwise end loop
	if len(data_dicts)==0 and len(intermediate_dict)==0 and len(registered_dict)==0:
		print("Missing Images Patient")
		return 1

	#Read Raw Images and Crop to Lung and Clinic Specs
	if len(intermediate_dict)==0 and len(data_dicts)==1:
		mainCrop(save_root,data_dicts,device,pxID,clinicInfo_path)
		intermediate_dict = FilesperPatient_Inter_LungCroped(save_root)

	#Read Lung and Clinic Cropped images and register them
	if len(intermediate_dict)==1 and len(registered_dict)==0:
		mainRegister(save_register, intermediate_dict, pxID)
		registered_dict = FilesPerPatient_Registered(save_register)

	#Evaluate Registration specifically in the ITV area
	if len(intermediate_dict)==1 and len(registered_dict)==1:
		mainEval(save_register, registered_dict, intermediate_dict, pxID)

	return 0
