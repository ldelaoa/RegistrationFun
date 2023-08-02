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
from mainCrop_fun_v2 import *


import torch


def main_peregrine(nifti_root,pxID,device,save_path,save_Registered,save_CSVs):
	file_path = os.path.join(nifti_root,str(pxID))
	save_root = save_path+str(pxID)+"/"
	if not os.path.exists(save_root):
		os.makedirs(save_root)
	save_register = save_Registered+str(pxID)+"/"
	if not os.path.exists(save_register):
		os.makedirs(save_register)
	
	#Create and look for Dictionaries of raw nifti, cropped and registered
	intermediate_dict = FilesperPatient_Inter_LungCroped(nifti_root+str(pxID)+"/")

	#Read Lung and Clinic Cropped images and register them
	if len(intermediate_dict)==1:
		mainRegister(save_register, intermediate_dict, pxID,save_CSVs)
		mainEval_dynamic(save_register,intermediate_dict,pxID,save_CSVs)
		#Evaluate Registration specifically in the ITV area

	return 0
