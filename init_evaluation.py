import torch
import os
import csv
import numpy as np
from main_fun_peregrine import *
from main_fun_peregrine_evaluation import *
#from deleteFiles_fun import *


if __name__ == "__main__":

	device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device:', device_cuda)

	#Laptop
	#nifti_root = "/home/umcg/Desktop/Ch2/Data/Data4Registration/Registration5/"
	#clinicInfo_path = os.path.join(nifti_root,"CollectedwClinicalInfo.csv")
	#save_Intermediate = "/home/umcg/Desktop/Ch2/Data/Data4Registration/Registration5_Intermediate/"
	#save_newRegistered = "/home/umcg/Desktop/Ch2/Data/Data4Registration/Registration5_Registered/"
	#save_CSVs = nifti_root
	#UMCG
	#nifti_root  = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti/"
	#clinicInfo_path = "C:/Users/delaOArevaLR/OneDrive - UMCG/Code/Code_From_Umcg/RegistrationCode/CollectedwClinicalInfo.csv"
	#save_Intermediate = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_reshaped/"
	#save_newRegistered = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_registered/"
	#save_CSVs = "C:/Users/delaOArevaLR/OneDrive - UMCG/Code/Code_From_Umcg/"
	#PEregrine
	nifti_root = "/scratch/p308104/newDicomData_Nifti_reshaped/"
	save_Intermediate = "/scratch/p308104/RegistratedNii_8versions/"
	save_newRegistered = "/scratch/p308104/RegistratedNii_8versions/"
	save_CSVs = "/home1/p308104/Registration_fun/"
	
	registeredPx = "/home1/p308104/Registration_fun/RegisteredPatientsv4_p3.txt"
	ids = np.loadtxt(registeredPx, dtype=int)
	registeredPx_list = ids.tolist()

	#_ = delete_files_with_word(save_Intermediate, "Clinic")

	total_px = []
	for root, _, _ in os.walk(nifti_root, topdown=False):
		patientID = root.split("/")[-1]
		if int(patientID) in registeredPx_list:
			print("PxID: ",patientID)
			pxok = main_peregrine_evaluation(nifti_root,patientID,device_cuda,save_Intermediate,save_newRegistered,save_CSVs)
		else:
		    print("Resgistered previously", patientID)
			
	if False:
		with open(save_CSVs+"ReviewOkPx_v4.csv", "a", newline="") as file_tmp:
			writer = csv.writer(file_tmp)
			writer.writerow(id_column)
			writer.writerow(total_px)

	print("THE END")

	
