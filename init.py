import torch
import os
import csv
from main_fun import *

#from deleteFiles_fun import *


if __name__ == "__main__":

	device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device:', device_cuda)

	#nifti_root = "/home/umcg/Desktop/Ch2/Data/Registration5/"
	#clinicInfo_path = os.path.join(nifti_root,"CollectedwClinicalInfo.csv")
	#save_Intermediate = "/home/umcg/Desktop/Ch2/Data/Registration5_Intermediate/"
	#save_newRegistered = "/home/umcg/Desktop/Ch2/Data/Registration5_Registered/"

	#conda activate name0
	nifti_root  = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti/"
	clinicInfo_path = "C:/Users/delaOArevaLR/OneDrive - UMCG/Code/Code_From_Umcg/RegistrationCode/CollectedwClinicalInfo.csv"
	save_Intermediate = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_reshaped/"
	save_newRegistered = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_registered/"
	save_CSVs = "C:/Users/delaOArevaLR/OneDrive - UMCG/Code/Code_From_Umcg/"

	id_column = clinicInfo_idcolumn(clinicInfo_path)

	#_ = delete_files_with_word(save_Intermediate, "Clinic")

	total_px = []
	for patientID in id_column:
		print(patientID)
		pxok = main(nifti_root,clinicInfo_path,patientID,device_cuda,save_Intermediate,save_newRegistered,save_CSVs)
		total_px.append(pxok)
	
	with open(save_CSVs+"ReviewOkPx_v2.csv", "a", newline="") as file_tmp:
		writer = csv.writer(file_tmp)
		writer.writerow(id_column)
		writer.writerow(total_px)

	print("THE END")

	
