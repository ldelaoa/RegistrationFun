import torch
import os

from main_fun import *


if __name__ == "__main__":

	device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device:', device_cuda)

	#nifti_root = "/home/umcg/Desktop/Ch2/Data/Registration5/"
	#clinicInfo_path = os.path.join(nifti_root,"CollectedwClinicalInfo.csv")

	#conda activate name0
	nifti_root  = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti/"
	clinicInfo_path = "C:/Users/delaOArevaLR/OneDrive - UMCG/Code/Code_From_Umcg/RegistrationCode/CollectedwClinicalInfo.csv"
	save_newFolder = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_reshaped/"
	save_newRegistered = "//zkh/appdata/RTDicom/DAMEproject/new_DicomData_Nifti_registered/"

	id_column = clinicInfo_idcolumn(clinicInfo_path)

	total_px = []
	for patientID in id_column:
		print(patientID)
		pxok = main(nifti_root,clinicInfo_path,patientID,device_cuda,save_newFolder,save_newRegistered)
		total_px.append(pxok)
	print("THE END")
	
