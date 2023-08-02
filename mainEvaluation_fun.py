import os.path
import csv

from clinicInfo_fun import clinicInfo_idcolumn
from FilesPerPatient_fun import *
from CreateLungMasks_fun import *
from display_LoadImgs_fun import *
from ReadAndResample_fun import *
from ReadAndResample_fun import OnlyRead_Intermediate
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



def mainEval(registered_dict,intermediate_dict,pxID,save_CSVs):
    print("Patient Already with Registered Images")
    tmp_path = save_CSVs + "Registration_metrics_v4.csv"

    ldctLung_v1_t,ldctLung_v2_t,petLung_v1_t,petLung_v2_t = OnlyRead_registered(registered_dict, True)#True is Lung Crop
    PlanCT_LungCrop_tensor, ITV_LungCrop_tensor, _, _, _, _ = OnlyRead_Intermediate(intermediate_dict, True, False)
    pet_LungCrop_binary1 = BinaryPET(petLung_v1_t)
    pet_LungCrop_binary2 = BinaryPET(petLung_v2_t)

    mse_avg_1,ssim_avg_1,psnr_avg_1 = similarMetrics(ldctLung_v1_t[0][0],PlanCT_LungCrop_tensor[0],ITV_LungCrop_tensor[0])
    mse_avg_2, ssim_avg_2, psnr_avg_2 = similarMetrics(ldctLung_v2_t[0][0], PlanCT_LungCrop_tensor[0],ITV_LungCrop_tensor[0])
    dice_1, haus_1 = metrics_fun_v1(torch.from_numpy(pet_LungCrop_binary1[0]), ITV_LungCrop_tensor)
    dice_2, haus_2 = metrics_fun_v1(torch.from_numpy(pet_LungCrop_binary2[0]), ITV_LungCrop_tensor)

    with open(tmp_path, "a", newline="") as file_tmp:
        writer = csv.writer(file_tmp)
        writer.writerow([pxID, "LungCrop", "Regist1", mse_avg_1, ssim_avg_1, psnr_avg_1, dice_1, haus_1])
        writer.writerow([pxID, "LungCrop", "Regist2", mse_avg_2, ssim_avg_2, psnr_avg_2, dice_2, haus_2])

    ldctClinic_v1_t,ldctClinic_v2_t,petClinic_v1_t,petClinic_v2_t = OnlyRead_registered(registered_dict, False)#False is Clinic
    PlanCT_ClinicCrop_tensor, ITV_ClinicCrop_tensor, _, _, _, _ = OnlyRead_Intermediate(intermediate_dict, False, True)
    pet_ClinicCrop_binary1 = BinaryPET(petClinic_v1_t)
    pet_ClinicCrop_binary2 = BinaryPET(petClinic_v2_t)

    mse_avg_11, ssim_avg_11, psnr_avg_11 = similarMetrics(ldctClinic_v1_t[0][0], PlanCT_ClinicCrop_tensor[0],ITV_ClinicCrop_tensor[0])
    mse_avg_22, ssim_avg_22, psnr_avg_22 = similarMetrics(ldctClinic_v2_t[0][0], PlanCT_ClinicCrop_tensor[0],ITV_ClinicCrop_tensor[0])
    dice_11, haus_11 = metrics_fun_v1(torch.from_numpy(pet_ClinicCrop_binary1[0]), ITV_ClinicCrop_tensor)
    dice_22, haus_22 = metrics_fun_v1(torch.from_numpy(pet_ClinicCrop_binary2[0]), ITV_ClinicCrop_tensor)

    with open(tmp_path, "a", newline="") as file_tmp:
        writer = csv.writer(file_tmp)
        writer.writerow([pxID, "ClinicCrop", "Regist1", mse_avg_11, ssim_avg_11, psnr_avg_11, dice_11, haus_11])
        writer.writerow([pxID, "ClinicCrop", "Regist2", mse_avg_22, ssim_avg_22, psnr_avg_22, dice_22, haus_22])


def mainEval_dynamic(file_path,intermediate_dict,pxID,save_CSVs):
    print("Patient with Registered Images")
    tmp_path = save_CSVs + "Registration_metrics_v4.csv"

    PlanCT_LungCrop_tensor, ITV_LungCrop_tensor, _, _, _, _ = OnlyRead_Intermediate(intermediate_dict, True, False)

    metrics_vector=[]
    for val0 in range(3):
        val=val0+9
        registered_dict = FilesPerPatient_Registered_dynamic(file_path,val)
        ldctLung_v1_t,petLung_v1_t = OnlyRead_registered_dynamic(registered_dict,val)
        pet_LungCrop_binary1 = BinaryPET(petLung_v1_t)
        mse_avg_1,ssim_avg_1,psnr_avg_1 = similarMetrics(ldctLung_v1_t[0][0],PlanCT_LungCrop_tensor[0],ITV_LungCrop_tensor[0])
        dice_1, haus_1 = metrics_fun_v1(torch.from_numpy(pet_LungCrop_binary1[0]), ITV_LungCrop_tensor)
        metrics_vector.append([mse_avg_1, ssim_avg_1, psnr_avg_1, dice_1, haus_1])

        with open(tmp_path, "a", newline="") as file_tmp:
            writer = csv.writer(file_tmp)
            writer.writerow([pxID, "LungCrop", "Regist"+str(val), mse_avg_1, ssim_avg_1, psnr_avg_1, dice_1, haus_1])
    print("End Evaluation")
    return 0