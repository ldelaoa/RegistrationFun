from monai.utils import first, set_determinism
import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Rotate90d,
    ToTensord,
    EnsureChannelFirstd,
)

#import os.path
#from nibabel.processing import resample_to_output
import nibabel as nib
#import numpy as np


def save_nifti_without_header(data, filename):
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, filename)


def OnlyRead_registered(dictionary,ClinicorLungbool):
    image_keys_lungCrop = ["ldctLung_v1","ldctLung_v2","petLung_v1","petLung_v2"]
    image_keys_Clinic = ["ldctClinic_v1", "ldctClinic_v2", "petClinic_v1", "petClinic_v2"]
    #image_keys_lungCrop = ["ldctLung_v2", "petLung_v2"]
    #image_keys_Clinic = ["ldctClinic_v2", "petClinic_v2"]

    if ClinicorLungbool:
        image_keys = image_keys_lungCrop
    else:
        image_keys = image_keys_Clinic

    load_transforms = Compose(
        [LoadImaged(keys=image_keys,image_only=True),EnsureChannelFirstd(keys=image_keys),])

    check_ds = Dataset(data=dictionary[:], transform=load_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)

    if ClinicorLungbool:
        ldctLung_v1_t,ldctLung_v2_t,petLung_v1_t,petLung_v2_t = (batch_data["ldctLung_v1"],batch_data["ldctLung_v2"],batch_data["petLung_v1"],batch_data["petLung_v2"])
        return ldctLung_v1_t,ldctLung_v2_t,petLung_v1_t,petLung_v2_t
    else:
        ldctClinic_v1_t,ldctClinic_v2_t,petClinic_v1_t,petClinic_v2_t = (batch_data["ldctClinic_v1"],batch_data["ldctClinic_v2"],batch_data["petClinic_v1"],batch_data["petClinic_v2"])
        return ldctClinic_v1_t,ldctClinic_v2_t,petClinic_v1_t,petClinic_v2_t
    

def OnlyRead_Intermediate(dictionary,LungCropTensors_bool,clinicTensors_bool):
    image_keys_lungCrop = ["PlanCT_LungCrop","ITV_LungCrop","PlanCT_LungMask_LungCrop","LDCT_LungCrop","PET_LungCrop","LDCT_LungMask_LungCrop"]
    image_keys_Clinic = ["PlanCT_Clinic", "ITV_Clinic","PlanCT_LungMask_Clinic","LDCT_Clinic", "PET_Clinic","LDCT_LungMask_Clinic"]
    if not LungCropTensors_bool and clinicTensors_bool: # Only Clinic TRUE
        image_keys = image_keys_Clinic
    if LungCropTensors_bool and not clinicTensors_bool:  # Only Clinic TRUE
        image_keys = image_keys_lungCrop

    load_transforms = Compose(
        [LoadImaged(keys=image_keys,image_only=True),EnsureChannelFirstd(keys=image_keys),])

    check_ds = Dataset(data=dictionary[:], transform=load_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)

    if LungCropTensors_bool and not clinicTensors_bool: # Only LungCrop TRUE
        PlanCT_LungCrop_tensor, ITV_LungCrop_tensor, PlanCT_LungMask_LungCrop_tensor = (batch_data["PlanCT_LungCrop"][0],batch_data["ITV_LungCrop"][0],batch_data["PlanCT_LungMask_LungCrop"][0])
        LDCT_LungCrop_tensor, PET_LungCrop_tensor, LDCT_LungMask_LungCrop_tensor = (batch_data["LDCT_LungCrop"][0],batch_data["PET_LungCrop"][0],batch_data["LDCT_LungMask_LungCrop"][0])

        return PlanCT_LungCrop_tensor,ITV_LungCrop_tensor,PlanCT_LungMask_LungCrop_tensor,LDCT_LungCrop_tensor,PET_LungCrop_tensor,LDCT_LungMask_LungCrop_tensor

    if not LungCropTensors_bool and clinicTensors_bool:  # Only Clinic TRUE
        PlanCT_Clinic_tensor, ITV_Clinic_tensor, PlanCT_LungMask_Clinic_tensor = (batch_data["PlanCT_Clinic"][0], batch_data["ITV_Clinic"][0], batch_data["PlanCT_LungMask_Clinic"][0])
        LDCT_Clinic_tensor, PET_Clinic_tensor, LDCT_LungMask_Clinic_tensor = (batch_data["LDCT_Clinic"][0], batch_data["PET_Clinic"][0], batch_data["LDCT_LungMask_Clinic"][0])

        return PlanCT_Clinic_tensor, ITV_Clinic_tensor, PlanCT_LungMask_Clinic_tensor, LDCT_Clinic_tensor, PET_Clinic_tensor, LDCT_LungMask_Clinic_tensor


def ReadAndOrient_monai(dictionary,device):
    image_keys = ["PlanCT", "ITV", "LDCT","PET"]

    load_transforms = Compose(
        [
            LoadImaged(keys=image_keys,image_only=True),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=["PlanCT", "ITV"], axcodes="LAS"), #(L', 'R'), ('P', 'A'), ('I', 'S'))
            Orientationd(keys=["LDCT","PET"], axcodes="LAS"),  # (L', 'R'), ('P', 'A'), ('I', 'S'))
            Rotate90d(keys=image_keys, k=1, spatial_axes=(0, 1)),
            ToTensord(keys=image_keys),
        ]
    )

    check_ds = Dataset(data=dictionary[:], transform=load_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)
    PlanCT_tensor, ITV_tensor = (batch_data["PlanCT"][0][0], batch_data["ITV"][0][0])
    LDCT_tensor, PET_tensor = (batch_data["LDCT"][0][0].to(device), batch_data["PET"][0][0].to(device))
    return PlanCT_tensor,ITV_tensor,LDCT_tensor,PET_tensor


def OnlyRead_registered_dynamic(dictionary,regist_v):
    word_ct = "ldctLung_v"
    word_pet = "petLung_v"
    image_keys = [word_ct, word_pet]

    load_transforms = Compose(
        [LoadImaged(keys=image_keys,image_only=True),EnsureChannelFirstd(keys=image_keys),])

    check_ds = Dataset(data=dictionary, transform=load_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)
    ldctLung_v,petLung_v = (batch_data["ldctLung_v"],batch_data["petLung_v"])
    return ldctLung_v,petLung_v
