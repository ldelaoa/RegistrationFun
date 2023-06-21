from monai.utils import first, set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Rotate90d,
    ToTensord,
)
import numpy as np
from monai.data import DataLoader, Dataset

#import os.path
#from nibabel.processing import resample_to_output
import nibabel as nib
#import numpy as np


def save_nifti_without_header(data, filename):
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, filename)


def OnlyRead_Intermediate(dictionary,LungCropTensors_bool,clinicTensors_bool):
    image_keys_lungCrop = ["PlanCT_LungCrop","ITV_LungCrop","PlanCT_LungMask_LungCrop","LDCT_LungCrop","PET_LungCrop","LDCT_LungMask_LungCrop"]
    image_keys_Clinic = ["PlanCT_Clinic", "ITV_Clinic","PlanCT_LungMask_Clinic","LDCT_Clinic", "PET_Clinic","LDCT_LungMask_Clinic"]
    if not LungCropTensors_bool and clinicTensors_bool: # Only Clinic TRUE
        image_keys = image_keys_Clinic
    if LungCropTensors_bool and not clinicTensors_bool:  # Only Clinic TRUE
        image_keys = image_keys_lungCrop

    load_transforms = Compose(
        [LoadImaged(keys=image_keys),EnsureChannelFirstd(keys=image_keys),])

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
            LoadImaged(keys=image_keys),
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
    PlanCT_tensor, ITV_tensor = (batch_data["PlanCT"][0][0].to(device), batch_data["ITV"][0][0].to(device))
    LDCT_tensor, PET_tensor = (batch_data["LDCT"][0][0].to(device), batch_data["PET"][0][0].to(device))
    return PlanCT_tensor,ITV_tensor,LDCT_tensor,PET_tensor


    def ReadAndResample_DEPRECATED(ldct_path, ldct_LM_path, pet_path, planCT_path, planCT_LM_path, itv_path):
        # target_voxel_sizes = (1,1,2)

        # Read Plan CT
        # planct_nii = nib.load(planCT_path[0])
        planct_reshaped = nib.load(planCT_path[0])
        target_voxel_sizes = planct_reshaped.header.get_zooms()
        # planct_reshaped = resample_to_output(planct_nii, voxel_sizes=target_voxel_sizes)
        planct_np = planct_reshaped.get_fdata()
        planct_np = np.transpose(planct_np, (2, 1, 0))

        # Read Plan CT Lung Mask
        # ct_LM_nii = nib.load(planCT_LM_path[0])
        # ct_LM_nii_np = ct_LM_nii.get_fdata()
        # ct_LM_nii_np = np.transpose(ct_LM_nii_np, (2, 1,0))
        # ct_LM_nii_2 =  nib.Nifti1Image(ct_LM_nii_np,ct_LM_nii.affine)
        # planct_LM_reshaped = resample_to_output(ct_LM_nii_2, voxel_sizes=target_voxel_sizes)
        planct_LM_reshaped = nib.load(planCT_LM_path[0])
        planct_LM_np = planct_LM_reshaped.get_fdata()
        # planct_LM_np = np.transpose(planct_LM_np, (2, 1,0))
        planct_LM_np = np.copy(np.round(planct_LM_np, decimals=0))

        # Read Tumor
        # itv_nii =nib.load(itv_path[0])
        # itv_reshaped = resample_to_output(itv_nii, voxel_sizes=target_voxel_sizes)
        itv_reshaped = nib.load(itv_path[0])
        itv_np = itv_reshaped.get_fdata()
        itv_np = np.transpose(itv_np, (2, 1, 0))
        itv_np = np.round(itv_np, decimals=0)

        # Read LowDose CT
        cropped_filename = ldct_path[0][:-7] + "_cropped.nii.gz"
        if os.path.exists(cropped_filename):
            print("Found cropped LDCT")
            ldct_nii = nib.load(cropped_filename)
            ldct_np = ldct_nii.get_fdata()
        else:
            ldct_nii = nib.load(ldct_path[0])
            ldct_reshaped = resample_to_output(ldct_nii, voxel_sizes=target_voxel_sizes)
            ldct_np = ldct_reshaped.get_fdata()
            ldct_np = np.transpose(ldct_np, (2, 1, 0))
            ldct_np = np.rot90(ldct_np, 2)
            ldct_np = ldct_np[:, :, ::-1]
            header = ldct_reshaped.header
            cropped_filename = ldct_path[0][:-7] + "_cropped.nii.gz"
            save_nifti_with_header(ldct_np, header, cropped_filename)

        # Read LowDose CT Lung Mask
        cropped_filename = ldct_LM_path[0][:-7] + "_cropped.nii.gz"
        if os.path.exists(cropped_filename):
            print("Found cropped LDCT Mask")
            ldct_LM_nii = nib.load(cropped_filename)
            ldct_LM_np = ldct_LM_nii.get_fdata()
            ldct_LM_np = ldct_LM_np[:, ::-1, :]
        else:
            ldct_LM_nii = nib.load(ldct_LM_path[0])
            ldct_LM_nii_np = np.transpose(ldct_LM_nii.get_fdata(), (2, 1, 0))
            ldct_LM_nii_2 = nib.Nifti1Image(ldct_LM_nii_np, ldct_LM_nii.affine)
            ldct_LM_reshaped = resample_to_output(ldct_LM_nii_2, voxel_sizes=target_voxel_sizes)
            ldct_LM_reshaped_np = np.round(ldct_LM_reshaped.get_fdata(), decimals=0)
            np.clip(ldct_LM_reshaped_np, 0, 2, out=ldct_LM_reshaped_np)
            ldct_LM_reshaped_np = np.flip(ldct_LM_reshaped_np, (0, 2))
            header = ldct_LM_reshaped.header
            cropped_filename = ldct_LM_path[0][:-7] + "_cropped.nii.gz"
            save_nifti_with_header(ldct_LM_reshaped_np, header, cropped_filename)

        # Read PET
        cropped_filename = pet_path[0][:-7] + "_cropped.nii.gz"
        if os.path.exists(cropped_filename):
            print("Found cropped PET")
            pet_nii = nib.load(cropped_filename)
            pet_np = pet_nii.get_fdata()
        else:
            pet_nii = nib.load(pet_path[0])
            pet_reshaped = resample_to_output(pet_nii, voxel_sizes=target_voxel_sizes)
            pet_np = np.transpose(pet_reshaped.get_fdata(), (2, 1, 0))
            pet_np = np.round(pet_np, decimals=0)
            pet_np = np.flip(pet_np, axis=2)
            header = pet_reshaped.header
            cropped_filename = pet_path[0][:-7] + "_cropped.nii.gz"
            save_nifti_with_header(pet_np, header, cropped_filename)

        if False:
            print("Pixels:")
            print("LD CT", ldct_reshaped.header.get_zooms())
            print("LD CT LM", ldct_LM_reshaped.header.get_zooms())
            print("PET ", pet_reshaped.header.get_zooms())
            print("PLANCT", planct_reshaped.header.get_zooms())
            print("Plan Ct LM ", planct_LM_reshaped.header.get_zooms())
            print("ITV", itv_reshaped.header.get_zooms())

        if True:
            print("Shapes:")
            print("LD CT", ldct_np.shape)
            print("LD CT LM", ldct_LM_np.shape)
            print("PET ", pet_np.shape)
            print("PLANCT", planct_np.shape)
            print("Plan Ct LM ", planct_LM_np.shape)
            print("ITV", itv_np.shape)

        return ldct_np, ldct_LM_np, pet_np, planct_np, planct_LM_np, itv_np
