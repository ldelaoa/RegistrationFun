import os.path
from nibabel.processing import resample_to_output
import nibabel as nib
import numpy as np


def save_nifti_with_header(data, header, filename):
    img = nib.Nifti1Image(data, None, header=header)
    nib.save(img, filename)


def ReadAndResample(ldct_path,ldct_LM_path,pet_path,planCT_path,planCT_LM_path,itv_path):
    #target_voxel_sizes = (1,1,2)

    #Read Plan CT
    #planct_nii = nib.load(planCT_path[0])
    planct_reshaped= nib.load(planCT_path[0])
    target_voxel_sizes = planct_reshaped.header.get_zooms()
    #planct_reshaped = resample_to_output(planct_nii, voxel_sizes=target_voxel_sizes)
    planct_np = planct_reshaped.get_fdata()
    planct_np = np.transpose(planct_np, (2, 1,0))
    
    #Read Plan CT Lung Mask
    #ct_LM_nii = nib.load(planCT_LM_path[0])
    #ct_LM_nii_np = ct_LM_nii.get_fdata()
    #ct_LM_nii_np = np.transpose(ct_LM_nii_np, (2, 1,0))
    #ct_LM_nii_2 =  nib.Nifti1Image(ct_LM_nii_np,ct_LM_nii.affine)
    #planct_LM_reshaped = resample_to_output(ct_LM_nii_2, voxel_sizes=target_voxel_sizes)
    planct_LM_reshaped = nib.load(planCT_LM_path[0])
    planct_LM_np = planct_LM_reshaped.get_fdata()
    #planct_LM_np = np.transpose(planct_LM_np, (2, 1,0))
    planct_LM_np = np.copy(np.round(planct_LM_np,decimals=0))
    
    #Read Tumor
    #itv_nii =nib.load(itv_path[0])
    #itv_reshaped = resample_to_output(itv_nii, voxel_sizes=target_voxel_sizes)
    itv_reshaped = nib.load(itv_path[0])
    itv_np = itv_reshaped.get_fdata()
    itv_np = np.transpose(itv_np, (2, 1,0))
    itv_np = np.round(itv_np,decimals=0)
    
    #Read LowDose CT
    cropped_filename = ldct_path[0][:-7] + "_cropped.nii.gz"
    if os.path.exists(cropped_filename):
        print("Found cropped LDCT")
        ldct_nii = nib.load(cropped_filename)
        ldct_np = ldct_nii.get_fdata()
    else:
        ldct_nii = nib.load(ldct_path[0])
        ldct_reshaped = resample_to_output(ldct_nii, voxel_sizes=target_voxel_sizes)
        ldct_np = ldct_reshaped.get_fdata()
        ldct_np = np.transpose(ldct_np, (2, 1,0))
        ldct_np = np.rot90(ldct_np,2)
        ldct_np = ldct_np[:, :, ::-1]
        header = ldct_reshaped.header
        cropped_filename = ldct_path[0][:-7] + "_cropped.nii.gz"
        save_nifti_with_header(ldct_np, header, cropped_filename)
    
    #Read LowDose CT Lung Mask
    cropped_filename = ldct_LM_path[0][:-7] + "_cropped.nii.gz"
    if os.path.exists(cropped_filename):
        print("Found cropped LDCT Mask")
        ldct_LM_nii = nib.load(cropped_filename)
        ldct_LM_np = ldct_LM_nii.get_fdata()
        ldct_LM_np = ldct_LM_np[:, ::-1, :]
    else:
        ldct_LM_nii = nib.load(ldct_LM_path[0])
        ldct_LM_nii_np = ldct_LM_nii.get_fdata()
        ldct_LM_nii_np = np.transpose(ldct_LM_nii_np, (2, 1,0))
        ldct_LM_nii_2 = nib.Nifti1Image(ldct_LM_nii_np,ldct_LM_nii.affine)
        ldct_LM_reshaped = resample_to_output(ldct_LM_nii_2, voxel_sizes=target_voxel_sizes)
        ldct_LM_np = ldct_LM_reshaped.get_fdata()
        ldct_LM_np = np.transpose(ldct_LM_np, (2, 1,0))
        ldct_LM_np = np.round(ldct_LM_np,decimals=0)
        ldct_LM_np = np.where(ldct_LM_np <= 0, 0, np.where(ldct_LM_np >= 2, 2, ldct_LM_np))
        ldct_LM_np = ldct_LM_np[::-1, :, :]
        ldct_LM_np = ldct_LM_np[:, :, ::-1]
        header = ldct_LM_reshaped.header
        cropped_filename = ldct_LM_path[0][:-7] + "_cropped.nii.gz"
        save_nifti_with_header(ldct_LM_np, header, cropped_filename)
    
    #Read PET
    cropped_filename = pet_path[0][:-7] + "_cropped.nii.gz"
    if os.path.exists(cropped_filename):
        print("Found cropped PET")
        pet_nii = nib.load(cropped_filename)
        pet_np = pet_nii.get_fdata()
    else:
        pet_nii =nib.load(pet_path[0])
        pet_reshaped = resample_to_output(pet_nii, voxel_sizes=target_voxel_sizes)
        pet_np = pet_reshaped.get_fdata()
        pet_np = np.transpose(pet_np, (2,1,0))
        pet_np = np.round(pet_np,decimals=0)
        pet_np = pet_np[:, :, ::-1]
        header = pet_reshaped.header
        cropped_filename = pet_path[0][:-7] + "_cropped.nii.gz"
        save_nifti_with_header(pet_np, header, cropped_filename)
    
    if False:
        print("Pixels:")
        print("LD CT",ldct_reshaped.header.get_zooms())
        print("LD CT LM",ldct_LM_reshaped.header.get_zooms())
        print("PET ",pet_reshaped.header.get_zooms())
        print("PLANCT",planct_reshaped.header.get_zooms())
        print("Plan Ct LM ",planct_LM_reshaped.header.get_zooms())
        print("ITV",itv_reshaped.header.get_zooms())

    if True:
        print("Shapes:")
        print("LD CT",ldct_np.shape)
        print("LD CT LM",ldct_LM_np.shape)
        print("PET ",pet_np.shape)
        print("PLANCT",planct_np.shape)
        print("Plan Ct LM ",planct_LM_np.shape)
        print("ITV",itv_np.shape)
        
    return ldct_np,ldct_LM_np,pet_np,planct_np,planct_LM_np,itv_np

