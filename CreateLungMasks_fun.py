import os.path
from lungmask import mask as lungmask_fun
import nibabel as nib
import numpy as np

##DO NOT use ScaleIntensity before creating Lung Masks


def CreateLungMasks(image_tensor,save_path,save_bool):
    # Get Lung mask and save it
    image_np = image_tensor.numpy()
    print("ImageNP shape: ",image_np.shape)
    print('Creating Lung Mask: ')
    lung_path = save_path + '_LungMask.nii.gz'
    if False:#os.path.exists(lung_path):
        print("LM already existing")
        lungmask_nii = nib.load(lung_path)
        lungmask_np = lungmask_nii.get_fdata()
        print("Sum: ",lungmask_np.shape)
        return lungmask_np
    else:
        print("Create new LM")
        #Numpy array support:
        #first axis containing slices
        #second axis with chest to back
        #third axis with right to left
        image_np = np.transpose(image_np,[2,0,1])
        lungmask = lungmask_fun.apply(image_np)  # default model is U-net(R231)
        lungmask = np.transpose(lungmask, [1,2,0])
        image_np = np.transpose(image_np, [1,2,0])
        print("Sum: ", np.sum(lungmask)," Shape: ",lungmask.shape)
        if save_bool:
            print("SAving LM")
            lungmask_ni = nib.Nifti1Image(lungmask,affine=np.eye(4))
            nib.save(lungmask_ni, lung_path)

        return lungmask


def CreateLungMasks_DEPRECATED(CT_fpaths,save_bool):
    # Get Lung mask and save it
    CT_path0 = CT_fpaths[0]
    CT_nii = nib.load(CT_path0)
    for ct in CT_fpaths:
        lung_path = ct[:-7] + '_LungMask.nii.gz'
        print('Creating Lung Mask: ', lung_path)
        input_image = sitk.ReadImage(ct, imageIO='NiftiImageIO')
        lungmask = lungmask_fun.apply(input_image)  # default model is U-net(R231)
        if save_bool:
            lungmask_ni = nib.Nifti1Image(lungmask, CT_nii.affine)
            nib.save(lungmask_ni, lung_path)
    return lungmask
