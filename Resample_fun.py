import numpy as np
import nibabel as nib
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    Spacingd,
)
from monai.data import DataLoader, Dataset


def pixDimsfromNifti(nii_path):
    nii_image = nib.load(nii_path)
    original_pixdims = nii_image.header.get_zooms()
    desired_pixdims = np.array([1,1,2])
    spacing_pix = original_pixdims / desired_pixdims

    return spacing_pix


def SpacingAndResampleToMatch(nii_path,image,mask):
    target_pixels = pixDimsfromNifti(nii_path)

    pictionary = [
        {"image": image_name, "mask": mask_name,"pixdims":pix_name}
        for image_name, mask_name,pix_name in zip([image], [mask],[target_pixels])
    ]

    print("Goal PixDims: ",pictionary[0]["pixdims"])
    print("Previous Shapes:",pictionary[0]["image"].shape,pictionary[0]["mask"].shape)

    spacing_transforms = Compose([

        Spacingd(keys=["image"], pixdim=pictionary[0]["pixdims"], mode="bilinear"),
        Spacingd(keys=["mask"], pixdim=[1,1,.5], mode="nearest"),
        # ToTensord(keys=["image","mask"]),
    ])

    check_ds = Dataset(data=pictionary, transform=spacing_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)
    spaced_image, spaced_mask = (batch_data["image"], batch_data["mask"])

    return spaced_image[0],spaced_mask[0]