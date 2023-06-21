import numpy as np
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    CropForegroundd,
    AsDiscreted,
    AddChanneld,
)
from monai.data import DataLoader, Dataset


def CropBinary_monai(image,mask):
    pictionary = [
        {"image": image_name, "mask": mask_name}
        for image_name, mask_name in zip([image], [mask])
    ]

    #print("Minimum value:", np.min(np.argwhere(pictionary[0]["mask"].numpy() != 0), axis=0))
    #print("Maximum value:", np.max(np.argwhere(pictionary[0]["mask"].numpy() != 0), axis=0))

    crop_transforms = Compose([
        AddChanneld(keys=["image","mask"]),
        AsDiscreted(keys=["mask"],treshold=.5),
        CropForegroundd(keys=["image","mask"],source_key="mask",k_divisible=32),
        #ToTensord(keys=["image","mask"]),
        ])

    check_ds = Dataset(data=pictionary, transform=crop_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)
    cropped_image, cropped_mask = (batch_data["image"], batch_data["mask"])

    return cropped_image[0],cropped_mask[0]


def CropBinary_DEPRECATED(binary_image,Middle_bool=False,coords_extra=None,extra=5):

    slices = np.any(binary_image, axis=(1, 2))
    rows = np.any(binary_image, axis=(0, 1))
    cols = np.any(binary_image, axis=(0, 2))
    
    min_slice, max_slice = np.where(slices)[0][[0, -1]]
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    min_slice =min_slice - extra if min_slice > extra else 0
    min_col =min_col- extra if min_col > extra else 0
    min_row =min_row- extra if min_row > extra else 0
    
    max_row = max_row+ extra
    max_slice = max_slice+ (extra//2)
    max_col =max_col+ extra 

    cropped_image = binary_image[min_slice:max_slice+1, min_row:max_row+1, min_col:max_col+1]
    
    return cropped_image,[min_col,min_row,min_slice,max_col,max_row,max_slice]
                        
