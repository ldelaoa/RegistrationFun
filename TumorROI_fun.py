import numpy as np
import nibabel as nib
from monai.utils.misc import first
from skimage.filters import threshold_multiotsu
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ImageFilterd,
    ForegroundMaskd,
    AsDiscreted,
    MapTransform,
    CropForegroundd,
    SqueezeDimd,
    ToTensord,
)
from monai.data import DataLoader, Dataset
from skimage.morphology import dilation
from skimage.morphology import ball


class BinaryPET_CropCTs(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        print(f"keys to sum: {self.keys}")
    def __call__(self, dictionary):
        dictionary = dict(dictionary)
        pet_cropped_1 = dictionary['BinaryPET']
        pet_cropped_1 = pet_cropped_1.numpy()
        unique_val = np.unique(pet_cropped_1)
        classes_num=3
        if len(unique_val) > classes_num:
            local_otsu = threshold_multiotsu(pet_cropped_1, classes=classes_num)
            otsu_lvl1 = local_otsu[-1]
            print("Otsu is :",otsu_lvl1,"All values: ",local_otsu)
            binary_pet = pet_cropped_1 > otsu_lvl1
        else:
            binary_pet = pet_cropped_1 > 0
            print("no uptake values found, thresh is ZERO")
        dictionary["BinaryPET"] = binary_pet.astype(np.uint8)
        if False:
            print("binary_pet Shape",binary_pet.shape)
            binary_pet = binary_pet.astype(np.uint8)
            dilatedPet = dilation(np.squeeze(binary_pet[0,:,:,:]), ball(8))
            tensor_dilated_pet = binary_pet
            tensor_dilated_pet[0,:,:,:] =  dilatedPet
            dictionary["BinaryPET"] = tensor_dilated_pet
        return dictionary


def TumorROI_fun(imagePET,imageLDCT,imagePlanCT,imageITV,planCT_LM,LDCT_LM,device):
    pictionary = [
        {"PET": imagePET_name, "PlanCT": imagePlanCT_name,"LDCT":imageLDCT_name,"ITV":imageITV_name,"BinaryPET":imageBinaryPet_name,"PlanCT_LM":imagePlanCTLM_name,"LDCT_LM":imageLDCTLM_name}
        for imagePET_name, imagePlanCT_name,imageLDCT_name,imageITV_name,imageBinaryPet_name,imagePlanCTLM_name,imageLDCTLM_name in zip([imagePET], [imagePlanCT],[imageLDCT],[imageITV],[imagePET],[planCT_LM],[LDCT_LM])
    ]

    resample_transforms = Compose([
        EnsureChannelFirstd(keys=["PET", "PlanCT", "LDCT", "ITV", "PlanCT_LM", "LDCT_LM","BinaryPET"]),
        SqueezeDimd(keys=["PET", "PlanCT", "LDCT", "ITV", "PlanCT_LM", "LDCT_LM","BinaryPET"]),
        BinaryPET_CropCTs(keys="BinaryPET"),
        #ImageFilterd(keys="BinaryPET",kernel="elliptical",kernel_size=3),
        CropForegroundd(keys=["PET","PlanCT","LDCT","ITV","PlanCT_LM","LDCT_LM","BinaryPET"],source_key="BinaryPET",k_divisible = 96),
        #ToTensord(keys=["PET","PlanCT","LDCT","ITV","PlanCT_LM","LDCT_LM"]),
        SqueezeDimd(keys=["PET", "PlanCT", "LDCT", "ITV", "PlanCT_LM", "LDCT_LM","BinaryPET"]),

    ])
    check_ds = Dataset(data=pictionary, transform=resample_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
    batch_data = first(check_loader)
    PET_TumorRoi = batch_data["PET"].to(device)
    LDCT_TumorRoi = batch_data["LDCT"].to(device)
    PlanCT_TumorRoi = batch_data["PlanCT"].to(device)
    ITV_TumorRoi = batch_data["ITV"].to(device)
    PlanCTLM_TumorRoi = batch_data["PlanCT_LM"].to(device)
    LDCTLM_TumorRoi = batch_data["LDCT_LM"].to(device)
    BinaryPET_TumorRoi = batch_data["BinaryPET"].to(device)

    print("Shape Tumor ROI",BinaryPET_TumorRoi.detach().cpu().numpy().shape)

    return PET_TumorRoi,LDCT_TumorRoi,PlanCT_TumorRoi,ITV_TumorRoi,PlanCTLM_TumorRoi,LDCTLM_TumorRoi,BinaryPET_TumorRoi