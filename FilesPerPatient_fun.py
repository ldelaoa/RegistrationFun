import os
import numpy as np


def FilesPerPatient(file_path):
    plan_ct = []
    pet_ct = []
    plan_ct_LM = []
    pet_ct_LM = []
    itv = []
    pet = []
    
    #print(file_path)
    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if ("Thorax" in f or "Ave" in f) and not("LungMask" in f) and (not("cropped" in f.lower()) or not("clinic" in f.lower())):
                plan_ct.append(os.path.join(file_path,f))
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and not("LungMask" in f) and (not("cropped" in f.lower()) or not("clinic" in f.lower())):
                pet_ct.append(os.path.join(file_path,f))
            
            if ("Thorax" in f or "Ave" in f) and "LungMask" in f and (not("cropped" in f.lower()) or not("clinic" in f.lower())):
                plan_ct_LM.append(os.path.join(file_path,f))
                
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and "LungMask" in f and (not("cropped" in f.lower()) or not("clinic" in f.lower())):
                pet_ct_LM.append(os.path.join(file_path,f))
                
            if ("ITVtumor" in f or "ITV" in f or "GTVtumor" in f or "GTV" in f) and not("cropped" in f.lower() or "clinic" in f.lower()):
                itv.append(os.path.join(file_path,f))
            
            if "pet" in f.lower() and (not("cropped" in f.lower()) or not("clinic" in f.lower())):
                pet.append(os.path.join(file_path,f))

    data_dicts = [
        {"PlanCT": plan_ct_name,"ITV":itv_name, "LDCT": ldct_name, "PET": pet_name}
        for plan_ct_name, itv_name, ldct_name, pet_name in zip(plan_ct,itv,pet_ct,pet)
    ]

    return plan_ct,pet_ct,data_dicts


def FilesperPatient_Inter_LungCroped(file_path):
    planct_path = []
    itv_path = []
    planct_LM_path = []
    ldct_path = []
    pet_path = []
    ldct_LM_path = []

    # print(file_path)
    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if ("PlanCT" in f) and not ("LungMask" in f) and "cropped" in f.lower() and not("clinic" in f.lower()):
                planct_path.append(os.path.join(file_path, f))
            if ("LDCT" in f) and not ("LungMask" in f) and "cropped" in f.lower() and not("clinic" in f.lower()):
                ldct_path.append(os.path.join(file_path, f))

            if ("PlanCT" in f) and "LungMask" in f and "cropped" in f.lower() and not("clinic" in f.lower()):
                planct_LM_path.append(os.path.join(file_path, f))

            if ("LDCT" in f) and "LungMask" in f and "cropped" in f.lower() and not("clinic" in f.lower()):
                ldct_LM_path.append(os.path.join(file_path, f))

            if ("ITVtumor" in f or "ITV" in f or "GTVtumor" in f or "GTV" in f) and "cropped" in f.lower() and not("clinic" in f.lower()):
                itv_path.append(os.path.join(file_path, f))

            if "pet" in f.lower() and "cropped" in f.lower() and not ("clinic" in f.lower()):
                pet_path.append(os.path.join(file_path, f))

    planct_clinic = []
    itv_clinic = []
    planct_LM_clinic = []
    ldct_clinic = []
    pet_clinic = []
    ldct_LM_clinic = []

    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if "pet" in f.lower() and "clinic" in f.lower() and not("cropped" in f.lower()):
                pet_clinic.append(os.path.join(file_path, f))
            if ("PlanCT" in f) and not ("LungMask" in f) and "clinic" in f.lower() and not("cropped" in f.lower()):
                planct_clinic.append(os.path.join(file_path, f))
            if ("LDCT" in f) and not ("LungMask" in f) and "clinic" in f.lower() and not("cropped" in f.lower()):
                ldct_clinic.append(os.path.join(file_path, f))

            if ("planct" in f.lower()) and "LungMask" in f and "clinic" in f.lower() and not("cropped" in f.lower()):
                planct_LM_clinic.append(os.path.join(file_path, f))

            if ("LDCT" in f) and "LungMask" in f and "clinic" in f.lower() and not("cropped" in f.lower()):
                ldct_LM_clinic.append(os.path.join(file_path, f))

            if ("ITV" in f) and "clinic" in f.lower() and not("cropped" in f.lower()):
                itv_clinic.append(os.path.join(file_path, f))



    data_dicts_intermediate = [{"PlanCT_LungCrop": planct_LungCrop_name,"ITV_LungCrop":itv_LungCrop_name,"PlanCT_LungMask_LungCrop":planCT_LM_LungCrop_name,
                                "LDCT_LungCrop": ldct_LungCrop_name,"PET_LungCrop":pet_LungCrop_name,"LDCT_LungMask_LungCrop":LDCT_LM_LungCrop_name,

                                "PlanCT_Clinic": planct_Clinic_name, "ITV_Clinic": itv_Clinic_name,"PlanCT_LungMask_Clinic": planCT_LM_Clinic_name,
                                "LDCT_Clinic": ldct_Clinic_name, "PET_Clinic": pet_Clinic_name,"LDCT_LungMask_Clinic": LDCT_LM_Clinic_name,

                                }
                               for planct_LungCrop_name,itv_LungCrop_name,planCT_LM_LungCrop_name,ldct_LungCrop_name,pet_LungCrop_name,LDCT_LM_LungCrop_name,
                               planct_Clinic_name,itv_Clinic_name,planCT_LM_Clinic_name,ldct_Clinic_name,pet_Clinic_name,LDCT_LM_Clinic_name in
                               zip(planct_path,itv_path,planct_LM_path,ldct_path,pet_path,ldct_LM_path,planct_clinic,itv_clinic,planct_LM_clinic,ldct_clinic,pet_clinic,ldct_LM_clinic)
                               ]

    return data_dicts_intermediate

