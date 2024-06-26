import os
import numpy as np



def FilesPerPatient_Registered(file_path):
    ldctClinic_v1 =[]
    ldctClinic_v2 =[]
    ldctLung_v1 = []
    ldctLung_v2 = []
    petClinic_v1 =[]
    petClinic_v2 =[]
    petLung_v1 = []
    petLung_v2 = []
    

    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if ("LDCT" in f) and "Clinic" in f and "v1" in f:
                ldctClinic_v1.append(os.path.join(file_path,f))
            if ("LDCT" in f) and "Clinic" in f and "v2" in f:
                ldctClinic_v2.append(os.path.join(file_path,f))
            if ("LDCT" in f) and "Lung" in f and "v1" in f:
                ldctLung_v1.append(os.path.join(file_path,f))
            if ("LDCT" in f) and "Lung" in f and "v2" in f:
                ldctLung_v2.append(os.path.join(file_path,f))
            
            if ("PET" in f) and "Clinic" in f and "v1" in f:
                petClinic_v1.append(os.path.join(file_path,f))
            if ("PET" in f) and "Clinic" in f and "v2" in f:
                petClinic_v2.append(os.path.join(file_path,f))
            if ("PET" in f) and "Lung" in f and "v1" in f:
                petLung_v1.append(os.path.join(file_path,f))
            if ("PET" in f) and "Lung" in f and "v2" in f:
                petLung_v2.append(os.path.join(file_path,f))

    data_dicts = [
        {"ldctClinic_v1":ldctClinic_v1_name,"ldctLung_v1":ldctLung_v1_name,"petClinic_v1":petClinic_v1_name,"petLung_v1":petLung_v1_name,
         "ldctClinic_v2":ldctClinic_v2_name,"ldctLung_v2":ldctLung_v2_name,"petClinic_v2":petClinic_v2_name,"petLung_v2":petLung_v2_name}
        for ldctClinic_v1_name,ldctLung_v1_name,petClinic_v1_name,petLung_v1_name,
        ldctClinic_v2_name,ldctLung_v2_name,petClinic_v2_name,petLung_v2_name,
        in zip(ldctClinic_v1,ldctLung_v1,petClinic_v1,petLung_v1,ldctClinic_v2,ldctLung_v2,petClinic_v2,petLung_v2)]

    return data_dicts


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
            if ("Thorax" in f or "Ave" in f) and not("LungMask" in f) and (not("cropped" in f.lower()) or not("clinic" in f.lower())) and not"Register" in f:
                plan_ct.append(os.path.join(file_path,f))
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and not("LungMask" in f) and (not("cropped" in f.lower()) or not("clinic" in f.lower())) and not"Register" in f:
                pet_ct.append(os.path.join(file_path,f))
            
            if ("Thorax" in f or "Ave" in f) and "LungMask" in f and (not("cropped" in f.lower()) or not("clinic" in f.lower())) and not"Register" in f:
                plan_ct_LM.append(os.path.join(file_path,f))
                
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and "LungMask" in f and (not("cropped" in f.lower()) or not("clinic" in f.lower())) and not"Register" in f:
                pet_ct_LM.append(os.path.join(file_path,f))
                
            if ("ITVtumor" in f or "ITV" in f or "GTVtumor" in f or "GTV" in f) and not("cropped" in f.lower() or "clinic" in f.lower()) and not"Register" in f:
                itv.append(os.path.join(file_path,f))
            
            if "pet" in f.lower() and (not("cropped" in f.lower()) or not("clinic" in f.lower())) and not"Register" in f:
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

    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if ("PlanCT" in f) and not ("LungMask" in f) and "cropped" in f.lower() and not("clinic" in f.lower()) and not"Register" in f:
                planct_path.append(os.path.join(file_path, f))
            if ("LDCT" in f) and not ("LungMask" in f) and "cropped" in f.lower() and not("clinic" in f.lower()) and not"Register" in f:
                ldct_path.append(os.path.join(file_path, f))

            if ("PlanCT" in f) and "LungMask" in f and "cropped" in f.lower() and not("clinic" in f.lower()) and not"Register" in f:
                planct_LM_path.append(os.path.join(file_path, f))

            if ("LDCT" in f) and "LungMask" in f and "cropped" in f.lower() and not("clinic" in f.lower()) and not"Register" in f:
                ldct_LM_path.append(os.path.join(file_path, f))

            if ("ITVtumor" in f or "ITV" in f or "GTVtumor" in f or "GTV" in f) and "cropped" in f.lower() and not("clinic" in f.lower()) and not"Register" in f:
                itv_path.append(os.path.join(file_path, f))

            if "pet" in f.lower() and "cropped" in f.lower() and not ("clinic" in f.lower()) and not"Register" in f:
                pet_path.append(os.path.join(file_path, f))

    data_dicts_intermediate = [{"PlanCT_LungCrop": planct_LungCrop_name, "ITV_LungCrop": itv_LungCrop_name,
                                "PlanCT_LungMask_LungCrop": planCT_LM_LungCrop_name,
                                "LDCT_LungCrop": ldct_LungCrop_name, "PET_LungCrop": pet_LungCrop_name,
                                "LDCT_LungMask_LungCrop": LDCT_LM_LungCrop_name}
                               for
                               planct_LungCrop_name, itv_LungCrop_name, planCT_LM_LungCrop_name, ldct_LungCrop_name, pet_LungCrop_name, LDCT_LM_LungCrop_name
                               in
                               zip(planct_path, itv_path, planct_LM_path, ldct_path, pet_path, ldct_LM_path)]
    print("Lens:",len(planct_path), len(itv_path), len(planct_LM_path), len(ldct_path), len(pet_path), len(ldct_LM_path))
    return data_dicts_intermediate


def FilesPerPatient_Registered_dynamic(file_path,regist_v):
    ldctLung_v = []
    petLung_v = []

    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if ("LDCT" in f) and "Lung" in f and str(regist_v) in f:
                ldctLung_v.append(os.path.join(file_path, f))
            if ("PET" in f) and "Lung" in f and str(regist_v) in f:
                petLung_v.append(os.path.join(file_path, f))
    data_dicts = [
        {"ldctLung_v": ldctLung_v_name,"petLung_v": petLung_v_name}
        for ldctLung_v_name, petLung_v_name, in zip(ldctLung_v,petLung_v)]
    print("Files Ok:",file_path,regist_v,len(data_dicts))
    return data_dicts
