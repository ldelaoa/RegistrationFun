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
            if ("Thorax" in f or "Ave" in f) and not("LungMask" in f) and not("_cropped" in f):
                plan_ct.append(os.path.join(file_path,f))
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and not("LungMask" in f) and not("_cropped" in f):
                pet_ct.append(os.path.join(file_path,f))
            
            if ("Thorax" in f or "Ave" in f) and "LungMask" in f and not("_cropped" in f):
                plan_ct_LM.append(os.path.join(file_path,f))
                
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and "LungMask" in f and not("_cropped" in f):
                pet_ct_LM.append(os.path.join(file_path,f))
                
            if "ITVtumor" in f or "ITV" in f or "GTVtumor" in f or "GTV" in f:
                itv.append(os.path.join(file_path,f))
            
            if "pet" in f.lower() and not("_cropped" in f):
                pet.append(os.path.join(file_path,f))

    data_dicts = [
        {"PlanCT": plan_ct_name,"ITV":itv_name, "LDCT": ldct_name, "PET": pet_name}
        for plan_ct_name, itv_name, ldct_name, pet_name in zip(plan_ct,itv,pet_ct,pet)
    ]
    print("Dictionary Created")
    return plan_ct,pet_ct,data_dicts
    #return data_dicts