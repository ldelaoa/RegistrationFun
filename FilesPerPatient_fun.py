import os
import numpy as np


def FilesPerPatient(file_path):
    plan_ct = []
    pet_ct = []
    plan_ct_LM = []
    pet_ct_LM = []
    tolungmask=[]
    itv = []
    pet = []
    
    #print(file_path)
    for root, dirs, files in os.walk(file_path, topdown=False):
        for f in files:
            if ("Thorax" in f or "Ave" in f) and not("LungMask" in f):
                tolungmask.append(os.path.join(file_path,f))
                plan_ct.append(os.path.join(file_path,f))
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and not("LungMask" in f):
                tolungmask.append(os.path.join(file_path,f))
                pet_ct.append(os.path.join(file_path,f))
            
            if ("Thorax" in f or "Ave" in f) and "LungMask" in f:
                tolungmask.append(os.path.join(file_path,f))
                plan_ct_LM.append(os.path.join(file_path,f))
                
            if ("AC_CT_Body" in f or "CT van PET" in f or "CT LD" in f or "AC CT" in f or "AC  CT" in f or "LD CT" in f or "rigide_ct" in f) and "LungMask" in f:
                tolungmask.append(os.path.join(file_path,f))
                pet_ct_LM.append(os.path.join(file_path,f))
                
            if "ITVtumor" in f or "ITV" in f or "GTVtumor" in f or "GTV" in f:
                itv.append(os.path.join(file_path,f))
            
            if "pet" in f.lower():
                pet.append(os.path.join(file_path,f))
                
    
    return plan_ct,plan_ct_LM,pet_ct,pet_ct_LM,pet,itv,tolungmask
