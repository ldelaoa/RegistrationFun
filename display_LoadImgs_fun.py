import matplotlib.pyplot as plt
import numpy as np


def display_LoadImgs(ldct_1,ldct_LM_1,pet_1,planCt_1,planCt_LM_1,itv_1):
    count=0
    for i in range(0,ldct_1.shape[0],20):
        if np.sum(ldct_LM_1[i,:,:])>0:
            plt.subplot(1,2,1),plt.imshow(ldct_1[i,:,:],'gray'),plt.axis('off')
            plt.contour(ldct_LM_1[i,:,:],colors='red')
            plt.subplot(1,2,2),plt.imshow(pet_1[i,:,:],'hot'),plt.axis('off')
            plt.contour(ldct_LM_1[i,:,:],colors='yellow')
            plt.show()
            count+=1
            if count>10:
                break
    print("PlanCT")
    count=0
    for i in range(0,planCt_1.shape[0],5):
        if np.sum(itv_1[i,:,:])>0:
            plt.subplot(1,1,1),plt.imshow(planCt_1[i,:,:],'gray'),plt.axis('off')
            plt.contour(itv_1[i,:,:],colors='yellow')
            plt.contour(planCt_LM_1[i,:,:],colors='red')
            plt.show()
            count+=1
            if count>10:
                break

    return 0

