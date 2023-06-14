import matplotlib.pyplot as plt
import numpy as np


def display_LoadImgs(planCt_1,planCt_LM_1):
    count=0
    for i in range(0,planCt_1.shape[2],5):
        if np.sum(planCt_LM_1[:,:,i])>0:
            print("Plan", i)
            plt.subplot(1,1,1),plt.imshow(planCt_1[:,:,i],'gray'),plt.axis('off')
            #plt.contour(itv_1[i,:,:],colors='yellow')
            plt.contour(planCt_LM_1[:,:,i],colors='red')
            plt.interactive(False)
            plt.show()
            count+=1
            if count>20:
                break
    return 0

