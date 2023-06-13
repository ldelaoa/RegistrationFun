import matplotlib.pyplot as plt
import numpy as np


def displayRegist(PETCT1,PETCT2,PETCT3,ITV,planCT,planCT_LungMask):
    count=0
    for i in range(0,planCT.shape[0],1):
        if np.sum(ITV[i,:,:])>0: 
            plt.subplot(1,4,1),plt.imshow(PETCT1[i,:,:],'gray'),plt.axis('off')
            plt.contour((ITV[i,:,:]),colors='yellow')
            plt.subplot(1,4,2),plt.imshow(PETCT2[i,:,:],'gray'),plt.axis('off')
            plt.contour((ITV[i,:,:]),colors='yellow')
            plt.subplot(1,4,3),plt.imshow(PETCT3[i,:,:],'gray'),plt.axis('off')
            plt.contour((ITV[i,:,:]),colors='yellow')
            plt.subplot(1,4,4),plt.imshow(planCT[i,:,:],'gray'),plt.axis('off')
            plt.contour((ITV[i,:,:]),colors='yellow')
            plt.contour(planCT_LungMask[i,:,:],colors='red')
            plt.tight_layout()
            plt.show()
            count+=1
        if count>50:
            print("break")
            break
    return 0
