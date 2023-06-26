import skimage.metrics as metrics
import numpy as np


def similarMetrics(img1,img2,itv):
    img1 =img1.numpy()
    img2 =img2.numpy()
    count,mse,ssim,psnr =(0,0,0,0)
    for i in range(img1.shape[-1]):
        if np.sum(itv[:,:,i])>0:
            mse = mse+ metrics.mean_squared_error(img1[:,:,i], img2[:,:,i])
            ssim = ssim+ metrics.structural_similarity(img1[:,:,i], img2[:,:,i],data_range=img2[:,:,i].max() - img2[:,:,i].min())
            psnr = psnr + metrics.peak_signal_noise_ratio(img1[:,:,i], img2[:,:,i],data_range=img2[:,:,i].max() - img2[:,:,i].min())

            count += 1
    if count > 0 : 
        mse_avg = mse/count
        ssim_avg = ssim/count
        psnr_avg = psnr/count
        return mse_avg,ssim_avg,psnr_avg
    else:
        return 0,0,0

    
